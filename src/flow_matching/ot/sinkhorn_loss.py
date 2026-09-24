"""
Sinkhorn Optimal Transport module for distribution-level matching.

Provides:
- Log-domain Sinkhorn algorithm (numerically stable)
- Transport plan computation
- OT map via barycentric projection
- Sinkhorn OT divergence loss (replaces MMD)
- Cost matrix computation (Euclidean or latent space)
"""

import math

import torch
import torch.nn as nn


def _log_sinkhorn(
    C: torch.Tensor,
    reg: float,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Log-domain Sinkhorn algorithm. Numerically stable using float64 internally.

    Args:
        C: cost matrix [B, B]
        reg: entropy regularization strength
        max_iters: maximum Sinkhorn iterations
        tol: convergence tolerance

    Returns:
        P: transport plan [B, B] (doubly stochastic up to tolerance)
    """
    B = C.shape[0]
    orig_dtype = C.dtype
    orig_device = C.device

    # Use double precision for numerical stability
    C = C.double()
    reg_d = float(reg)

    # Log-domain: f, g are log-potentials
    f = torch.zeros(B, device=orig_device, dtype=torch.float64)
    g = torch.zeros(B, device=orig_device, dtype=torch.float64)

    # -C/reg for the kernel in log domain
    log_K = -C / reg_d

    # Clamp extreme values to prevent overflow
    log_K = torch.clamp(log_K, min=-1e10, max=1e10)

    log_mu = -torch.log(torch.tensor(B, device=orig_device, dtype=torch.float64))
    log_nu = log_mu

    for _it in range(max_iters):
        f_old = f.clone()

        # Update f: f = log_mu - logsumexp_j(log_K_ij + g_j)
        log_Kg = log_K + g.unsqueeze(0)  # [B, B], g_j broadcasted to each row
        f = log_mu - torch.logsumexp(log_Kg, dim=1)  # sum over columns j

        # Update g: g = log_nu - logsumexp_i(log_K_ij + f_i)
        log_Kf = log_K + f.unsqueeze(1)  # [B, B], f_i broadcasted to each column
        g = log_nu - torch.logsumexp(log_Kf, dim=0)  # sum over rows i

        # Check convergence
        if torch.max(torch.abs(f - f_old)) < tol:
            break

    # Build transport plan
    # P[i,j] = exp(log_mu + log_nu + log_K[i,j] + f[i] + g[j])
    log_P = log_mu + log_nu + log_K + f.unsqueeze(1) + g.unsqueeze(0)

    # Clamp for safety
    max_log = torch.max(log_P)
    log_P = log_P - max_log  # shift for numerical stability
    P = torch.exp(log_P)

    # Ensure non-negative
    P = torch.clamp(P, min=0.0)
    P = P / P.sum()

    # Convert back to original dtype
    return P.to(dtype=orig_dtype)


def compute_cost_matrix(
    X: torch.Tensor,
    Y: torch.Tensor,
    cost_type: str = "euclidean",
    latent_encoder: nn.Module | None = None,
    gene_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute pairwise cost matrix between two batches.

    Args:
        X: [B, G] first batch
        Y: [B, G] second batch
        cost_type: 'euclidean' or 'latent'
        latent_encoder: optional encoder for latent-space cost

    Returns:
        C: [B, B] squared cost matrix
    """
    if gene_weights is not None:
        weights = gene_weights.to(device=X.device, dtype=X.dtype).clamp_min(0.0)
        while weights.dim() < X.dim():
            weights = weights.unsqueeze(0)
        scale = torch.sqrt(weights.clamp_min(1e-12))
        X = X * scale
        Y = Y * scale

    if cost_type == "latent" and latent_encoder is not None:
        with torch.set_grad_enabled(latent_encoder.training):
            X_enc = latent_encoder(X)
            Y_enc = latent_encoder(Y)
        C = torch.cdist(X_enc, Y_enc, p=2) ** 2
    else:
        C = torch.cdist(X, Y, p=2) ** 2

    return C


def compute_sinkhorn_plan(
    X: torch.Tensor,
    Y: torch.Tensor,
    reg: float = 0.05,
    cost_type: str = "euclidean",
    latent_encoder: nn.Module | None = None,
    normalize_cost: bool = True,
    gene_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Sinkhorn transport plan between two batches.

    Args:
        X: [B, G] first batch
        Y: [B, G] second batch
        reg: entropy regularization
        cost_type: 'euclidean' or 'latent'
        latent_encoder: optional encoder
        normalize_cost: if True, scale cost by its median for stability

    Returns:
        P: [B, B] transport plan
    """
    C = compute_cost_matrix(
        X,
        Y,
        cost_type=cost_type,
        latent_encoder=latent_encoder,
        gene_weights=gene_weights,
    )

    if normalize_cost:
        # Scale by mean of pairwise distances for numerical stability
        with torch.no_grad():
            mean_cost = C.mean().clamp_min(1e-12)
        C = C / mean_cost

    P = _log_sinkhorn(C, reg=reg)
    return P


def ot_map_barycentric(
    P: torch.Tensor,
    X1: torch.Tensor,
) -> torch.Tensor:
    """
    Compute OT map T(x0) via barycentric projection.

    T(x0_i) = sum_j P_normed[i,j] * x1_j

    Args:
        P: transport plan [B, B]
        X1: target batch [B, G]

    Returns:
        T_map: [B, G] OT-mapped targets corresponding to each x0
    """
    # Normalize rows to sum to 1
    row_sum = P.sum(dim=1, keepdim=True).clamp_min(1e-12)
    P_normed = P / row_sum

    # Barycentric projection: T[i] = sum_j P_normed[i,j] * X1[j]
    T_map = P_normed @ X1

    return T_map


def _target_cost_scale(C_yy: torch.Tensor) -> torch.Tensor:
    batch_size = C_yy.size(0)
    if batch_size > 1:
        off_diagonal_sum = C_yy.sum() - C_yy.diagonal().sum()
        scale = off_diagonal_sum / (batch_size * (batch_size - 1))
    else:
        scale = C_yy.mean()
    return scale.detach().clamp_min(1e-6)


def _regularized_ot_objective(C: torch.Tensor, reg: float) -> torch.Tensor:
    # The optimal plan can be detached by the envelope theorem; this also avoids
    # retaining all Sinkhorn iterations in the model's backward graph.
    with torch.no_grad():
        plan = _log_sinkhorn(C.detach().float(), reg=reg)
    cost = C.float()
    reference_log_prob = -math.log(plan.numel())
    kl = (
        plan
        * (torch.log(plan.clamp_min(torch.finfo(plan.dtype).tiny)) - reference_log_prob)
    ).sum()
    return (plan * cost).sum() + float(reg) * kl


def sinkhorn_ot_loss(
    X_pred: torch.Tensor,
    X_true: torch.Tensor,
    reg: float = 0.05,
    cost_type: str = "euclidean",
    latent_encoder: nn.Module | None = None,
    debiased: bool = True,
    gene_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sinkhorn OT divergence loss between predicted and true distributions.

    This replaces the MMD loss. Computes the entropic Sinkhorn divergence:
        S_eps(X, Y) = OT_eps(X, Y) - 0.5 * (OT_eps(X, X) + OT_eps(Y, Y))

    Each OT objective includes transport cost and KL regularization. Cross and
    self costs share a detached scale derived from the target distribution.

    Args:
        X_pred: [B, G] predicted perturbed cells
        X_true: [B, G] ground truth perturbed cells
        reg: entropy regularization
        cost_type: 'euclidean' or 'latent'
        latent_encoder: optional encoder
        debiased: if True, use debiased Sinkhorn divergence

    Returns:
        loss: scalar Sinkhorn OT loss
    """
    C_cross = compute_cost_matrix(
        X_pred,
        X_true,
        cost_type=cost_type,
        latent_encoder=latent_encoder,
        gene_weights=gene_weights,
    )

    C_yy = compute_cost_matrix(
        X_true,
        X_true,
        cost_type=cost_type,
        latent_encoder=latent_encoder,
        gene_weights=gene_weights,
    )
    cost_scale = _target_cost_scale(C_yy)
    C_cross = C_cross / cost_scale
    C_yy = C_yy / cost_scale
    ot_cost = _regularized_ot_objective(C_cross, reg=reg)

    if debiased:
        # Debiased Sinkhorn divergence:
        # S_eps(X,Y) = OT_eps(X,Y) - 0.5*(OT_eps(X,X) + OT_eps(Y,Y))
        C_xx = compute_cost_matrix(
            X_pred,
            X_pred,
            cost_type=cost_type,
            latent_encoder=latent_encoder,
            gene_weights=gene_weights,
        )
        C_xx = C_xx / cost_scale

        ot_xx = _regularized_ot_objective(C_xx, reg=reg)
        ot_yy = _regularized_ot_objective(C_yy, reg=reg)

        loss = ot_cost - 0.5 * (ot_xx + ot_yy)
    else:
        loss = ot_cost

    return loss


class LatentCostEncoder(nn.Module):
    """
    Small trainable encoder for computing OT cost in latent space.
    Maps sparse gene expression to a compact latent representation.
    """

    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinkhornOTLoss(nn.Module):
    """
    Trainable Sinkhorn OT loss module with optional latent encoder.
    """

    def __init__(
        self,
        reg: float = 0.05,
        cost_type: str = "euclidean",
        latent_dim: int = 64,
        input_dim: int = 1000,
        debiased: bool = True,
    ):
        super().__init__()
        self.reg = reg
        self.cost_type = cost_type
        self.debiased = debiased

        if cost_type == "latent":
            self.latent_encoder = LatentCostEncoder(input_dim=input_dim, latent_dim=latent_dim)
        else:
            self.latent_encoder = None

    def forward(self, X_pred: torch.Tensor, X_true: torch.Tensor) -> torch.Tensor:
        return sinkhorn_ot_loss(
            X_pred=X_pred,
            X_true=X_true,
            reg=self.reg,
            cost_type=self.cost_type,
            latent_encoder=self.latent_encoder,
            debiased=self.debiased,
        )
