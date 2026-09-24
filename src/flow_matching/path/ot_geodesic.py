"""
OT Geodesic ProbPath: Wasserstein geodesic (McCann interpolation) for Flow Matching.

Replaces the linear interpolation path x_t = (1-t)*x0 + t*x1 with the
OT displacement interpolation: x_t = (1-t)*x0 + t*T(x0), where T is the
optimal transport map from source to target distribution.

This ensures the flow trajectory stays on the data manifold rather than
crossing low-density regions.
"""

from torch import Tensor

from src.flow_matching.path.affine import AffineProbPath
from src.flow_matching.path.path import ProbPath
from src.flow_matching.path.path_sample import PathSample
from src.flow_matching.path.scheduler.scheduler import CondOTScheduler, Scheduler
from src.flow_matching.utils import expand_tensor_like
from src.flow_matching.ot.sinkhorn_loss import (
    compute_sinkhorn_plan,
    ot_map_barycentric,
)


class OTGeodesicProbPath(ProbPath):
    """
    Optimal Transport displacement interpolation path.

    Instead of independent linear interpolation per gene dimension:
        x_t = sigma_t * x_0 + alpha_t * x_1   (random pairing)

    We use OT-aligned displacement:
        x_t = sigma_t * x_0 + alpha_t * T(x_0)   (OT-optimal pairing)

    where T(x_0) is the barycentric projection from the Sinkhorn transport plan
    computed between x_0 and x_1 batches.

    This keeps the trajectory on the data manifold and provides more
    meaningful velocity targets for the flow matching model.
    """

    def __init__(
        self,
        scheduler: Scheduler | None = None,
        reg: float = 0.05,
        normalize_cost: bool = True,
    ):
        self.scheduler = scheduler if scheduler is not None else CondOTScheduler()
        self.reg = reg
        self.normalize_cost = normalize_cost

    def sample(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor,
        gene_weights: Tensor | None = None,
    ) -> PathSample:
        """
        Sample from the OT geodesic probability path.

        Computes the Sinkhorn transport plan P* between x_0 and x_1,
        derives the OT map T via barycentric projection, and constructs
        the interpolated state x_t along the Wasserstein geodesic.

        Args:
            x_0: source data (noise), shape (batch_size, n_genes)
            x_1: target data (perturbed cells), shape (batch_size, n_genes)
            t: times in [0,1], shape (batch_size,)

        Returns:
            PathSample with OT-geodesic x_t, dx_t
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        scheduler_output = self.scheduler(t)

        alpha_t = expand_tensor_like(
            input_tensor=scheduler_output.alpha_t, expand_to=x_1
        )
        sigma_t = expand_tensor_like(
            input_tensor=scheduler_output.sigma_t, expand_to=x_1
        )
        d_alpha_t = expand_tensor_like(
            input_tensor=scheduler_output.d_alpha_t, expand_to=x_1
        )
        d_sigma_t = expand_tensor_like(
            input_tensor=scheduler_output.d_sigma_t, expand_to=x_1
        )

        # Compute Sinkhorn transport plan between x_0 and x_1
        P = compute_sinkhorn_plan(
            X=x_0,
            Y=x_1,
            reg=self.reg,
            cost_type="euclidean",
            latent_encoder=None,
            normalize_cost=self.normalize_cost,
            gene_weights=gene_weights,
        )

        # OT map via barycentric projection: T(x_0[i]) = sum_j P_normed[i,j] * x_1[j]
        T_map = ot_map_barycentric(P, x_1)

        # OT displacement interpolation (Wasserstein geodesic)
        # x_t = sigma_t * x_0 + alpha_t * T(x_0)
        # dx_t = d_sigma_t * x_0 + d_alpha_t * T(x_0)
        x_t = sigma_t * x_0 + alpha_t * T_map
        dx_t = d_sigma_t * x_0 + d_alpha_t * T_map

        return PathSample(x_t=x_t, dx_t=dx_t, x_1=T_map, x_0=x_0, t=t)

    def target_to_velocity(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """Inherited from AffineProbPath — same scheduler-based conversion."""
        return _affine_target_to_velocity(self.scheduler, x_1, x_t, t)

    def velocity_to_target(self, velocity: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Recover the endpoint implied by a predicted velocity.

        For this OT path the recovered endpoint is the barycentric OT target
        T(x_0), which is the correct terminal object for distribution matching.
        """
        return _affine_velocity_to_target(self.scheduler, velocity, x_t, t)


def _affine_target_to_velocity(
    scheduler: Scheduler, x_1: Tensor, x_t: Tensor, t: Tensor
) -> Tensor:
    """Convert from x_1 representation to velocity using affine scheduler math."""
    scheduler_output = scheduler(t)
    alpha_t = scheduler_output.alpha_t
    d_alpha_t = scheduler_output.d_alpha_t
    sigma_t = scheduler_output.sigma_t
    d_sigma_t = scheduler_output.d_sigma_t
    a_t = d_sigma_t / sigma_t
    b_t = (d_alpha_t * sigma_t - d_sigma_t * alpha_t) / sigma_t
    return a_t * x_t + b_t * x_1


def _affine_velocity_to_target(
    scheduler: Scheduler, velocity: Tensor, x_t: Tensor, t: Tensor
) -> Tensor:
    scheduler_output = scheduler(t)
    alpha_t = scheduler_output.alpha_t
    d_alpha_t = scheduler_output.d_alpha_t
    sigma_t = scheduler_output.sigma_t
    d_sigma_t = scheduler_output.d_sigma_t
    a_t = -d_sigma_t / (d_alpha_t * sigma_t - d_sigma_t * alpha_t)
    b_t = sigma_t / (d_alpha_t * sigma_t - d_sigma_t * alpha_t)
    a_t = expand_tensor_like(input_tensor=a_t, expand_to=x_t)
    b_t = expand_tensor_like(input_tensor=b_t, expand_to=x_t)
    return a_t * x_t + b_t * velocity
