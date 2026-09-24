import accelerate
import torch
import torch.nn as nn
import tyro
from config.config_flow import FlowConfig as Config
import torch.nn.functional as F
import time
from torch.utils.data import Dataset, DataLoader
import jax
import random
from src.data_process.data import Data, PerturbationDataset
from src.flow_matching.ot import OTPlanSampler
from src.flow_matching.ot.sinkhorn_loss import sinkhorn_ot_loss
from src.loss.auxiliary import (
    build_gene_weights,
    build_response_gene_scores,
    weighted_ot_loss_enabled,
    weighted_ot_path_enabled,
)
from src.flow_matching.path import AffineProbPath
from src.flow_matching.path.ot_geodesic import OTGeodesicProbPath
from src.flow_matching.solver import ODESolver
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.models.perturbation.moduls import PerturbationEmbedding
import pdb
import tqdm
from src.flow_matching.path.scheduler import CondOTScheduler
import scanpy as sc
import os
from src.data_process.utils import build_generated_anndata

import json
from accelerate import Accelerator,DistributedDataParallelKwargs
import torchdiffeq
from tqdm import trange
import numpy as np
from cell_eval import MetricsEvaluator
import anndata as ad
import pandas as pd
from src.utils.utils import save_checkpoint, load_checkpoint, make_lognorm_poisson_noise, pick_eval_score, process_vocab, set_requires_grad_for_p_only, get_perturbation_emb, set_seed

ot_sampler = OTPlanSampler(method="exact")
contrast_bank = None

def gaussian_kernel(x, y, sigma=1.0):
    beta = 1.0 / (2.0 * sigma**2)
    dist = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-beta * dist)

def mmd_loss(pred, tgt, sigma=1.0):
    xx = gaussian_kernel(pred, pred, sigma).mean(dim=(1))
    yy = gaussian_kernel(tgt, tgt, sigma).mean(dim=(1))
    xy = gaussian_kernel(pred, tgt, sigma).mean(dim=(1))
    return (xx + yy - 2 * xy).mean()

def pairwise_sq_dists(X, Y):
    # X:[m,d], Y:[n,d] -> [m,n]
    return torch.cdist(X, Y, p=2)**2

@torch.no_grad()
def median_sigmas(X, scales=(0.5, 1.0, 2.0, 4.0)):
    Z = X
    D2 = pairwise_sq_dists(Z, Z)
    tri = D2[~torch.eye(D2.size(0), dtype=bool, device=D2.device)]
    m = torch.median(tri).clamp_min(1e-12)          
    s2 = torch.tensor(scales, device=Z.device) * m 
    sigmas = torch.sqrt(s2)                
    return [float(s.item()) for s in sigmas]

def mmd2_unbiased_multi_sigma(X, Y, sigmas):
    """
    """
    m, n = X.size(0), Y.size(0)
    Dxx = pairwise_sq_dists(X, X)   # [m,m]
    Dyy = pairwise_sq_dists(Y, Y)   # [n,n]
    Dxy = pairwise_sq_dists(X, Y)   # [m,n]

    vals = []
    for sigma in sigmas:
        beta = 1.0 / (2.0 * (sigma ** 2) + 1e-12)
        Kxx = torch.exp(-beta * Dxx)
        Kyy = torch.exp(-beta * Dyy)
        Kxy = torch.exp(-beta * Dxy)

        term_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1) + 1e-12)
        term_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1) + 1e-12)
        term_xy = Kxy.mean()  # / (m*n)
        vals.append(term_xx + term_yy - 2.0 * term_xy)

    return torch.stack(vals).mean()

def scheduled_weight(max_weight, iteration, start_step=0, warmup_steps=0):
    if iteration < start_step:
        return 0.0
    if warmup_steps <= 0:
        return max_weight
    progress = min(1.0, float(iteration - start_step + 1) / float(warmup_steps))
    return max_weight * progress


def _matrix_mean_row(X):
    mean = X.mean(axis=0)
    if hasattr(mean, "A1"):
        return mean.A1
    return np.asarray(mean).reshape(-1)


def build_condition_contrast_bank(data_manager, vocab, config):
    if not getattr(config, "use_contrast_loss", False):
        return None

    adata = data_manager.adata_train
    if "perturbation_covariates" not in adata.obs:
        if {"Drug1", "Drug2"}.issubset(adata.obs.columns):
            adata.obs["perturbation_covariates"] = adata.obs[["Drug1", "Drug2"]].apply(
                lambda x: "+".join(x),
                axis=1,
            )
        elif "Drug1" in adata.obs:
            adata.obs["perturbation_covariates"] = adata.obs["Drug1"].astype(str)
        else:
            adata.obs["perturbation_covariates"] = adata.obs["condition"].astype(str)

    control_mask = adata.obs["is_control"].to_numpy()
    if control_mask.sum() == 0:
        raise ValueError("Contrast loss requires control cells in the training data.")
    control_mean = _matrix_mean_row(adata.X[control_mask])

    condition_ids = []
    condition_deltas = []
    condition_names = []
    conditions = sorted(
        condition for condition in adata.obs["perturbation_covariates"].unique()
        if condition not in {"control", "control+control"}
    )

    for condition in conditions:
        parts = condition.split("+")
        try:
            if config.perturbation_function == "crisper":
                ids = vocab.encode(parts)
            else:
                ids = [data_manager.perturbation_dict[part] for part in parts]
        except KeyError:
            continue

        mask = (adata.obs["perturbation_covariates"] == condition).to_numpy()
        if mask.sum() == 0:
            continue
        delta = _matrix_mean_row(adata.X[mask]) - control_mean
        condition_ids.append(ids)
        condition_deltas.append(delta.astype(np.float32))
        condition_names.append(condition)

    if len(condition_ids) < 2:
        print("[Contrast] Disabled: fewer than two training perturbation prototypes.")
        return None

    ids = torch.tensor(condition_ids, dtype=torch.long)
    deltas = torch.tensor(np.stack(condition_deltas), dtype=torch.float32)
    print(
        f"[Contrast] condition prototype bank: {len(condition_names)} conditions, "
        f"{deltas.size(1)} genes, weight={config.contrast_loss_weight}, "
        f"start={config.contrast_start_step}, warmup={config.contrast_warmup_steps}"
    )
    return {"ids": ids, "deltas": deltas, "names": condition_names}


def condition_delta_contrast_loss(
    x1_hat,
    target,
    source,
    perturbation_id,
    input_gene_ids,
    gene_weights,
    config,
    iteration,
):
    global contrast_bank
    if contrast_bank is None or not getattr(config, "use_contrast_loss", False):
        return x1_hat.new_zeros(())

    contrast_weight = scheduled_weight(
        getattr(config, "contrast_loss_weight", 0.01),
        iteration,
        start_step=getattr(config, "contrast_start_step", 0),
        warmup_steps=getattr(config, "contrast_warmup_steps", 0),
    )
    amp_weight = scheduled_weight(
        getattr(config, "contrast_amp_weight", 0.0),
        iteration,
        start_step=getattr(config, "contrast_start_step", 0),
        warmup_steps=getattr(config, "contrast_warmup_steps", 0),
    )
    if contrast_weight <= 0 and amp_weight <= 0:
        return x1_hat.new_zeros(())

    bank_ids = contrast_bank["ids"].to(device=x1_hat.device)
    bank_deltas = contrast_bank["deltas"].to(device=x1_hat.device, dtype=x1_hat.dtype)
    if bank_ids.size(0) < 2:
        return x1_hat.new_zeros(())

    prototypes = bank_deltas[:, input_gene_ids]
    batch_keys, inverse = torch.unique(
        perturbation_id.detach().view(perturbation_id.size(0), -1),
        dim=0,
        return_inverse=True,
    )
    losses = []
    temperature = max(float(getattr(config, "contrast_temperature", 0.1)), 1e-6)
    hard_bonus = float(getattr(config, "contrast_hard_negative_bonus", 0.0))
    topk = int(getattr(config, "contrast_topk", 0))

    for group_idx, key in enumerate(batch_keys):
        positive = torch.where((bank_ids == key.unsqueeze(0)).all(dim=1))[0]
        if positive.numel() == 0:
            continue
        pos_idx = positive[0]
        group_mask = inverse == group_idx
        pred_delta = x1_hat[group_mask].mean(dim=0) - source[group_mask].mean(dim=0)

        proto = prototypes
        pred = pred_delta
        if topk > 0 and topk < pred.numel():
            response = proto[pos_idx].abs()
            keep = torch.zeros_like(response)
            keep[torch.topk(response, k=topk).indices] = 1.0
            pred = pred * keep
            proto = proto * keep.unsqueeze(0)
        else:
            scale = torch.sqrt(gene_weights.clamp_min(1e-12))
            pred = pred * scale
            proto = proto * scale.unsqueeze(0)

        pred = F.normalize(pred.unsqueeze(0), dim=-1, eps=1e-8)
        proto = F.normalize(proto, dim=-1, eps=1e-8)
        logits = (pred @ proto.T).squeeze(0) / temperature
        if hard_bonus > 0:
            shared = (bank_ids == key.unsqueeze(0)).any(dim=1)
            shared[pos_idx] = False
            logits = logits + hard_bonus * shared.to(dtype=logits.dtype)

        label = pos_idx.view(1)
        group_loss = x1_hat.new_zeros(())
        if contrast_weight > 0:
            group_loss = group_loss + contrast_weight * F.cross_entropy(
                logits.unsqueeze(0),
                label,
            )
        if amp_weight > 0:
            pred_norm = pred_delta.norm(p=2)
            true_norm = prototypes[pos_idx].norm(p=2).detach().clamp_min(1e-6)
            amp_loss = F.relu(pred_norm - true_norm).pow(2) / true_norm.pow(2)
            group_loss = group_loss + amp_weight * amp_loss
        losses.append(group_loss)

    if not losses:
        return x1_hat.new_zeros(())
    return torch.stack(losses).mean()


def estimate_terminal_from_velocity(path, velocity, x_t, t):
    if hasattr(path, "velocity_to_target"):
        return path.velocity_to_target(velocity, x_t, t)
    return x_t + velocity * (1 - t).unsqueeze(-1)


def make_eval_save_path(base_path, iteration):
    save_path = os.path.join(base_path, f'iteration_{iteration}')
    if not os.path.exists(save_path):
        return save_path

    suffix = 1
    while True:
        candidate = os.path.join(base_path, f'iteration_{iteration}_repeat{suffix}')
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def build_drug_mechanism_features(data_manager, config):
    if config.perturbation_function == 'crisper' or config.drug_embedding_mode == 'id_mean':
        return None

    if config.drug_embedding_mode != 'mechanism_interaction':
        raise ValueError(f"Unsupported drug embedding mode: {config.drug_embedding_mode}")

    fingerprints = data_manager.adata.uns.get("fingerprints", {})
    if not fingerprints:
        raise ValueError(
            "Drug mechanism encoding requires adata.uns['fingerprints']. "
            "Delete a stale processed.h5ad and rerun preprocessing if necessary."
        )

    pathway_by_drug = {}
    for drug_col, pathway_col in [("Drug1", "pathway1"), ("Drug2", "pathway2")]:
        if drug_col not in data_manager.adata.obs or pathway_col not in data_manager.adata.obs:
            continue
        pairs = data_manager.adata.obs[[drug_col, pathway_col]].drop_duplicates()
        pathway_by_drug.update(dict(zip(pairs[drug_col].astype(str), pairs[pathway_col].astype(str))))

    pathway_names = sorted(set(pathway_by_drug.values()))
    pathway_to_idx = {name: idx for idx, name in enumerate(pathway_names)}
    first_fingerprint = next(iter(fingerprints.values()))
    fingerprint_dim = int(np.asarray(first_fingerprint).size)
    features = torch.zeros(
        config.ntoken,
        fingerprint_dim + len(pathway_names),
        dtype=torch.float32,
    )

    fingerprint_count = 0
    pathway_count = 0
    for drug_name, drug_id in data_manager.perturbation_dict.items():
        if drug_id >= config.ntoken:
            raise ValueError(f"Drug id {drug_id} exceeds ntoken={config.ntoken}")
        if drug_name in fingerprints:
            features[drug_id, :fingerprint_dim] = torch.as_tensor(
                np.asarray(fingerprints[drug_name]),
                dtype=torch.float32,
            )
            fingerprint_count += 1
        if drug_name in pathway_by_drug:
            pathway_idx = pathway_to_idx[pathway_by_drug[drug_name]]
            features[drug_id, fingerprint_dim + pathway_idx] = 1.0
            pathway_count += 1

    print(
        f"[Drug encoder] mechanism features: {fingerprint_dim} fingerprint bits + "
        f"{len(pathway_names)} pathways; coverage fingerprints={fingerprint_count}/"
        f"{len(data_manager.perturbation_dict)}, pathways={pathway_count}/"
        f"{len(data_manager.perturbation_dict)}"
    )
    return features


def train_step(
    source,
    target,
    perturbation_id,
    vf,
    criterion,
    accelerator,
    path,
    noise_type='Poisson',
    mode="predict_y",
    config=None,
    iteration=0,
    cell_type_id=None,
):
    B = source.shape[0]
    device = accelerator.device

    input_gene_ids = torch.randperm(source.shape[-1], device=device)[:config.infer_top_gene]
    source = source[:,input_gene_ids]
    target = target[:,input_gene_ids]
    if cell_type_id is not None:
        cell_type_id = cell_type_id.to(device)
    gene = gene_ids.repeat(B,1).to(device)
    gene_input = gene[:,input_gene_ids]
    use_weighted_path = weighted_ot_path_enabled(config)
    use_weighted_loss = weighted_ot_loss_enabled(config)
    use_response_scores = (
        getattr(config, "use_de_loss", False)
        or use_weighted_path
        or use_weighted_loss
        or getattr(config, "perturbation_target_gene_boost", 0.0) > 0
    )
    if use_response_scores:
        response_scores = build_response_gene_scores(
            source,
            target,
            topk=getattr(config, "de_topk", 0),
        )
    else:
        response_scores = torch.zeros(source.size(1), device=source.device, dtype=source.dtype)
    target_boost = getattr(config, "perturbation_target_gene_boost", 0.0)
    de_gene_weights = build_gene_weights(
        response_scores,
        getattr(config, "de_gene_weight_strength", 1.0),
        gene_input=gene_input,
        perturbation_id=perturbation_id,
        target_boost=target_boost,
    )
    ot_gene_weights = build_gene_weights(
        response_scores,
        getattr(config, "ot_gene_weight_strength", 1.0),
        gene_input=gene_input,
        perturbation_id=perturbation_id,
        target_boost=target_boost,
    )

    if mode=="predict_y":
        t = torch.rand(B, device=device)
        if noise_type=="Gaussian":
            target_noise = torch.randn_like(source)
        elif noise_type=="Poisson":
            target_noise = make_lognorm_poisson_noise(
                target_log=source,
                alpha=getattr(config, "poisson_alpha", 0.8),
                per_cell_L=getattr(config, "poisson_target_sum", 1e4),
            )
        if isinstance(path, OTGeodesicProbPath) and use_weighted_path:
            path_x1 = path.sample(t=t, x_0=target_noise, x_1=target, gene_weights=ot_gene_weights)
        else:
            path_x1 = path.sample(t=t, x_0=target_noise, x_1=target)
        predicted_x_t_velocity = vf(
            gene_input,
            path_x1.x_t,
            path_x1.t,
            source,
            perturbation_id,
            gene_input,
            mode=mode,
            perturbation_dropout_prob=getattr(config, "perturbation_dropout_prob", 0.0),
            cell_context_dropout_prob=getattr(config, "cell_context_dropout_prob", 0.0),
            cell_type_id=cell_type_id,
        )
        cfm_loss = ((predicted_x_t_velocity - path_x1.dx_t)**2).mean()
        loss = cfm_loss
        loss_terms = {"cfm": cfm_loss.detach()}
        x1_hat = estimate_terminal_from_velocity(path, predicted_x_t_velocity, path_x1.x_t, t)

        if getattr(config, "use_de_loss", False):
            de_weight = scheduled_weight(
                getattr(config, "de_loss_weight", 0.1),
                iteration,
                start_step=getattr(config, "de_loss_start_step", 0),
                warmup_steps=getattr(config, "de_loss_warmup_steps", 0),
            )
            if de_weight > 0:
                de_loss = (
                    ((x1_hat - path_x1.x_1.detach()) ** 2) * de_gene_weights.unsqueeze(0)
                ).mean()
                loss = loss + de_loss * de_weight
                loss_terms["de_raw"] = de_loss.detach()
                loss_terms["de_weighted"] = (de_loss.detach() * de_weight)

        if getattr(config, "use_contrast_loss", False):
            contrast_loss = condition_delta_contrast_loss(
                x1_hat=x1_hat,
                target=target,
                source=source,
                perturbation_id=perturbation_id,
                input_gene_ids=input_gene_ids,
                gene_weights=de_gene_weights,
                config=config,
                iteration=iteration,
            )
            loss = loss + contrast_loss

        # --- Distribution-level loss ---
        if config.use_ot_loss:
            # Sinkhorn OT loss between predicted terminal state and ground truth
            # (replaces MMD with geometry-aware Optimal Transport)
            ot_weight = scheduled_weight(
                config.ot_loss_weight,
                iteration,
                start_step=getattr(config, "ot_loss_start_step", 0),
                warmup_steps=getattr(config, "ot_loss_warmup_steps", 0),
            )
            if ot_weight > 0:
                loss_gene_weights = ot_gene_weights if use_weighted_loss else None
                _ot_loss = sinkhorn_ot_loss(
                    X_pred=x1_hat,
                    X_true=target,
                    reg=config.ot_reg,
                    cost_type=config.ot_cost_type,
                    latent_encoder=None,
                    debiased=config.ot_debiased,
                    gene_weights=loss_gene_weights,
                )
                loss = loss + _ot_loss * ot_weight
                loss_terms["ot_raw"] = _ot_loss.detach()
                loss_terms["ot_weighted"] = (_ot_loss.detach() * ot_weight)
        elif config.use_mmd_loss:
            # Original MMD loss (kept for backward compatibility)
            sigmas = median_sigmas(target, scales=(0.5,1.0,2.0,4.0))
            _mmd_loss = mmd2_unbiased_multi_sigma(x1_hat, target, sigmas)
            loss = loss + _mmd_loss * config.gamma

    elif mode=="predict_p":
        t_p = torch.ones(B, device=device)  # Or uniform(0.7,1.0)
        predicted_p_embed = vf(
            gene_input,
            target,
            t_p,
            source,
            perturbation_id,
            gene_input,
            mode=mode,
            cell_type_id=cell_type_id,
        )
        if hasattr(vf, "module"):
            base_vf = vf.module
        else:
            base_vf = vf
        p_embed_gt = base_vf.get_perturbation_emb(perturbation_id=perturbation_id, cell_1=source)
        pred = F.normalize(predicted_p_embed, dim=-1)
        tgt  = F.normalize(p_embed_gt.detach(), dim=-1)
        loss = 1 - (pred * tgt).sum(dim=-1).mean()  # cosine distance
        loss_terms = {"predict_p": loss.detach()}
    
    loss_terms["total"] = loss.detach()
    return loss, loss_terms

@torch.inference_mode()
def test(data_sampler, vf, accelerator,  batch_size=128, path='./',vocab=None,scheme='mse'):
    gene_ids_test = vocab.encode(list(data_sampler.adata.var_names))
    
    gene_ids_test = torch.tensor(gene_ids_test, dtype=torch.long, device=device)
    perturbation_name_list = data_sampler._perturbation_covariates
    control_data = data_sampler.get_control_data()
    all_pred_expressions = [control_data['src_cell_data']]
    obs_perturbation_name_pred = ['control']*control_data['src_cell_data'].shape[0]
    obs_cell_type_pred = list(control_data.get(
        'cell_type_labels',
        np.repeat('', control_data['src_cell_data'].shape[0]),
    ))
    all_target_expressions = [control_data['src_cell_data']]
    obs_perturbation_name_real = ['control']*control_data['src_cell_data'].shape[0]
    obs_cell_type_real = list(control_data.get(
        'cell_type_labels',
        np.repeat('', control_data['src_cell_data'].shape[0]),
    ))
    count = 0
    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(getattr(config, "eval_seed", 12345))
    print('perturbation_name_list:',len(perturbation_name_list))
    for perturbation_name in perturbation_name_list:
        perturbation_data = data_sampler.get_perturbation_data(perturbation_name)
        target = perturbation_data['tgt_cell_data']
        perturbation_id = perturbation_data['condition_id']
        if data_sampler.data_name == 'kang_ifnb':
            control_data_for_condition = data_sampler.get_control_data(perturbation_data.get('cell_type'))
        else:
            control_data_for_condition = control_data
        source = control_data_for_condition['src_cell_data']
        perturbation_id = perturbation_id.to(device)
        cell_type_id = perturbation_data.get('cell_type_id')
        if cell_type_id is not None and cell_type_id.numel() > 0:
            cell_type_id = cell_type_id[0].view(1).to(device)
        if config.perturbation_function == 'crisper':
            perturbation_name_crisper = [inverse_dict[int(p_id)] for p_id in perturbation_id[0].cpu().numpy()]
            perturbation_id = torch.tensor(vocab.encode(perturbation_name_crisper), dtype=torch.long, device=device)
            perturbation_id = perturbation_id.repeat(source.shape[0],1)
        
        N = min(config.eval_num_cells, source.shape[0])
        idx = torch.randperm(source.shape[0], generator=eval_generator)[:N]
        source = source[idx].to(device)
        
        pred_expressions = []
        for i in trange(0, N, batch_size):
            batch_perturbation_id = perturbation_id[0].repeat(source[i:i+batch_size].shape[0],1)
            batch_cell_type_id = None
            if cell_type_id is not None:
                batch_cell_type_id = cell_type_id.repeat(source[i:i+batch_size].shape[0])
            batch_perturbation_id = batch_perturbation_id.to(accelerator.device)
            
            pred_expression = generate_sample(
                wrapped_vf,
                source[i:i+batch_size],
                batch_perturbation_id,
                vf,
                gene_ids=gene_ids_test,
                gene_all=gene_ids_test,
                steps=config.eval_ode_steps,
                method=config.eval_ode_method,
                cell_type_id=batch_cell_type_id,
            )
            pred_expressions.append(pred_expression)
            
        pred_expressions = torch.cat(pred_expressions, dim=0).cpu().numpy()
        all_pred_expressions.append(pred_expressions)
        all_target_expressions.append(target)
        obs_perturbation_name_pred.extend([perturbation_name] * pred_expressions.shape[0])
        obs_perturbation_name_real.extend([perturbation_name] * target.shape[0])
        cell_type_labels = perturbation_data.get(
            'cell_type_labels',
            np.repeat('', target.shape[0]),
        )
        obs_cell_type_pred.extend([perturbation_data.get('cell_type', '')] * pred_expressions.shape[0])
        obs_cell_type_real.extend(cell_type_labels)
        # count += 1
        # if count > 3:
        #     break

    all_pred_expressions = np.concatenate(all_pred_expressions, axis=0)
    all_target_expressions = np.concatenate(all_target_expressions, axis=0)
    obs_pred = pd.DataFrame({
        'perturbation': obs_perturbation_name_pred,
        'cell_type': obs_cell_type_pred,
    })
    obs_real = pd.DataFrame({
        'perturbation': obs_perturbation_name_real,
        'cell_type': obs_cell_type_real,
    })
    # Preserve feature identity for downstream biological validation. Older
    # exports omitted ``var`` and AnnData silently used positional names.
    output_var = pd.DataFrame(index=data_sampler.adata.var_names.astype(str).copy())
    pred = ad.AnnData(X=all_pred_expressions, obs=obs_pred, var=output_var.copy())
    real = ad.AnnData(X=all_target_expressions, obs=obs_real, var=output_var.copy())
    

    eval_score = None
    if accelerator.is_main_process:
        evaluator = MetricsEvaluator(
            adata_pred=pred,
            adata_real=real,
            control_pert="control",
            pert_col="perturbation",
            num_threads=32,
        )
        (results, agg_results) = evaluator.compute()
        
        results.write_csv(os.path.join(path, 'results.csv'))
        agg_results.write_csv(os.path.join(path, 'agg_results.csv'))
        pred.write_h5ad(os.path.join(path, 'pred.h5ad'))
        real.write_h5ad(os.path.join(path, 'real.h5ad'))

        eval_score = pick_eval_score(agg_results, scheme)
        print(f"Current evaluation score: {eval_score:.4f}")
    
    return eval_score

def wrapped_vf(target,t,source,perturbation_id,vf,gene_ids, gene_all, cell_type_id=None):
    
    gene = gene_ids.repeat(source.shape[0],1).to(device)
    predicted_x_t_velocity = vf(gene,target,t,source,perturbation_id,gene_all, cell_type_id=cell_type_id)
    
    return predicted_x_t_velocity

@torch.no_grad()
def generate_sample(
    wrapped_vf,
    source,
    condition_vec=None,
    vf=None,
    gene_ids=None,
    gene_all=None,
    steps=20,
    method="rk4",
    cell_type_id=None,
):
    n_genes = source.shape[1]

    noise_type = config.noise_type
    if noise_type=="Gaussian":
        target_noise = torch.randn(source.shape[0], n_genes, device=source.device)
    elif noise_type=="Poisson":
        target_noise = make_lognorm_poisson_noise(
            target_log=source,
            alpha=getattr(config, "poisson_alpha", 0.8),
            per_cell_L=getattr(config, "poisson_target_sum", 1e4),
        )
        
    traj = torchdiffeq.odeint(lambda t,x: wrapped_vf(x,t,source,condition_vec,vf,gene_ids,gene_all,cell_type_id),
                              target_noise,
                              torch.linspace(0,1,steps).to(source.device),
                              atol=1e-4,
                              rtol=1e-4,
                              method=method)
    # t = torch.linspace(0,1,steps).to(source.device)
    # traj = [target_noise + 0.8*wrapped_vf(target_noise,t,source,condition_vec,vf,gene_ids,gene_all)]
    
    return torch.clamp(traj[-1], min=0)
    
if __name__ == "__main__":
    config = tyro.cli(Config)
    set_seed(config.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=None if config.mixed_precision == 'no' else config.mixed_precision,
        kwargs_handlers=[ddp_kwargs]
    )
    if accelerator.is_main_process:
        print(config)
        print(
            f"[Batch] micro={config.batch_size}, accumulation="
            f"{config.gradient_accumulation_steps}, effective_per_device="
            f"{config.batch_size * config.gradient_accumulation_steps}"
        )
        save_path = config.make_path()
        os.makedirs(save_path, exist_ok=True)
    device = accelerator.device

    # ---- Path selection: OT Geodesic or Linear ----
    if config.use_ot_path:
        path = OTGeodesicProbPath(
            scheduler=CondOTScheduler(),
            reg=config.ot_path_reg,
            normalize_cost=config.ot_path_normalize_cost,
        )
        if accelerator.is_main_process:
            print(f"[Path] Using OT Geodesic path (reg={config.ot_path_reg})")
    else:
        path = AffineProbPath(scheduler=CondOTScheduler())
        if accelerator.is_main_process:
            print("[Path] Using Linear interpolation path")

    if config.use_ot_loss:
        if accelerator.is_main_process:
            print(
                f"[Loss] Using Sinkhorn OT loss (reg={config.ot_reg}, "
                f"cost={config.ot_cost_type}, weight={config.ot_loss_weight}, "
                f"start={config.ot_loss_start_step}, warmup={config.ot_loss_warmup_steps})"
            )
    elif config.use_mmd_loss:
        if accelerator.is_main_process:
            print(f"[Loss] Using MMD loss (gamma={config.gamma})")
    else:
        if accelerator.is_main_process:
            print("[Loss] Using CFM loss only")
    if accelerator.is_main_process and config.use_de_loss:
        print(
            f"[Loss] Using DE endpoint loss (weight={config.de_loss_weight}, "
            f"start={config.de_loss_start_step}, warmup={config.de_loss_warmup_steps})"
        )
    if accelerator.is_main_process and (
        weighted_ot_path_enabled(config) or weighted_ot_loss_enabled(config)
    ):
        print(
            f"[OT gene weights] path={weighted_ot_path_enabled(config)}, "
            f"loss={weighted_ot_loss_enabled(config)}, "
            f"strength={config.ot_gene_weight_strength}"
        )
    # ---- End path selection ----
    
    data_manager = Data('./data')

    data_manager.load_data(config.data_name)
    data_manager.process_data(
        n_top_genes=config.n_top_genes,
        infer_top_gene=config.infer_top_gene,
        split_method=config.split_method,
        fold=config.fold,
        test_condition=config.test_condition,
        test_fraction=config.test_fraction,
        drug_embedding_mode=config.drug_embedding_mode,
        use_negative_edge=config.use_negative_edge,
        k=config.topk,
    )
    if config.data_name in ['kang_ifnb', 'kang', 'GSE96583_Kang_IFNb']:
        config.n_cell_types = getattr(data_manager, 'n_cell_types', config.n_cell_types)
        if config.disable_cell_type_embedding or config.conditioning_mode == 'time_only':
            config.use_cell_type_embedding = False
    train_sampler, valid_sampler, test_dl = data_manager.load_flow_data(batch_size=config.batch_size)
    
    train_dataset = PerturbationDataset(train_sampler, config.batch_size)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True)  # batch_size=1 因为每个getitem本身就是一个batch
    mask_path = data_manager.mask_path
    drug_mechanism_features = build_drug_mechanism_features(data_manager, config)
    vf = instantiate_model(config.model_type,
                           ntoken = config.ntoken,
                           d_model = config.d_model,
                           d_perturbation = config.d_model,
                           fusion_method = config.fusion_method,
                           attention_backend = config.attention_backend,
                           nlayers = config.nlayers,
                           conditioning_mode = config.conditioning_mode,
                           perturbation_function = config.perturbation_function,
                           drug_embedding_mode = config.drug_embedding_mode,
                           drug_mechanism_features = drug_mechanism_features,
                           use_cell_type_embedding = config.use_cell_type_embedding,
                           n_cell_types = config.n_cell_types,
                           mask_path = mask_path
                           )
    
    model_path = config.make_path()

    vocab = process_vocab(data_manager, config)

    gene_ids = vocab.encode(list(data_manager.adata.var_names))
    
    gene_ids = torch.tensor(gene_ids, dtype=torch.long, device=device)
    contrast_bank = build_condition_contrast_bank(data_manager, vocab, config)
    if contrast_bank is not None:
        contrast_bank["ids"] = contrast_bank["ids"].to(device)
        contrast_bank["deltas"] = contrast_bank["deltas"].to(device)
    
    save_path = config.make_path()
    if accelerator.is_main_process:
        with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as config_file:
            json.dump(vars(config), config_file, indent=2, sort_keys=True)
    best_loss = float('inf')
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(vf.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps, eta_min=config.eta_min)
    
    start_iteration = 0
    if config.checkpoint_path != '':
        loaded_iteration, _ = load_checkpoint(
            config.checkpoint_path,
            vf,
            optimizer,
            scheduler,
            scheduler_total_steps=config.steps,
        )
        start_iteration = loaded_iteration + 1
    vf = accelerator.prepare(vf)
    optimizer, scheduler, dataloader = accelerator.prepare(optimizer,scheduler,dataloader)
    inverse_dict = {v: str(k) for k, v in data_manager.perturbation_dict.items()}
    pbar = tqdm.tqdm(total=config.steps, initial=start_iteration)
    iteration = start_iteration
    optimizer.zero_grad(set_to_none=True)
    while iteration < config.steps:
        for batch_data in dataloader:
            
            source = batch_data['src_cell_data'].squeeze(0)
            target = batch_data['tgt_cell_data'].squeeze(0)
            perturbation_id = batch_data['condition_id'].squeeze(0).to(device)
            cell_type_id = batch_data.get('cell_type_id')
            if cell_type_id is not None:
                cell_type_id = cell_type_id.squeeze(0).to(device)
            if config.perturbation_function == 'crisper':
                perturbation_name = [inverse_dict[int(p_id)] for p_id in perturbation_id[0].cpu().numpy()]
                perturbation_id = torch.tensor(vocab.encode(perturbation_name), dtype=torch.long, device=device)
                perturbation_id = perturbation_id.repeat(source.shape[0],1)
            
            
            with accelerator.accumulate(vf):
                set_requires_grad_for_p_only(vf, p_only=config.mode)
                loss, loss_terms = train_step(
                    source, target, perturbation_id, vf, criterion, accelerator,
                    path, noise_type=config.noise_type, mode=config.mode,
                    config=config, iteration=iteration, cell_type_id=cell_type_id,
                )
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            completed_step = iteration + 1
            if (
                accelerator.is_main_process
                and config.print_every > 0
                and completed_step % config.print_every == 0
            ):
                summary = ", ".join(
                    f"{name}={float(value):.6f}" for name, value in loss_terms.items()
                )
                print(f"[Loss terms] step={completed_step}: {summary}")
            should_eval = (
                completed_step % config.eval_every == 0
                or completed_step == config.steps
                or (iteration == 0 and config.eval_at_start)
            )
            if should_eval:
                save_path_ = make_eval_save_path(save_path, completed_step)
                os.makedirs(save_path_, exist_ok=True)
                if accelerator.is_main_process:
                    print(f"saving step {completed_step}'s checkpoint...")
                    
                    save_checkpoint(
                        model=accelerator.unwrap_model(vf), 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        iteration=iteration, 
                        eval_score=None,  # 不需要评估分数
                        save_path=save_path_, 
                        is_best=False
                    )
                eval_score = test(valid_sampler, vf, accelerator, batch_size=config.batch_size, path=save_path_,vocab=vocab)
                
            accelerator.wait_for_everyone()
            
            pbar.update(1)
            pbar.set_description(f'loss: {loss.item():.4f}, iteration: {iteration}')
            iteration += 1
            
            
