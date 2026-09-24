import torch


def build_response_gene_scores(source, target, topk=0):
    response = (target.mean(dim=0) - source.mean(dim=0)).abs()
    if 0 < topk < response.numel():
        selected_values, selected_indices = torch.topk(response, k=topk)
        scores = torch.zeros_like(response)
        scores[selected_indices] = selected_values / selected_values.mean().clamp_min(1e-6)
        return scores
    return response / response.mean().clamp_min(1e-6)


def build_gene_weights(
    response_scores,
    strength,
    gene_input=None,
    perturbation_id=None,
    target_boost=0.0,
):
    weights = torch.ones_like(response_scores) + float(strength) * response_scores
    if target_boost > 0 and gene_input is not None and perturbation_id is not None:
        pert_ids = perturbation_id.reshape(-1).to(device=gene_input.device)
        target_mask = torch.isin(gene_input[0], pert_ids)
        weights = weights + target_boost * target_mask.to(dtype=weights.dtype)
    return weights / weights.mean().clamp_min(1e-6)


def weighted_ot_path_enabled(config):
    return bool(
        getattr(config, "use_weighted_ot_cost", False)
        or getattr(config, "use_weighted_ot_path", False)
    )


def weighted_ot_loss_enabled(config):
    return bool(
        getattr(config, "use_weighted_ot_cost", False)
        or getattr(config, "use_weighted_ot_loss", False)
    )
