from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cell_eval import MetricsEvaluator
from src.data_process.data import Data


CONTROL = "control"


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def mean_row(x) -> np.ndarray:
    mean = x.mean(axis=0)
    if hasattr(mean, "A1"):
        return mean.A1.astype(np.float32)
    return np.asarray(mean, dtype=np.float32).reshape(-1)


def normalize_token(token: str) -> str:
    token = str(token)
    return CONTROL if token == "ctrl" else token


def base_condition(condition: str) -> str:
    condition = str(condition)
    if "|" in condition:
        condition = condition.split("|", 1)[0]
    return condition.replace("ctrl", CONTROL)


def perturbation_parts(condition: str) -> list[str]:
    parts = [normalize_token(part) for part in base_condition(condition).split("+")]
    return [part for part in parts if part and part != CONTROL]


def condition_column(adata) -> str:
    if "perturbation_covariates" in adata.obs:
        return "perturbation_covariates"
    if {"Drug1", "Drug2"}.issubset(adata.obs.columns):
        adata.obs["perturbation_covariates"] = adata.obs[["Drug1", "Drug2"]].apply(
            lambda row: "+".join(row.astype(str)),
            axis=1,
        )
        return "perturbation_covariates"
    if "condition" in adata.obs:
        return "condition"
    raise ValueError("Cannot infer perturbation condition column.")


class MeanPerturbationBaseline:
    def __init__(self, train_adata, eval_genes, mode: str):
        self.mode = "condition_mean" if mode == "linear" else mode
        self.eval_genes = list(eval_genes)
        self.train = train_adata[:, self.eval_genes]
        self.cond_col = condition_column(self.train)
        self.has_cell_type = "cell_type" in self.train.obs
        self.fallbacks = Counter()

        self.global_control_mean = self._control_mean(cell_type=None)
        self.condition_delta = {}
        self.component_delta = {}
        self._fit_exact_condition_deltas()
        self._fit_component_deltas()

    def _mask(self, condition: str | None = None, cell_type: str | None = None):
        obs = self.train.obs
        mask = np.ones(self.train.n_obs, dtype=bool)
        if condition is not None:
            mask &= obs[self.cond_col].astype(str).to_numpy() == condition
        if cell_type is not None and self.has_cell_type:
            mask &= obs["cell_type"].astype(str).to_numpy() == str(cell_type)
        return mask

    def _control_mask(self, cell_type: str | None = None):
        mask = self.train.obs["is_control"].to_numpy(dtype=bool).copy()
        if cell_type is not None and self.has_cell_type:
            mask &= self.train.obs["cell_type"].astype(str).to_numpy() == str(cell_type)
        return mask

    def _control_mean(self, cell_type: str | None):
        mask = self._control_mask(cell_type)
        if mask.sum() == 0:
            if cell_type is not None and hasattr(self, "global_control_mean"):
                self.fallbacks[f"missing_cell_type_control:{cell_type}"] += 1
                return self.global_control_mean
            if cell_type is not None:
                return self._control_mean(None)
            raise ValueError("Training data has no control cells.")
        return mean_row(self.train.X[mask])

    def _cell_types(self) -> list[str | None]:
        if not self.has_cell_type:
            return [None]
        values = sorted(self.train.obs["cell_type"].astype(str).unique())
        return [None, *values]

    def _fit_exact_condition_deltas(self) -> None:
        obs = self.train.obs
        non_control = ~obs["is_control"].to_numpy(dtype=bool)
        conditions = sorted(obs.loc[non_control, self.cond_col].astype(str).unique())
        for cell_type in self._cell_types():
            control_mean = self._control_mean(cell_type)
            for condition in conditions:
                mask = self._mask(condition=condition, cell_type=cell_type) & non_control
                if mask.sum() == 0:
                    continue
                key = (base_condition(condition), cell_type)
                self.condition_delta[key] = mean_row(self.train.X[mask]) - control_mean

    def _fit_component_deltas(self) -> None:
        buckets = defaultdict(list)
        for (condition, cell_type), delta in self.condition_delta.items():
            parts = perturbation_parts(condition)
            if len(parts) == 1:
                buckets[(parts[0], cell_type)].append(delta)
        for key, values in buckets.items():
            self.component_delta[key] = np.stack(values, axis=0).mean(axis=0)

    def _lookup(self, mapping, key: str, cell_type: str | None):
        if cell_type is not None and (key, cell_type) in mapping:
            return mapping[(key, cell_type)]
        if (key, None) in mapping:
            return mapping[(key, None)]
        return None

    def delta_for(self, condition: str, cell_type: str | None = None) -> np.ndarray:
        if self.mode == "control":
            return np.zeros_like(self.global_control_mean)

        condition = base_condition(condition)
        if self.mode == "condition_mean":
            delta = self._lookup(self.condition_delta, condition, cell_type)
            if delta is None:
                self.fallbacks["condition_mean_to_control"] += 1
                return np.zeros_like(self.global_control_mean)
            return delta

        if self.mode != "additive":
            raise ValueError(f"Unknown baseline mode: {self.mode}")

        parts = perturbation_parts(condition)
        deltas = []
        for part in parts:
            delta = self._lookup(self.component_delta, part, cell_type)
            if delta is None:
                self.fallbacks[f"missing_component:{part}"] += 1
            else:
                deltas.append(delta)

        if deltas:
            return np.stack(deltas, axis=0).sum(axis=0)

        exact = self._lookup(self.condition_delta, condition, cell_type)
        if exact is not None:
            self.fallbacks["additive_to_exact_condition"] += 1
            return exact

        self.fallbacks["additive_to_control"] += 1
        return np.zeros_like(self.global_control_mean)

    def predict(self, source: np.ndarray, condition: str, cell_type: str | None = None) -> np.ndarray:
        delta = self.delta_for(condition, cell_type=cell_type)
        return source.astype(np.float32, copy=False) + delta.reshape(1, -1)


def build_output_path(args, baseline_name: str | None = None) -> Path:
    baseline_name = baseline_name or args.baseline
    heldout = args.test_condition.replace("+", "_") if args.test_condition else f"fold{args.fold}"
    name = (
        f"{args.data_name}_{baseline_name}_{args.split_method}_{heldout}_"
        f"hvg{args.n_top_genes}_eval{args.infer_top_gene}_seed{args.seed}"
    )
    return Path(args.result_path) / name


def evaluate_baseline(data_manager, test_sampler, baseline, args, save_path: Path) -> None:
    rng = np.random.default_rng(args.seed)
    control_data = test_sampler.get_control_data()

    all_pred = [as_numpy(control_data["src_cell_data"]).astype(np.float32)]
    all_real = [as_numpy(control_data["src_cell_data"]).astype(np.float32)]
    obs_pred_pert = [CONTROL] * all_pred[0].shape[0]
    obs_real_pert = [CONTROL] * all_real[0].shape[0]
    obs_pred_cell_type = list(
        control_data.get("cell_type_labels", np.repeat("", all_pred[0].shape[0]))
    )
    obs_real_cell_type = list(
        control_data.get("cell_type_labels", np.repeat("", all_real[0].shape[0]))
    )

    for condition in test_sampler._perturbation_covariates:
        pert_data = test_sampler.get_perturbation_data(str(condition))
        target = as_numpy(pert_data["tgt_cell_data"]).astype(np.float32)
        cell_type = pert_data.get("cell_type")

        if getattr(test_sampler, "data_name", "") == "kang_ifnb":
            control_for_condition = test_sampler.get_control_data(cell_type)
        else:
            control_for_condition = control_data

        source = as_numpy(control_for_condition["src_cell_data"]).astype(np.float32)
        n_source = source.shape[0]
        if args.eval_num_cells > 0:
            n_eval = min(args.eval_num_cells, n_source)
            source = source[rng.permutation(n_source)[:n_eval]]

        pred = baseline.predict(source, str(condition), cell_type=cell_type)
        if args.clip_nonnegative:
            pred = np.clip(pred, a_min=0.0, a_max=None)

        all_pred.append(pred)
        all_real.append(target)
        obs_pred_pert.extend([str(condition)] * pred.shape[0])
        obs_real_pert.extend([str(condition)] * target.shape[0])
        obs_pred_cell_type.extend([cell_type or ""] * pred.shape[0])
        obs_real_cell_type.extend(
            list(pert_data.get("cell_type_labels", np.repeat("", target.shape[0])))
        )

    pred_adata = ad.AnnData(
        X=np.concatenate(all_pred, axis=0),
        obs=pd.DataFrame({"perturbation": obs_pred_pert, "cell_type": obs_pred_cell_type}),
        var=pd.DataFrame(index=test_sampler.adata.var_names),
    )
    real_adata = ad.AnnData(
        X=np.concatenate(all_real, axis=0),
        obs=pd.DataFrame({"perturbation": obs_real_pert, "cell_type": obs_real_cell_type}),
        var=pd.DataFrame(index=test_sampler.adata.var_names),
    )

    evaluator = MetricsEvaluator(
        adata_pred=pred_adata,
        adata_real=real_adata,
        control_pert=CONTROL,
        pert_col="perturbation",
        num_threads=args.num_threads,
    )
    results, agg_results = evaluator.compute()

    save_path.mkdir(parents=True, exist_ok=True)
    results.write_csv(str(save_path / "results.csv"))
    agg_results.write_csv(str(save_path / "agg_results.csv"))
    pred_adata.write_h5ad(save_path / "pred.h5ad")
    real_adata.write_h5ad(save_path / "real.h5ad")

    with open(save_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    with open(save_path / "fallbacks.json", "w", encoding="utf-8") as f:
        json.dump(dict(baseline.fallbacks), f, indent=2, sort_keys=True)

    print(f"Saved mean baseline outputs to: {save_path}")
    print(agg_results.to_pandas().round(5))
    if baseline.fallbacks:
        print(f"Fallbacks: {dict(baseline.fallbacks)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control / additive / linear mean baseline for perturbation prediction."
    )
    parser.add_argument(
        "--baseline",
        default="additive",
        help=(
            "One baseline, a comma-separated list, or 'all'. "
            "Available: control, additive, condition_mean, linear."
        ),
    )
    parser.add_argument("--data_name", default="norman")
    parser.add_argument("--n_top_genes", type=int, default=5000)
    parser.add_argument("--infer_top_gene", type=int, default=1000)
    parser.add_argument("--split_method", default="additive")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--test_condition", default="")
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--drug_embedding_mode", default="id_mean")
    parser.add_argument("--result_path", default="./result/mean_baselines")
    parser.add_argument("--eval_num_cells", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--clip_nonnegative", action="store_true", default=True)
    parser.add_argument("--no_clip_nonnegative", dest="clip_nonnegative", action="store_false")
    return parser.parse_args()


def selected_baselines(value: str) -> list[str]:
    available = ["control", "additive", "condition_mean", "linear"]
    if value == "all":
        return available
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names).difference(available))
    if unknown:
        raise ValueError(f"Unknown baseline(s): {unknown}. Available: {available}")
    return names


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_manager = Data("./data")
    data_manager.load_data(args.data_name)
    data_manager.process_data(
        n_top_genes=args.n_top_genes,
        infer_top_gene=args.infer_top_gene,
        split_method=args.split_method,
        fold=args.fold,
        test_condition=args.test_condition,
        test_fraction=args.test_fraction,
        drug_embedding_mode=args.drug_embedding_mode,
        build_mask=False,
    )
    train_sampler, test_sampler, _ = data_manager.load_flow_data(batch_size=args.eval_num_cells)

    for baseline_name in selected_baselines(args.baseline):
        print(f"\n[Mean baseline] running {baseline_name}")
        baseline = MeanPerturbationBaseline(
            train_adata=train_sampler.adata,
            eval_genes=test_sampler.adata.var_names,
            mode=baseline_name,
        )
        evaluate_baseline(
            data_manager=data_manager,
            test_sampler=test_sampler,
            baseline=baseline,
            args=args,
            save_path=build_output_path(args, baseline_name),
        )


if __name__ == "__main__":
    main()
