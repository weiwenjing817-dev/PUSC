
from dataclasses import dataclass
import hashlib
import os

NORMAN_DATASETS = {
    'norman',
    'norman_fast',
    'norman_subset',
    'norman_test',
    'norman_umi_go_filtered',
}


@dataclass
class FlowConfig:
    # Flow model type
    model_type: str = 'hierarchical'

    # Flow Matching specific parameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    ntoken: int = 512
    d_model: int = 512
    lr: float = 1e-5
    steps: int = 5000
    eta_min: float = 1e-7
    devices: str = "1"
    seed: int = 42
    test_only: bool = False
    # Perturbation related parameters
    data_name: str = "combosciplex"
    perturbation_function: str = 'crisper' 
    drug_embedding_mode: str = 'id_mean' # id_mean, mechanism_interaction
    noise_type: str = "Gaussian"
    poisson_alpha: float = 0.8
    poisson_target_sum: int = -1

    print_every: int = 5000
    eval_every: int = 5000
    eval_num_cells: int = 128
    eval_ode_steps: int = 20
    eval_ode_method: str = "rk4"
    eval_seed: int = 12345
    eval_at_start: bool = False
    mode: str = 'predict_y' # predict_y, predict_p
    result_path: str = './result'
    perturbation_fusion_method: str = 'sum' # mlp, sum
    fusion_method: str = 'cross' # cross , concat, add
    attention_backend: str = 'sdpa' # manual, sdpa
    mixed_precision: str = 'no' # no, fp16, bf16
    nlayers: int = 0 # 0 selects the architecture default
    conditioning_mode: str = 'multimodal' # time_only, multimodal
    infer_top_gene: int = 1000
    n_top_genes: int = 5000
    checkpoint_path: str = ''
    gamma: float = 0.0
    split_method: str = 'additive'
    test_condition: str = ''
    test_fraction: float = 0.2
    use_mmd_loss: bool = False
    fold: int = 0
    use_negative_edge: bool = False
    topk: int = 15

    # --- Sinkhorn OT Loss (replaces MMD) ---
    use_ot_loss: bool = False
    ot_reg: float = 0.05
    ot_cost_type: str = "euclidean"  # 'euclidean' or 'latent'
    ot_latent_dim: int = 64
    ot_debiased: bool = True
    ot_loss_weight: float = 1.0
    ot_loss_warmup_steps: int = 0
    ot_loss_start_step: int = 0
    use_weighted_ot_cost: bool = False
    use_weighted_ot_path: bool = False
    use_weighted_ot_loss: bool = False
    ot_gene_weight_strength: float = 1.0

    # --- OT Geodesic Path (replaces linear interpolation) ---
    use_ot_path: bool = False
    ot_path_reg: float = 0.05
    ot_path_normalize_cost: bool = True

    # --- DE-aware supervision / condition regularization ---
    use_de_loss: bool = False
    de_loss_weight: float = 0.1
    de_loss_warmup_steps: int = 0
    de_loss_start_step: int = 0
    de_gene_weight_strength: float = 1.0
    de_topk: int = 0
    perturbation_target_gene_boost: float = 0.0
    perturbation_dropout_prob: float = 0.0
    cell_context_dropout_prob: float = 0.0
    use_cell_type_embedding: bool = False
    disable_cell_type_embedding: bool = False
    n_cell_types: int = 0

    # --- Condition-level delta-response contrastive loss ---
    use_contrast_loss: bool = False
    contrast_loss_weight: float = 0.01
    contrast_temperature: float = 0.1
    contrast_start_step: int = 10000
    contrast_warmup_steps: int = 10000
    contrast_topk: int = 200
    contrast_hard_negative_bonus: float = 0.0
    contrast_amp_weight: float = 0.0
    
    def __post_init__(self):
        if self.attention_backend not in {'manual', 'sdpa'}:
            raise ValueError(f"Invalid attention backend: {self.attention_backend}")
        if self.mixed_precision not in {'no', 'fp16', 'bf16'}:
            raise ValueError(f"Invalid mixed precision mode: {self.mixed_precision}")
        if self.ot_cost_type not in {'euclidean', 'latent'}:
            raise ValueError(f"Invalid OT cost type: {self.ot_cost_type}")
        if self.ot_reg <= 0 or self.ot_path_reg <= 0:
            raise ValueError("OT regularization parameters must be positive")
        nonnegative_fields = (
            'ot_loss_weight',
            'ot_loss_start_step',
            'ot_loss_warmup_steps',
            'ot_gene_weight_strength',
            'de_loss_weight',
            'de_loss_start_step',
            'de_loss_warmup_steps',
            'de_gene_weight_strength',
            'de_topk',
        )
        for field_name in nonnegative_fields:
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.use_weighted_ot_cost and (
            self.use_weighted_ot_path or self.use_weighted_ot_loss
        ):
            raise ValueError(
                "use_weighted_ot_cost is a legacy umbrella flag and cannot be combined "
                "with the split weighted OT flags"
            )
        if self.use_weighted_ot_path and not self.use_ot_path:
            raise ValueError("use_weighted_ot_path requires use_ot_path")
        if self.use_weighted_ot_loss and not self.use_ot_loss:
            raise ValueError("use_weighted_ot_loss requires use_ot_loss")
        if self.use_weighted_ot_cost and not (self.use_ot_path or self.use_ot_loss):
            raise ValueError("use_weighted_ot_cost requires an OT path or OT loss")
        if self.data_name in {'kang', 'GSE96583_Kang_IFNb'}:
            self.data_name = 'kang_ifnb'
        if self.data_name.lower() in {'pc9', 'pc9_combined', 'gse149215_pc9'}:
            self.data_name = 'pc9'
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError('test_fraction must be between 0 and 1')
        if self.data_name in NORMAN_DATASETS:
            # Norman is a single-cell-type K562 CRISPR perturbation dataset.
            # Use three-way condition fusion only: time + cell context + perturbation.
            self.use_cell_type_embedding = False
            self.disable_cell_type_embedding = True
            self.n_cell_types = 0
        if self.data_name == 'kang_ifnb':
            if self.n_cell_types <= 0:
                self.n_cell_types = 8
            if self.disable_cell_type_embedding or self.conditioning_mode == 'time_only':
                self.use_cell_type_embedding = False
            else:
                self.use_cell_type_embedding = True
        if self.disable_cell_type_embedding:
            self.use_cell_type_embedding = False
        if self.conditioning_mode == 'time_only':
            self.use_cell_type_embedding = False
        if self.data_name == 'norman_umi_go_filtered':
            self.n_top_genes = 5054
        if self.data_name == 'norman':
            self.n_top_genes = 5000
        if self.data_name in ['norman_subset', 'norman_fast', 'norman_test']:
            self.n_top_genes = 500
        path = self.make_path()

    def make_path(self):
        ot_tag = ''
        if self.use_ot_loss and self.use_ot_path:
            ot_tag = 'OTboth'
        elif self.use_ot_loss:
            ot_tag = 'OTloss'
        elif self.use_ot_path:
            ot_tag = 'OTpath'
        elif self.use_mmd_loss:
            ot_tag = f'MMD{self.gamma}'
        if self.use_weighted_ot_cost:
            ot_tag += 'W'
        else:
            if self.use_weighted_ot_path:
                ot_tag += 'P'
            if self.use_weighted_ot_loss:
                ot_tag += 'L'
        if self.use_de_loss:
            ot_tag += 'D'
        if self.use_ot_loss or self.use_ot_path or self.use_de_loss:
            aux_signature = (
                self.use_ot_loss,
                self.ot_reg,
                self.ot_cost_type,
                self.ot_debiased,
                self.ot_loss_weight,
                self.ot_loss_start_step,
                self.ot_loss_warmup_steps,
                self.use_ot_path,
                self.ot_path_reg,
                self.ot_path_normalize_cost,
                self.use_weighted_ot_cost,
                self.use_weighted_ot_path,
                self.use_weighted_ot_loss,
                self.ot_gene_weight_strength,
                self.use_de_loss,
                self.de_loss_weight,
                self.de_loss_start_step,
                self.de_loss_warmup_steps,
                self.de_gene_weight_strength,
                self.de_topk,
                self.perturbation_target_gene_boost,
            )
            aux_hash = hashlib.sha1(repr(aux_signature).encode('ascii')).hexdigest()[:8]
            ot_tag += f'A{aux_hash}'
        if self.perturbation_dropout_prob > 0 or self.cell_context_dropout_prob > 0:
            ot_tag += f'CD{self.cell_context_dropout_prob}-PD{self.perturbation_dropout_prob}'
        if self.use_cell_type_embedding:
            ot_tag += f'CT{self.n_cell_types}'
        if self.use_contrast_loss:
            ot_tag += (
                f'CTR{self.contrast_loss_weight}'
                f'T{self.contrast_temperature}'
                f'K{self.contrast_topk}'
            )
            if self.contrast_amp_weight > 0:
                ot_tag += f'AMP{self.contrast_amp_weight}'

        data_tag = self.data_name
        if self.data_name == 'pc9':
            heldout_tag = self.test_condition.replace('+', '_') if self.test_condition else f'fold{self.fold}'
            data_tag = f'{self.data_name}-{self.split_method}-{heldout_tag}'

        exp_name = '-'.join([
            data_tag,
            f'{self.model_type}',
            self.fusion_method,
            self.attention_backend,
            self.mixed_precision,
            f'L{self.nlayers if self.nlayers > 0 else "auto"}',
            self.conditioning_mode,
            self.drug_embedding_mode,
            f'lr{self.lr}',
            f'd{self.d_model}',
            f'ig{self.infer_top_gene}',
            f'mb{self.batch_size}xga{self.gradient_accumulation_steps}',
            ot_tag,
            f's{self.seed}',
            f'f{self.fold}',
            f'tk{self.topk}',
        ])
        return os.path.join(self.result_path, exp_name)
