# config.py
# Chess Edge-Head Transformer training – config dataclasses
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    shards_dir: str = "data/shards"
    train_shards: List[str] = field(default_factory=lambda: [f"shard_{i:02d}.jsonl" for i in range(0, 9)])  # 00..08
    val_shard: str = "shard_09.jsonl"
    num_workers: int = 4                 # Can use multiple workers on Linux
    prefetch_factor: Optional[int] = 2   # Prefetch batches per worker
    pin_memory: bool = True
    persistent_workers: bool = True      # Keep workers alive between epochs
    multiprocessing_context: Optional[str] = None  # None uses 'fork' on Linux (default, most efficient)
    shuffle_seed: int = 42
    max_lines_cache: int = 0             # 0 = stream line offsets only
    batch_size: int = 64
    drop_last: bool = True


@dataclass
class ModelConfig:
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.10
    rope_theta: float = 10000.0
    edge_proj_dim: int = 256
    displacement_bias: bool = True
    # Sequence length from your dataset:
    # 1 [CLS] + 64 squares + 1 [TURN] + 4 castle flags + 1 [EP_* or EP_NONE] + 1 [HMC_*] = 72 tokens
    max_seq_len: int = 72


@dataclass
class LossConfig:
    label_smoothing: float = 0.0
    edge_weight: float = 1.0
    # Make promo head "worth a lot less"
    promo_weight: float = 1.0
    value_weight: float = 1.0
    aux_move_type_weight: float = 0.0  # set >0 only if you add labels
    aux_in_check_weight: float = 0.0   # set >0 only if you add labels
    decov_lambda: float = 1e-3
    orth_lambda: float = 1e-3


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0


@dataclass
class SchedConfig:
    warmup_steps: int = 3000
    t0_steps: int = 80000
    t_mult: float = 1.2
    min_lr_ratio: float = 0.1
    warmup_mode: str = "linear"


@dataclass
class TrainConfig:
    run_dir: str = "runs/chess"
    logs_dir: str = "logs"
    plots_dir: str = "plots"
    seed: int = 42
    train_forever: bool = True
    # precision
    use_amp: bool = False               # FORCE FP32 only
    allow_tf32: bool = False            # disable TF32 for accuracy
    # steps & reporting
    keep_recent_steps: int = 1000
    log_interval: int = 10
    eval_every_epoch: int = 1
    save_every_epoch: int = 1
    # gradient accumulation and tqdm smoothing
    grad_accum_steps: int = 1
    display_smooth_steps: int = 50      # window for smoothing tqdm metrics


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    sched: SchedConfig = field(default_factory=SchedConfig)
    train: TrainConfig = field(default_factory=TrainConfig)