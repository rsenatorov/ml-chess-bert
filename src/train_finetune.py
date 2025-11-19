# train_finetune.py
# Finetuning script with constant low LR, loads pretrained checkpoint
# SPECIAL: Combines ALL base shards (00-09) with finetune shards (00-09) to prevent catastrophic forgetting
# All outputs isolated with "finetune" naming to avoid overwriting base training

import json
import time
import random
import faulthandler
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import Config
from logger import get_logger, CSVLogger
from utils import set_seed, RingBuffer, format_seconds, get_memory_stats
from data_loader import make_loader, ShardDataset, EXPECTED_TOKEN_COUNT
from model import ChessEdgeModel
from loss import EdgePromoValueLoss
from metrics import batch_metrics
from checkpoint import CheckpointManager
from plots import plot_recent_train, plot_val_history

faulthandler.enable()


class CombinedShardDataset(Dataset):
    """
    Combines base training shard with finetuning shard to prevent catastrophic forgetting.
    
    This dataset mixes data from the original positional training with new tactical training,
    ensuring the model maintains its positional understanding while learning tactical patterns.
    
    The combination is reshuffled every epoch for maximum randomization.
    """
    
    def __init__(self, base_shard_path: Path, finetune_shard_path: Path, shuffle_seed: int = 42):
        """
        Initialize combined dataset from two shard files.
        
        Args:
            base_shard_path: Path to original training shard (e.g., data/shards/shard_00.jsonl)
            finetune_shard_path: Path to finetuning shard (e.g., data/shards_finetune/shard_00.jsonl)
            shuffle_seed: Random seed for reproducible shuffling
        """
        self.base_shard_path = Path(base_shard_path)
        self.finetune_shard_path = Path(finetune_shard_path)
        self.shuffle_seed = shuffle_seed
        self.current_epoch = 0
        
        # Verify both shards exist
        if not self.base_shard_path.exists():
            raise FileNotFoundError(
                f"Base training shard not found: {self.base_shard_path}\n"
                f"This shard is needed to prevent catastrophic forgetting by mixing "
                f"original positional data with new tactical data."
            )
        if not self.finetune_shard_path.exists():
            raise FileNotFoundError(f"Finetuning shard not found: {self.finetune_shard_path}")
        
        # Create both underlying datasets
        self.base_ds = ShardDataset(self.base_shard_path, shuffle_seed)
        self.finetune_ds = ShardDataset(self.finetune_shard_path, shuffle_seed)
        
        self.base_len = len(self.base_ds)
        self.finetune_len = len(self.finetune_ds)
        self.total_len = self.base_len + self.finetune_len
        
        # Build initial shuffled index mapping
        self._build_index_map()
    
    def _build_index_map(self):
        """
        Create shuffled mapping of combined indices to (dataset_id, dataset_idx).
        
        This creates a random interleaving of samples from both datasets.
        The interleaving is different for each epoch (controlled by current_epoch).
        """
        # Create list of tuples: (dataset_id, idx_within_dataset)
        # dataset_id: 0 = base, 1 = finetune
        # Build in local variable first to avoid any corruption issues
        index_map = []
        
        for i in range(self.base_len):
            index_map.append((0, i))
        
        for i in range(self.finetune_len):
            index_map.append((1, i))
        
        # Shuffle the combined indices with epoch-dependent seed
        # This ensures different interleaving each epoch
        rng = random.Random(self.shuffle_seed + self.current_epoch * 9973)
        rng.shuffle(index_map)
        
        # Now assign to instance variable
        self.index_map = index_map
    
    def shuffle(self, epoch: int):
        """
        Shuffle for a new epoch.
        
        This does three levels of shuffling:
        1. Shuffles the base dataset's internal indices
        2. Shuffles the finetune dataset's internal indices  
        3. Creates a new random interleaving of samples from both datasets
        
        Args:
            epoch: Current epoch number (affects shuffle seed)
        """
        self.current_epoch = epoch
        
        # Shuffle both underlying datasets with their internal mechanisms
        self.base_ds.shuffle(epoch)
        self.finetune_ds.shuffle(epoch)
        
        # Rebuild the combined index map with new epoch-dependent shuffle
        self._build_index_map()
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx: int):
        """
        Get item by delegating to the appropriate underlying dataset.
        
        The index_map tells us which dataset and which index within that dataset.
        """
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"Index {idx} out of range [0, {self.total_len})")
        
        dataset_id, sub_idx = self.index_map[idx]
        
        if dataset_id == 0:
            return self.base_ds[sub_idx]
        else:
            return self.finetune_ds[sub_idx]


def get_finetune_config() -> Config:
    """
    Create a config specifically for finetuning.
    
    Key differences from base training:
    - Points to shards_finetune dataset
    - Much lower learning rate (10x lower than base)
    - Separate run directory to avoid overwriting
    - No learning rate scheduler (constant LR)
    """
    cfg = Config()
    
    # Point to finetuning shards
    cfg.data.shards_dir = "data/shards_finetune"
    
    # Isolated run directory for all finetuning outputs
    cfg.train.run_dir = "runs/chess_finetune"
    
    # Lower learning rate for finetuning (1/10th of base training)
    # This gentle learning rate prevents catastrophic forgetting
    cfg.optim.lr = 3e-5  # Base was 3e-4
    
    # Slightly lower weight decay for finetuning
    cfg.optim.weight_decay = 0.01  # Base was 0.05
    
    return cfg


class ConstantLR:
    """
    Dummy scheduler that maintains constant learning rate.
    This is crucial for finetuning to avoid forgetting.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch=None):
        """Does nothing - LR stays constant"""
        pass
    
    def state_dict(self):
        """Return state for checkpointing"""
        return {'base_lrs': self.base_lrs}
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint"""
        if 'base_lrs' in state_dict:
            self.base_lrs = state_dict['base_lrs']


class FinetuneTrainer:
    """
    Trainer specifically designed for finetuning a pretrained model.
    
    SPECIAL FEATURE: For ALL shards (00-09), combines base training data with finetuning data
    to prevent catastrophic forgetting. The model learns tactical patterns while
    maintaining its positional understanding across the entire dataset.
    """
    
    def __init__(self, cfg: Config, pretrained_checkpoint: str):
        """
        Initialize finetuning trainer.
        
        Args:
            cfg: Finetuning configuration
            pretrained_checkpoint: Path to pretrained model checkpoint to load
        """
        self.cfg = cfg
        self.pretrained_ckpt_path = pretrained_checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(cfg.train.seed)

        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.train.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.train.allow_tf32)

        # Paths - all isolated in finetune directory
        self.run_dir = Path(cfg.train.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.run_dir / cfg.train.plots_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Logger with finetune-specific naming
        self.logger = get_logger("finetune", cfg.train.logs_dir)
        self.logger.info("=" * 70)
        self.logger.info("Starting FINETUNING (Combined Base+Finetune for ALL shards 00-09)")
        self.logger.info(f"Loading pretrained model from: {pretrained_checkpoint}")
        self.logger.info(f"Finetuning data: {cfg.data.shards_dir}")
        self.logger.info(f"Base training data: data/shards/")
        self.logger.info(f"ALL shards will combine positional + tactical data")
        self.logger.info(f"Learning rate: {cfg.optim.lr:.2e} (constant, no scheduler)")
        self.logger.info("=" * 70)

        # CSV loggers with finetune naming
        self.csv_train = CSVLogger(
            self.run_dir / "metrics_train_finetune.csv",
            fieldnames=["global_step","epoch","shard","loss","edge_acc","promo_acc","promo_n","wdl_acc","lr"]
        )
        self.csv_val = CSVLogger(
            self.run_dir / "metrics_val_finetune.csv",
            fieldnames=["epoch","val_loss","edge_acc","promo_acc","wdl_acc","seconds"]
        )

        # Model - will load pretrained weights after creation
        self.model = ChessEdgeModel(cfg).to(self.device)
        self.criterion = EdgePromoValueLoss(cfg, self.model)
        
        # Load pretrained weights BEFORE creating optimizer
        self._load_pretrained_weights()
        
        # Optimizer with lower learning rate for finetuning
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.optim.lr,
            betas=(cfg.optim.beta1, cfg.optim.beta2),
            weight_decay=cfg.optim.weight_decay,
            eps=cfg.optim.eps,
        )
        
        # Use constant LR scheduler (no learning rate changes)
        self.scheduler = ConstantLR(self.optimizer)
        self.logger.info(f"Using constant learning rate: {cfg.optim.lr:.2e}")

        # Checkpointing - separate from base training
        self.ckpt = CheckpointManager(str(self.run_dir))
        self.state = {
            "epoch": 0,
            "next_train_shard_idx": 0,
            "val_history": {"loss": [], "edge_acc": [], "promo_acc": [], "wdl_acc": []},
            "total_seconds": 0.0,
            "global_step": 0,
            "finetune_mode": True,  # Flag to identify finetune checkpoints
            "pretrained_source": str(pretrained_checkpoint),
        }
        
        # Try to resume from previous finetuning checkpoint if exists
        self._try_load_finetune_checkpoint()

        # Recent window for plots & smoothed tqdm
        self.recent = RingBuffer(cfg.train.keep_recent_steps)
        self.smooth_window = int(cfg.train.display_smooth_steps)

    def _load_pretrained_weights(self):
        """
        Load pretrained model weights from base training.
        This is the critical step that transfers learned knowledge.
        """
        pretrained_path = Path(self.pretrained_ckpt_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {pretrained_path}\n"
                f"Please ensure you have a trained model from base training."
            )
        
        self.logger.info(f"Loading pretrained weights from: {pretrained_path}")
        
        try:
            payload = torch.load(pretrained_path, map_location=self.device)
            
            # Extract model state dict from checkpoint
            if isinstance(payload, dict):
                if "model" in payload:
                    state_dict = payload["model"]
                elif "state_dict" in payload:
                    state_dict = payload["state_dict"]
                else:
                    # Assume the payload itself is the state dict
                    state_dict = payload
            else:
                raise ValueError("Unexpected checkpoint format")
            
            # Handle DDP wrapper prefix if present
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            
            # Load weights into model
            self.model.load_state_dict(state_dict, strict=True)
            
            # Log info about pretrained checkpoint
            if isinstance(payload, dict):
                if "state" in payload and "epoch" in payload["state"]:
                    pretrain_epoch = payload["state"]["epoch"]
                    self.logger.info(f"Loaded pretrained model from epoch {pretrain_epoch}")
            
            self.logger.info("Successfully loaded pretrained weights!")
            self.logger.info("Model is now ready for finetuning on combined base+tactical data.")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pretrained checkpoint: {e}\n"
                f"Please verify the checkpoint file is valid and compatible."
            )

    def _try_load_finetune_checkpoint(self):
        """
        Try to resume from a previous finetuning checkpoint.
        This is separate from loading the initial pretrained model.
        """
        ck = self.ckpt.load_latest()
        if ck is None:
            self.logger.info("No previous finetuning checkpoint found. Starting fresh finetune.")
            return
        
        # Verify this is a finetune checkpoint
        if not ck.get("state", {}).get("finetune_mode", False):
            self.logger.warning(
                "Found checkpoint but it's not marked as finetune mode. "
                "Starting fresh finetune instead."
            )
            return
        
        try:
            self.model.load_state_dict(ck["model"])
            self.optimizer.load_state_dict(ck["optimizer"])
            if "scheduler" in ck:
                self.scheduler.load_state_dict(ck["scheduler"])
            self.state = ck["state"]
            self.logger.info(
                f"Resumed finetuning from epoch {self.state['epoch']}, "
                f"shard {self.state['next_train_shard_idx']}"
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to load finetuning checkpoint: {e}. "
                f"Starting fresh finetune from pretrained model."
            )

    def _save_epoch(self):
        """Save checkpoint with finetune-specific naming."""
        try:
            payload = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "state": self.state,
                "config": json.loads(json.dumps(self.cfg, default=lambda o: o.__dict__)),
            }
            self.ckpt.save_epoch(self.state["epoch"], payload)
            self.logger.info(f"Finetuning checkpoint saved for epoch {self.state['epoch']}")
        except Exception as e:
            self.logger.error(
                f"Failed to save finetuning checkpoint for epoch {self.state['epoch']}: {e}. "
                f"Training will continue but progress may be lost."
            )

    def _make_combined_loader(self, shard_name: str, epoch_for_shuffle: Optional[int]):
        """
        Create a DataLoader that combines base + finetune data for the given shard.
        
        This function combines data from data/shards/{shard_name} with 
        data/shards_finetune/{shard_name} to prevent catastrophic forgetting.
        
        Args:
            shard_name: Name of shard file (e.g., "shard_00.jsonl")
            epoch_for_shuffle: Epoch number for shuffling (None = no shuffle)
            
        Returns:
            DataLoader with combined dataset
        """
        base_shard_path = Path("data/shards") / shard_name
        finetune_shard_path = Path(self.cfg.data.shards_dir) / shard_name
        
        # Create combined dataset
        ds = CombinedShardDataset(
            base_shard_path, 
            finetune_shard_path, 
            shuffle_seed=self.cfg.data.shuffle_seed
        )
        
        # Shuffle if this is a training loader
        if epoch_for_shuffle is not None:
            ds.shuffle(epoch_for_shuffle)
        
        # Collate function (copied from data_loader.py since it's nested in make_loader)
        def collate(batch: List[Dict]) -> Dict:
            import logging
            logger = logging.getLogger("train")
            
            B = len(batch)

            # CRITICAL SAFETY CHECK: Verify all token tensors have correct shape
            token_shapes = [b["tokens"].shape[0] for b in batch]
            if not all(s == EXPECTED_TOKEN_COUNT for s in token_shapes):
                bad_indices = [i for i, s in enumerate(token_shapes) if s != EXPECTED_TOKEN_COUNT]
                error_msg = (
                    f"Token shape mismatch in batch collation! Expected all samples to have "
                    f"{EXPECTED_TOKEN_COUNT} tokens, but got shapes: {token_shapes}. "
                    f"Bad indices in batch: {bad_indices}. "
                    f"This should never happen with the new dataloader - please report this bug!"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Stack simple tensors
            tokens = torch.stack([b["tokens"] for b in batch], dim=0)
            edge_target = torch.stack([b["edge_target"] for b in batch], dim=0)
            promo_target = torch.stack([b["promo_target"] for b in batch], dim=0)
            is_promo = torch.stack([b["is_promo"] for b in batch], dim=0)
            wdl_target = torch.stack([b["wdl_target"] for b in batch], dim=0)

            # Edge mask
            edge_mask = torch.zeros(B, 64, 64, dtype=torch.bool)
            for bi, b in enumerate(batch):
                try:
                    for (fi, ti) in b["edge_list"]:
                        fi_i = int(fi); ti_i = int(ti)
                        if not (0 <= fi_i < 64 and 0 <= ti_i < 64):
                            logger.error(f"Invalid indices for edge_mask: bi={bi}, fi={fi_i}, ti={ti_i}")
                            continue
                        edge_mask[bi, fi_i, ti_i] = True
                except Exception as e:
                    logger.error(f"Error populating edge_mask for batch item {bi}: {e}")
                    continue
            edge_mask = edge_mask.contiguous()

            # Promo mask
            promo_mask = torch.zeros(B, 64, 64, 4, dtype=torch.bool)
            for bi, b in enumerate(batch):
                try:
                    for (fi, ti), promo_types in b["promo_map"].items():
                        fi_i = int(fi); ti_i = int(ti)
                        if not (0 <= fi_i < 64 and 0 <= ti_i < 64):
                            logger.error(f"Out-of-bounds promo edge: bi={bi}, fi={fi_i}, ti={ti_i}")
                            continue
                        for p in promo_types:
                            p_i = int(p)
                            if not (0 <= p_i < 4):
                                logger.error(f"Promo type out of bounds: p={p_i}")
                                continue
                            promo_mask[bi, fi_i, ti_i, p_i] = True
                except Exception as e:
                    logger.error(f"Error populating promo_mask for batch item {bi}: {e}")
                    continue
            promo_mask = promo_mask.contiguous()

            return {
                "tokens": tokens,
                "edge_target": edge_target,
                "promo_target": promo_target,
                "is_promo": is_promo,
                "wdl_target": wdl_target,
                "edge_mask": edge_mask,
                "promo_mask": promo_mask,
            }
        
        # Build kwargs for DataLoader
        kwargs = dict(
            dataset=ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,  # We handle shuffling in the dataset
            num_workers=self.cfg.data.num_workers,
            collate_fn=collate,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=True,
        )

        if self.cfg.data.num_workers > 0:
            if self.cfg.data.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.cfg.data.prefetch_factor)
            kwargs["persistent_workers"] = bool(self.cfg.data.persistent_workers)
            if self.cfg.data.multiprocessing_context:
                kwargs["multiprocessing_context"] = self.cfg.data.multiprocessing_context

        loader = DataLoader(**kwargs)
        return loader

    def _smoothed_postfix(self):
        """Calculate smoothed metrics for tqdm display."""
        keys = ["loss", "edge_acc", "promo_acc", "promo_n", "wdl_acc"]
        hist = self.recent.to_lists(keys)
        
        def mean_last(lst, k):
            if not lst:
                return 0.0
            sub = lst[-k:] if len(lst) > k else lst
            return sum(sub) / max(1, len(sub))
        
        k = self.smooth_window
        L = mean_last(hist["loss"], k)
        E = mean_last(hist["edge_acc"], k)
        W = mean_last(hist["wdl_acc"], k)
        
        promo_accs = hist["promo_acc"][-k:] if len(hist["promo_acc"]) > k else hist["promo_acc"]
        promo_ns = hist["promo_n"][-k:] if len(hist["promo_n"]) > k else hist["promo_n"]
        num = sum(a * n for a, n in zip(promo_accs, promo_ns))
        den = sum(promo_ns) if promo_ns else 0
        P = (num / den) if den > 0 else None
        
        return L, E, P, W

    def train_one_shard(self, shard_name: str):
        """Train on a single shard (combined base + finetune) for one epoch."""
        self.logger.info(f"=== Finetune E{self.state['epoch']:04d} on COMBINED {shard_name} ===")
        self.logger.info(f"    Combining: data/shards/{shard_name} + data/shards_finetune/{shard_name}")
        
        loader = self._make_combined_loader(shard_name, epoch_for_shuffle=self.state["epoch"])
        
        self.model.train()
        t0 = time.time()
        pbar = tqdm(total=len(loader), ncols=150, desc=f"Finetune {shard_name}")
        accum = int(self.cfg.train.grad_accum_steps)
        self.optimizer.zero_grad(set_to_none=True)
        
        for step, batch in enumerate(loader, 1):
            try:
                for k, v in list(batch.items()):
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device, non_blocking=True)

                outputs = self.model(batch["tokens"])
                loss, loss_dict = self.criterion(batch, outputs)
                
                if not torch.isfinite(loss):
                    self.logger.error("Non-finite loss encountered; skipping batch.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                
                mets = batch_metrics(batch, outputs)
                scaled_loss = loss / accum
                scaled_loss.backward()

                if step % accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
                    self.optimizer.step()
                    # Note: scheduler.step() does nothing for ConstantLR
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                rec = {
                    "loss": loss_dict["total"],
                    "edge_acc": mets["edge_acc"],
                    "promo_acc": mets["promo_acc"],
                    "promo_n": float(mets["promo_n"]),
                    "wdl_acc": mets["wdl_acc"],
                }
                self.recent.append(rec)
                self.state["global_step"] += 1

                if step % self.cfg.train.log_interval == 0:
                    L, E, P, W = self._smoothed_postfix()
                    p_text = f"{(P*100):5.1f}%" if P is not None else "  —  "
                    lr_now = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({
                        "L": f"{L:.3f}",
                        "E": f"{E*100:5.1f}%",
                        "P": p_text,
                        "W": f"{W*100:5.1f}%",
                        "LR": f"{lr_now:.2e}",
                    })
                    self.csv_train.append({
                        "global_step": self.state["global_step"],
                        "epoch": self.state["epoch"],
                        "shard": shard_name,
                        "loss": L,
                        "edge_acc": E,
                        "promo_acc": (P if P is not None else ""),
                        "promo_n": rec["promo_n"],
                        "wdl_acc": W,
                        "lr": lr_now,
                    })
            except Exception as e:
                self.logger.exception(
                    f"Error during finetuning batch {step} on {shard_name}: {e}. "
                    f"Skipping batch and continuing."
                )
                continue

            pbar.update(1)
        
        pbar.close()
        dt = time.time() - t0
        self.state["total_seconds"] += dt
        self.logger.info(f"Shard {shard_name} done in {format_seconds(dt)} | {get_memory_stats()}")

    @torch.no_grad()
    def validate_on_val_shard(self):
        """Validate on combined validation shard (base + finetune)."""
        shard = self.cfg.data.val_shard
        self.logger.info(f"=== Finetune E{self.state['epoch']:04d} Validate on COMBINED {shard} ===")
        self.logger.info(f"    Combining: data/shards/{shard} + data/shards_finetune/{shard}")
        
        try:
            loader = self._make_combined_loader(shard, epoch_for_shuffle=None)
        except Exception as e:
            self.logger.error(f"Failed to create validation loader: {e}. Skipping validation.")
            return

        self.model.eval()
        t0 = time.time()
        loss_sum = 0.0
        edge_hits = 0
        promo_hits = 0
        promo_cnt = 0
        wdl_hits = 0
        count = 0
        pbar = tqdm(total=len(loader), ncols=150, desc=f"Val {shard}")
        
        for step, batch in enumerate(loader, 1):
            try:
                for k, v in list(batch.items()):
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device, non_blocking=True)
                
                outputs = self.model(batch["tokens"])
                loss, _ = self.criterion(batch, outputs)
                mets = batch_metrics(batch, outputs)
                bs = batch["tokens"].size(0)

                loss_sum += float(loss.detach().cpu().item())
                edge_hits += mets["edge_acc"] * bs
                wdl_hits += mets["wdl_acc"] * bs
                
                if mets["promo_n"] > 0:
                    promo_hits += mets["promo_acc"] * mets["promo_n"]
                    promo_cnt += mets["promo_n"]
                
                count += bs

                pbar.set_postfix({
                    "L": f"{loss_sum/max(1,count):.3f}",
                    "E": f"{(edge_hits/max(1,count))*100:5.1f}%",
                    "P": f"{(promo_hits/max(1,promo_cnt))*100:5.1f}%" if promo_cnt>0 else "  —  ",
                    "W": f"{(wdl_hits/max(1,count))*100:5.1f}%",
                })
            except Exception as e:
                self.logger.exception(
                    f"Error during validation batch {step} on {shard}: {e}. "
                    f"Skipping batch and continuing."
                )
                continue

            pbar.update(1)
        
        pbar.close()
        dt = time.time() - t0
        self.state["total_seconds"] += dt

        if count == 0:
            self.logger.error("Validation produced zero valid samples; skipping metrics.")
            return

        val_loss = loss_sum / max(1, count)
        edge_acc = edge_hits / max(1, count)
        promo_acc = (promo_hits / max(1, promo_cnt)) if promo_cnt > 0 else 0.0
        wdl_acc = wdl_hits / max(1, count)
        
        self.logger.info(
            f"FINETUNE VAL loss={val_loss:.4f} edge={edge_acc*100:.2f}% "
            f"promo={promo_acc*100:.2f}% wdl={wdl_acc*100:.2f}% "
            f"in {format_seconds(dt)} | {get_memory_stats()}"
        )

        self.state["val_history"]["loss"].append(val_loss)
        self.state["val_history"]["edge_acc"].append(edge_acc)
        self.state["val_history"]["promo_acc"].append(promo_acc)
        self.state["val_history"]["wdl_acc"].append(wdl_acc)

        try:
            recent = self.recent.to_lists(["loss","edge_acc","promo_acc","wdl_acc"])
            plot_recent_train(recent, self.plots_dir / "train_recent_finetune.png")
            plot_val_history(self.state["val_history"], self.plots_dir / "val_history_finetune.png")
        except Exception as e:
            self.logger.error(f"Failed to generate plots: {e}. Continuing without plots.")

        try:
            self.csv_val.append({
                "epoch": self.state["epoch"],
                "val_loss": val_loss,
                "edge_acc": edge_acc,
                "promo_acc": (promo_acc if promo_cnt > 0 else ""),
                "wdl_acc": wdl_acc,
                "seconds": dt,
            })
        except Exception as e:
            self.logger.error(f"Failed to write validation CSV: {e}")

    def run(self):
        """Main finetuning loop - cycles through combined shards indefinitely."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("Starting infinite finetuning loop with COMBINED datasets...")
        self.logger.info("All shards (00-09) will mix positional + tactical data")
        self.logger.info("Press Ctrl+C to stop and save checkpoint")
        self.logger.info("=" * 70 + "\n")
        
        while True:
            try:
                idx = self.state["next_train_shard_idx"] % len(self.cfg.data.train_shards)
                shard = self.cfg.data.train_shards[idx]

                self.train_one_shard(shard)

                self.state["epoch"] += 1
                self.state["next_train_shard_idx"] = (idx + 1) % len(self.cfg.data.train_shards)

                if self.state["epoch"] % self.cfg.train.eval_every_epoch == 0:
                    self.validate_on_val_shard()

                if self.state["epoch"] % self.cfg.train.save_every_epoch == 0:
                    self._save_epoch()

            except KeyboardInterrupt:
                self.logger.info("\nInterrupted by user. Saving final checkpoint...")
                self._save_epoch()
                self.logger.info("Finetuning stopped gracefully.")
                break
            except Exception as e:
                self.logger.exception(f"Fatal error in finetuning loop at epoch {self.state['epoch']}: {e}")
                self.logger.info("Attempting to save checkpoint before exit...")
                self._save_epoch()
                raise


def main():
    """
    Main entry point for finetuning.
    
    IMPORTANT: Specify the path to your pretrained checkpoint below.
    This should be a checkpoint from your base training run.
    """
    # Path to pretrained model from base training
    # UPDATE THIS to point to your actual pretrained checkpoint
    PRETRAINED_CHECKPOINT = "runs/chess/checkpoints/checkpoint_epoch_18.pt"
    
    # You can also specify a specific epoch checkpoint:
    # PRETRAINED_CHECKPOINT = "runs/chess/checkpoints/checkpoint_epoch_50.pt"
    
    # Verify the pretrained checkpoint exists
    if not Path(PRETRAINED_CHECKPOINT).exists():
        print(f"ERROR: Pretrained checkpoint not found at: {PRETRAINED_CHECKPOINT}")
        print("\nPlease update PRETRAINED_CHECKPOINT in train_finetune.py to point to your")
        print("trained model from base training (runs/chess/checkpoints/checkpoint_*.pt)")
        return
    
    # Create finetuning config
    cfg = get_finetune_config()
    
    print("\n" + "=" * 70)
    print("Chess Model Finetuning with Combined Datasets")
    print("=" * 70)
    print(f"Pretrained model:     {PRETRAINED_CHECKPOINT}")
    print(f"Base training data:   data/shards/")
    print(f"Finetuning data:      {cfg.data.shards_dir}")
    print(f"ALL shards combined:  positional + tactical data")
    print(f"Output directory:     {cfg.train.run_dir}")
    print(f"Learning rate:        {cfg.optim.lr:.2e} (constant, no scheduler)")
    print("=" * 70 + "\n")
    
    # Create trainer and start finetuning
    trainer = FinetuneTrainer(cfg, PRETRAINED_CHECKPOINT)
    trainer.run()


if __name__ == "__main__":
    main()