# train.py
# Core training script (no CLI). Infinite shard cycling. FP32 only. Val on shard_09 each epoch.
import json
import time
import faulthandler
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from config import Config
from logger import get_logger, CSVLogger
from utils import set_seed, RingBuffer, format_seconds, get_memory_stats
from data_loader import make_loader
from model import ChessEdgeModel
from loss import EdgePromoValueLoss
from metrics import batch_metrics
from scheduler import CosineWithRestartsLR
from checkpoint import CheckpointManager
from plots import plot_recent_train, plot_val_history

faulthandler.enable()


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(cfg.train.seed)

        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.train.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.train.allow_tf32)

        # Paths
        self.run_dir = Path(cfg.train.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.run_dir / cfg.train.plots_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger("train", cfg.train.logs_dir)
        self.logger.info("Starting training (FP32, TF32 disabled)")

        # CSV loggers
        self.csv_train = CSVLogger(
            self.run_dir / "metrics_train.csv",
            fieldnames=["global_step","epoch","shard","loss","edge_acc","promo_acc","promo_n","wdl_acc","lr"]
        )
        self.csv_val = CSVLogger(
            self.run_dir / "metrics_val.csv",
            fieldnames=["epoch","val_loss","edge_acc","promo_acc","wdl_acc","seconds"]
        )

        # Model / Opt / Sched
        self.model = ChessEdgeModel(cfg).to(self.device)
        self.criterion = EdgePromoValueLoss(cfg, self.model)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.optim.lr,
            betas=(cfg.optim.beta1, cfg.optim.beta2),
            weight_decay=cfg.optim.weight_decay,
            eps=cfg.optim.eps,
        )
        self.scheduler = CosineWithRestartsLR(
            self.optimizer,
            warmup_steps=cfg.sched.warmup_steps,
            t0_steps=cfg.sched.t0_steps,
            t_mult=cfg.sched.t_mult,
            min_lr_ratio=cfg.sched.min_lr_ratio,
            warmup_mode=cfg.sched.warmup_mode,
        )

        # Checkpointing
        self.ckpt = CheckpointManager(str(self.run_dir))
        self.state = {
            "epoch": 0,
            "next_train_shard_idx": 0,
            "val_history": {"loss": [], "edge_acc": [], "promo_acc": [], "wdl_acc": []},
            "total_seconds": 0.0,
            "global_step": 0,
        }
        self._try_load()

        # Recent window for plots & smoothed tqdm
        self.recent = RingBuffer(cfg.train.keep_recent_steps)
        self.smooth_window = int(cfg.train.display_smooth_steps)

    def _try_load(self):
        ck = self.ckpt.load_latest()
        if ck is None:
            self.logger.info("No previous checkpoint. Fresh run.")
            return
        try:
            self.model.load_state_dict(ck["model"])
            self.optimizer.load_state_dict(ck["optimizer"])
            self.scheduler.load_state_dict(ck["scheduler"])
            self.state = ck["state"]
            self.logger.info(f"Resumed at epoch={self.state['epoch']} next_shard={self.state['next_train_shard_idx']}")
        except Exception as e:
            self.logger.warning(f"Failed to fully load checkpoint (starting fresh components). Reason: {e}")

    def _save_epoch(self):
        try:
            payload = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if hasattr(self.scheduler, "state_dict") else {},
                "state": self.state,
                "config": json.loads(json.dumps(self.cfg, default=lambda o: o.__dict__)),
            }
            self.ckpt.save_epoch(self.state["epoch"], payload)
            self.logger.info(f"Checkpoint saved for epoch {self.state['epoch']}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint for epoch {self.state['epoch']}: {e}. Training will continue but progress may be lost.")

    def _make_loader_for(self, shard_name: str, epoch_for_shuffle: int, is_train: bool):
        p = Path(self.cfg.data.shards_dir) / shard_name
        return make_loader(
            p,
            batch_size=self.cfg.data.batch_size,
            shuffle_epoch=(epoch_for_shuffle if is_train else None),
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            prefetch_factor=self.cfg.data.prefetch_factor,
            persistent_workers=self.cfg.data.persistent_workers,
            mp_context=self.cfg.data.multiprocessing_context
        )

    def _smoothed_postfix(self):
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
        promo_ns   = hist["promo_n"][-k:]   if len(hist["promo_n"])   > k else hist["promo_n"]
        num = sum(a * n for a, n in zip(promo_accs, promo_ns))
        den = sum(promo_ns) if promo_ns else 0
        P = (num / den) if den > 0 else None
        return L, E, P, W

    def train_one_shard(self, shard_name: str):
        self.logger.info(f"=== E{self.state['epoch']:04d} Train on {shard_name} ===")
        loader = self._make_loader_for(shard_name, epoch_for_shuffle=self.state["epoch"], is_train=True)
        self.model.train()
        t0 = time.time()
        pbar = tqdm(total=len(loader), ncols=150, desc=f"Train {shard_name}")
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
                self.logger.exception(f"Error during training batch {step} on {shard_name}: {e}. Skipping batch and continuing.")
                continue

            pbar.update(1)
        pbar.close()
        dt = time.time() - t0
        self.state["total_seconds"] += dt
        self.logger.info(f"Shard {shard_name} done in {format_seconds(dt)} | {get_memory_stats()}")

    @torch.no_grad()
    def validate_on_val_shard(self):
        shard = self.cfg.data.val_shard
        self.logger.info(f"=== E{self.state['epoch']:04d} Validate on {shard} ===")
        try:
            loader = self._make_loader_for(shard, epoch_for_shuffle=0, is_train=False)
        except Exception as e:
            self.logger.error(f"Failed to create validation loader: {e}. Skipping validation for this epoch.")
            return

        self.model.eval()
        t0 = time.time()
        loss_sum = 0.0
        edge_hits = 0
        promo_hits = 0
        promo_cnt = 0
        wdl_hits = 0
        count = 0
        pbar = tqdm(total=len(loader), ncols=150, desc=f"Val   {shard}")
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
                wdl_hits  += mets["wdl_acc"]  * bs
                if mets["promo_n"] > 0:
                    promo_hits += mets["promo_acc"] * mets["promo_n"]
                    promo_cnt  += mets["promo_n"]
                count += bs

                pbar.set_postfix({
                    "L": f"{loss_sum/max(1,count):.3f}",
                    "E": f"{(edge_hits/max(1,count))*100:5.1f}%",
                    "P": f"{(promo_hits/max(1,promo_cnt))*100:5.1f}%" if promo_cnt>0 else "  —  ",
                    "W": f"{(wdl_hits/max(1,count))*100:5.1f}%",
                })
            except Exception as e:
                self.logger.exception(f"Error during validation batch {step} on {shard}: {e}. Skipping batch and continuing.")
                continue

            pbar.update(1)
        pbar.close()
        dt = time.time() - t0
        self.state["total_seconds"] += dt

        if count == 0:
            self.logger.error("Validation produced zero valid samples; skipping metrics aggregation for this epoch.")
            return

        val_loss = loss_sum / max(1, count)
        edge_acc = edge_hits / max(1, count)
        promo_acc = (promo_hits / max(1, promo_cnt)) if promo_cnt > 0 else 0.0
        wdl_acc = wdl_hits / max(1, count)
        self.logger.info(f"VAL loss={val_loss:.4f} edge={edge_acc*100:.2f}% promo={promo_acc*100:.2f}% wdl={wdl_acc*100:.2f}% in {format_seconds(dt)} | {get_memory_stats()}")

        self.state["val_history"]["loss"].append(val_loss)
        self.state["val_history"]["edge_acc"].append(edge_acc)
        self.state["val_history"]["promo_acc"].append(promo_acc)
        self.state["val_history"]["wdl_acc"].append(wdl_acc)

        try:
            recent = self.recent.to_lists(["loss","edge_acc","promo_acc","wdl_acc"])
            plot_recent_train(recent, self.plots_dir / "train_recent.png")
            plot_val_history(self.state["val_history"], self.plots_dir / "val_history.png")
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
                self.logger.info("Interrupted by user. Saving checkpoint before exit...")
                self._save_epoch()
                break
            except Exception as e:
                self.logger.exception(f"Fatal error in training loop at epoch {self.state['epoch']}: {e}")
                self.logger.info("Attempting to save checkpoint before exit...")
                self._save_epoch()
                raise


if __name__ == "__main__":
    cfg = Config()
    Trainer(cfg).run()
