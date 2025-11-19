# checkpoint.py
from pathlib import Path
from typing import Dict, Optional
import torch


class CheckpointManager:
    """
    Epoch-only checkpointing (saves at epoch end). No mid-epoch persistence.
    """
    def __init__(self, run_dir: str):
        self.ckpt_dir = Path(run_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def latest_epoch_ckpt(self) -> Optional[Path]:
        cands = sorted(self.ckpt_dir.glob("checkpoint_epoch_*.pt"))
        return cands[-1] if cands else None

    def save_epoch(self, epoch: int, payload: Dict):
        final = self.ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(payload, final)
        
        # Update pointer to latest checkpoint
        latest = self.ckpt_dir / "checkpoint_latest.pt"
        torch.save(payload, latest)

    def load_latest(self) -> Optional[Dict]:
        latest = self.ckpt_dir / "checkpoint_latest.pt"
        if latest.exists():
            try:
                return torch.load(latest, map_location="cpu")
            except Exception:
                return None
        return None