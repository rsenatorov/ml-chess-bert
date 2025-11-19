# utils.py
import random
import logging
from typing import Tuple

import torch

logger = logging.getLogger("train")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_seconds(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class RingBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buf = []
    def append(self, item):
        self.buf.append(item)
        if len(self.buf) > self.capacity:
            self.buf.pop(0)
    def to_lists(self, keys):
        out = {k: [] for k in keys}
        for it in self.buf:
            for k in keys:
                out[k].append(it.get(k, 0.0))
        return out


def parse_move_str(m: str) -> Tuple[int, int, bool, int]:
    """
    Parse move string like 'Rd7Rg7', 'Pd2Qd1', 'Ph3Ph2' (promo encoded in piece letter before dest).
    Returns: (from_idx, to_idx, is_promo, promo_idx[Q=0,R=1,B=2,N=3 or -1])
    Raises on invalid input.
    """
    def sq_to_idx(sq: str) -> int:
        if len(sq) != 2:
            raise ValueError(f"Invalid square format: {sq}")
        file_char = sq[0].lower()
        rank_char = sq[1]
        if not ('a' <= file_char <= 'h'):
            raise ValueError(f"Invalid file in square: {sq}")
        if not ('1' <= rank_char <= '8'):
            raise ValueError(f"Invalid rank in square: {sq}")
        file = ord(file_char) - ord('a')  # 0-7
        rank = int(rank_char) - 1         # 0-7
        idx = rank * 8 + file             # 0-63
        if not (0 <= idx <= 63):
            raise ValueError(f"Square {sq} produced out-of-bounds index {idx}")
        return idx

    try:
        is_promo = False
        promo_idx = -1
        if len(m) < 5:
            raise ValueError(f"Move string too short: {m}")
        to_sq = m[-2:]
        from_sq = m[1:3]
        if len(m) == 6:
            p = m[3].upper()
            is_promo = p in "QRBN"
            if is_promo:
                promo_idx = {"Q": 0, "R": 1, "B": 2, "N": 3}[p]
                if not (0 <= promo_idx <= 3):
                    raise ValueError(f"Invalid promo index: {promo_idx}")
        fi = sq_to_idx(from_sq)
        ti = sq_to_idx(to_sq)
        return int(fi), int(ti), bool(is_promo), int(promo_idx)
    except Exception as e:
        logger.error(f"Failed to parse move string '{m}': {e}.")
        raise


def hmc_bucket(token: str) -> str:
    """
    Convert HMC token like HMC_2 -> HMC_00_09 etc if needed.
    If already bucketed HMC_10_19 etc, return as is.
    """
    if "_" in token and token.count("_") == 2:
        return token
    try:
        n = int(token.split("_")[1])
        lo = (n // 10) * 10
        hi = lo + 9
        return f"HMC_{lo:02d}_{hi:02d}"
    except Exception:
        return "HMC_00_09"


def get_memory_stats() -> str:
    """
    Return a compact string with GPU/CPU memory stats for logging.
    """
    parts = []
    try:
        if torch.cuda.is_available():
            cur = torch.cuda.memory_allocated() / (1024**2)
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            parts.append(f"GPU mem {cur:.0f}MB (peak {peak:.0f}MB)")
    except Exception:
        pass
    return " | ".join(parts) if parts else "no CUDA info"