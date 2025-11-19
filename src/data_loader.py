# data_loader.py
# JSONL shard streaming dataset for chess positions
# MULTIPROCESSING-SAFE: No shared file handles, exact byte-range reads
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from utils import parse_move_str, hmc_bucket

logger = logging.getLogger("train")

PIECE_TOKENS = [
    "EMPTY","wP","wN","wB","wR","wQ","wK","bP","bN","bB","bR","bQ","bK"
]
SPECIAL_FIXED = [
    "CLS","TURN_W","TURN_B",
    "WK_CASTLE_NO","WK_CASTLE_YES",
    "WQ_CASTLE_NO","WQ_CASTLE_YES",
    "BK_CASTLE_NO","BK_CASTLE_YES",
    "BQ_CASTLE_NO","BQ_CASTLE_YES",
    "EP_NONE",
]
EP_FILES = [f"EP_{c.upper()}" for c in "abcdefgh"]
HMC_BUCKETS = [f"HMC_{i*10:02d}_{i*10+9:02d}" for i in range(10)]

# Build vocabulary map
VOCAB_LIST = PIECE_TOKENS + SPECIAL_FIXED + EP_FILES + HMC_BUCKETS
VOCAB = {tok: i for i, tok in enumerate(VOCAB_LIST)}

# CRITICAL: Expected token count per position
EXPECTED_TOKEN_COUNT = 72


def encode_position_1d(tokens_1d: List[str]) -> List[int]:
    """
    Expect: ["CLS", 64 squares, TURN_*, 4 castle flags per side *_YES/NO,
             EP_* or EP_NONE (maybe EP_e6), HMC_*]
    Map EP_<square> -> EP_<file>; HMC_* -> tens bucket.
    
    CRITICAL: This function validates token count and raises ValueError if incorrect.
    """
    # VALIDATE TOKEN COUNT FIRST
    if len(tokens_1d) != EXPECTED_TOKEN_COUNT:
        raise ValueError(
            f"Invalid token count: expected {EXPECTED_TOKEN_COUNT}, got {len(tokens_1d)}. "
            f"First 5 tokens: {tokens_1d[:5]}, Last 5 tokens: {tokens_1d[-5:]}"
        )
    
    out: List[int] = []
    for t in tokens_1d:
        if t.startswith("EP_") and t != "EP_NONE":
            # Normalize EP to file only (EP_A..EP_H)
            try:
                file_c = t.split("_")[1][0].upper()
                t = f"EP_{file_c}"
            except Exception:
                t = "EP_NONE"
        elif t.startswith("HMC_"):
            t = hmc_bucket(t)

        if t not in VOCAB:
            # Robust fallback mapping
            if t in PIECE_TOKENS:
                t = "EMPTY"
            elif t.startswith("EP_"):
                t = "EP_NONE"
            elif t.startswith("HMC_"):
                t = "HMC_00_09"

        out.append(VOCAB.get(t, 0))
    
    # Double-check output length
    if len(out) != EXPECTED_TOKEN_COUNT:
        raise ValueError(f"Encoding produced {len(out)} tokens instead of {EXPECTED_TOKEN_COUNT}")
    
    return out


def make_edge_mask_and_promo_map(legal_moves: List[str]) -> Tuple[List[Tuple[int,int]], Dict[Tuple[int,int], List[int]]]:
    """
    Returns:
      edge_list: list of (from, to) tuples for legal edges (to be converted to mask in collate)
      promo_map: dict[(from,to)] -> list of promo_type indices allowed at that edge

    Returns Python structures; we build tensors in the main process for safety.
    """
    edge_list: List[Tuple[int, int]] = []
    promo_map: Dict[Tuple[int,int], List[int]] = {}

    for mv in legal_moves:
        try:
            fi, ti, is_p, pidx = parse_move_str(mv)
            if not isinstance(fi, int) or not isinstance(ti, int):
                logger.warning(f"Non-integer indices in move: {mv}")
                continue
            if not (0 <= fi < 64 and 0 <= ti < 64):
                logger.warning(f"Out-of-bounds in move: {mv} => fi={fi}, ti={ti}")
                continue

            edge_list.append((int(fi), int(ti)))

            if is_p:
                if not isinstance(pidx, int) or not (0 <= pidx < 4):
                    logger.warning(f"Invalid promo index in move: {mv} -> pidx={pidx}")
                    continue
                key = (int(fi), int(ti))
                lst = promo_map.setdefault(key, [])
                if pidx not in lst:
                    lst.append(int(pidx))

        except Exception as e:
            logger.warning(f"Failed to parse legal move '{mv}': {e}. Skipping.")
            continue

    return edge_list, promo_map


class ShardDataset(Dataset):
    """
    MULTIPROCESSING-SAFE dataset for JSONL shards.
    
    KEY DESIGN DECISIONS FOR SAFETY:
    1. Stores exact byte offsets AND line lengths during initialization
    2. NO persistent file handles - opens fresh for each read
    3. Reads exact byte ranges, never uses readline()
    4. Each worker process gets independent file access
    5. Validation at multiple layers to catch any corruption
    
    This eliminates ALL race conditions that can occur with multiprocessing.
    """
    def __init__(self, shard_path: Path, shuffle_seed: int = 42):
        self.shard_path = Path(shard_path)
        self.shuffle_seed = shuffle_seed
        self._offsets: List[int] = []
        self._line_lengths: List[int] = []  # NEW: Store exact line lengths
        self._corrupted_count = 0
        self._build_offsets_and_lengths()
        self._validate_first_sample()

    def _build_offsets_and_lengths(self):
        """Build index of exact byte positions and lengths for each line."""
        self._offsets.clear()
        self._line_lengths.clear()
        
        with self.shard_path.open("rb") as f:
            offset = 0
            for line in f:
                self._offsets.append(offset)
                self._line_lengths.append(len(line))
                offset += len(line)
        
        if not self._offsets:
            raise RuntimeError(f"Shard {self.shard_path} has no lines.")
        
        logger.info(f"Indexed {len(self._offsets):,} lines from {self.shard_path.name}")

    def _validate_first_sample(self):
        """Validate that we can read the first sample successfully."""
        try:
            _ = self._read_json_at_index(0)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read first sample from {self.shard_path}. "
                f"File may be corrupted. Error: {e}"
            )

    def shuffle(self, epoch: int):
        """Shuffle the offset indices (not the file itself)."""
        rng = random.Random(self.shuffle_seed + epoch * 9973)
        # Shuffle both lists in sync
        combined = list(zip(self._offsets, self._line_lengths))
        rng.shuffle(combined)
        self._offsets, self._line_lengths = zip(*combined)
        self._offsets = list(self._offsets)
        self._line_lengths = list(self._line_lengths)

    def __len__(self):
        return len(self._offsets)

    def _read_json_at_index(self, idx: int) -> Dict:
        """
        Read JSON at exact index using byte-range read.
        THREAD-SAFE: Opens its own file handle, reads exact bytes, closes.
        """
        offset = self._offsets[idx]
        length = self._line_lengths[idx]
        
        # Open file fresh for this read (no shared handles!)
        with self.shard_path.open("rb") as f:
            f.seek(offset)
            raw = f.read(length)
        
        if not raw:
            raise EOFError(f"Empty read at index {idx}, offset {offset}")
        
        # Decode and parse
        s = raw.decode("utf-8", errors="strict").rstrip("\r\n")
        return json.loads(s)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get item with robust error handling and retry logic.
        """
        max_retries = 20
        tried = set()
        
        for retry_count in range(max_retries):
            j = idx
            if j in tried:
                # Try nearby samples with increasing distance
                offset_delta = (retry_count // 2 + 1) * (1 if retry_count % 2 == 0 else -1)
                j = (idx + offset_delta) % len(self._offsets)
            tried.add(j)
            
            try:
                obj = self._read_json_at_index(j)
                
                # CRITICAL VALIDATION: Check token count BEFORE encoding
                pos_tokens = obj["position_1d"]
                if not isinstance(pos_tokens, list):
                    raise ValueError(f"position_1d is not a list: {type(pos_tokens)}")
                
                if len(pos_tokens) != EXPECTED_TOKEN_COUNT:
                    self._corrupted_count += 1
                    if self._corrupted_count <= 10:
                        logger.error(
                            f"CORRUPTED DATA at index {j} in {self.shard_path.name}: "
                            f"Expected {EXPECTED_TOKEN_COUNT} tokens, got {len(pos_tokens)}. "
                            f"First 5 tokens: {pos_tokens[:5]}, Last 5 tokens: {pos_tokens[-5:]}. "
                            f"Retrying with different sample (attempt {retry_count + 1}/{max_retries})."
                        )
                    continue
                
                # Encode position (will raise if validation fails)
                pos_ids = encode_position_1d(pos_tokens)
                
                # Parse target move
                fi, ti, is_promo, promo_idx = parse_move_str(obj["target"])

                if not isinstance(fi, int) or not isinstance(ti, int):
                    raise ValueError(f"Non-integer target indices: fi={fi}, ti={ti}")
                if not (0 <= fi < 64 and 0 <= ti < 64):
                    raise ValueError(f"Target indices out of bounds: fi={fi}, ti={ti}")
                if is_promo and (not isinstance(promo_idx, int) or not (0 <= promo_idx < 4)):
                    raise ValueError(f"Invalid promo_idx: {promo_idx}")

                edge_target = int(fi) * 64 + int(ti)
                edge_list, promo_map = make_edge_mask_and_promo_map(obj["legal_moves"])

                return {
                    "tokens": torch.tensor(pos_ids, dtype=torch.long),
                    "edge_target": torch.tensor(edge_target, dtype=torch.long),
                    "promo_target": torch.tensor(promo_idx if is_promo else -1, dtype=torch.long),
                    "is_promo": torch.tensor(1 if is_promo else 0, dtype=torch.long),
                    "wdl_target": torch.tensor({-1:0,0:1,1:2}[int(obj["wdl"])], dtype=torch.long),
                    "edge_list": edge_list,
                    "promo_map": promo_map,
                }
            except Exception as e:
                logger.warning(
                    f"Failed to read/parse index {j} in {self.shard_path}: {e}. "
                    f"Retry {retry_count + 1}/{max_retries}"
                )
                continue

        # Exhausted all retries - use first sample as fallback
        logger.error(
            f"CRITICAL: Exhausted {max_retries} retries for idx {idx} in {self.shard_path}. "
            f"Using fallback (first sample). Total corrupted encountered: {self._corrupted_count}"
        )
        
        try:
            obj = self._read_json_at_index(0)
            pos_ids = encode_position_1d(obj["position_1d"])
            fi, ti, is_promo, promo_idx = parse_move_str(obj["target"])
            edge_target = int(fi) * 64 + int(ti)
            edge_list, promo_map = make_edge_mask_and_promo_map(obj["legal_moves"])
            return {
                "tokens": torch.tensor(pos_ids, dtype=torch.long),
                "edge_target": torch.tensor(edge_target, dtype=torch.long),
                "promo_target": torch.tensor(promo_idx if is_promo else -1, dtype=torch.long),
                "is_promo": torch.tensor(1 if is_promo else 0, dtype=torch.long),
                "wdl_target": torch.tensor({-1:0,0:1,1:2}[int(obj["wdl"])], dtype=torch.long),
                "edge_list": edge_list,
                "promo_map": promo_map,
            }
        except Exception as e:
            raise RuntimeError(
                f"FATAL: Even fallback sample failed in {self.shard_path}. "
                f"The shard file is severely corrupted. Error: {e}"
            )


def make_loader(
    shard_path: Path,
    batch_size: int,
    shuffle_epoch: Optional[int],
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: Optional[int],
    persistent_workers: bool,
    mp_context: Optional[str],
):
    ds = ShardDataset(shard_path)
    if shuffle_epoch is not None:
        ds.shuffle(shuffle_epoch)

    def collate(batch: List[Dict]) -> Dict:
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

    # Build kwargs
    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        drop_last=True,
    )

    if num_workers > 0:
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = bool(persistent_workers)
        if mp_context:
            kwargs["multiprocessing_context"] = mp_context

    loader = DataLoader(**kwargs)
    return loader