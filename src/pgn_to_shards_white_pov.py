#!/usr/bin/env python3
"""
pgn_to_shards_white_pov.py  (no CLI)

Reads PGN files from data/games/stockfish/ (supports .txt, .pgn, .pgn.gz)
and emits JSONL shards in data/shards/:
  shard_00.jsonl ... shard_08.jsonl  (train)
  shard_09.jsonl                     (val)

Each JSON line matches your codebase format:
{
  "position_1d": [...72 tokens...],
  "legal_moves": ["Pe2e4", "Ng1f3", ...],
  "target": "Pe2e4",
  "wdl": 1 | 0 | -1      # ALWAYS White perspective
}

- Token layout: CLS + 64 squares a1..h8 + TURN_[W/B] + 4 castle flags + EP_* or EP_NONE + HMC_n
- Move format: "<P>{from}{to}" or "<P>{from}{Q/R/B/N}{to}" (promo letter at index 3)
- Per-game stable split into shards (sha1 of common tags) to avoid train/val leakage.
- Processes ALL moves (both sides).
- Stops after exactly 100,000 games.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Tuple
import gzip
import io
import json
import os
import hashlib
import re

import chess
import chess.pgn
from tqdm import tqdm

# -------- paths & limits --------
INPUT_DIR   = Path("data/games/stockfish")   # where your per-game PGN .txt files live
SHARDS_DIR  = Path("data/shards")
N_TRAIN_SHARDS = 9
VAL_SHARD_IDX  = 9
TARGET_GAMES   = 100_000  # stop after exactly this many games

# ---------- tokenization (72 tokens) ----------
def _piece_token(piece: chess.Piece | None) -> str:
    if piece is None:
        return "EMPTY"
    return f"{'w' if piece.color else 'b'}{piece.symbol().upper()}"

def tokens_from_board(board: chess.Board) -> list[str]:
    toks: list[str] = ["CLS"]
    # a1..h8 (file a..h, rank 1..8)
    for rank in range(1, 9):
        for file in range(0, 8):
            sq = chess.square(file, rank - 1)
            toks.append(_piece_token(board.piece_at(sq)))
    # side to move
    toks.append("TURN_W" if board.turn == chess.WHITE else "TURN_B")
    # castling flags
    toks.append("WK_CASTLE_YES" if board.has_kingside_castling_rights(chess.WHITE) else "WK_CASTLE_NO")
    toks.append("WQ_CASTLE_YES" if board.has_queenside_castling_rights(chess.WHITE) else "WQ_CASTLE_NO")
    toks.append("BK_CASTLE_YES" if board.has_kingside_castling_rights(chess.BLACK) else "BK_CASTLE_NO")
    toks.append("BQ_CASTLE_YES" if board.has_queenside_castling_rights(chess.BLACK) else "BQ_CASTLE_NO")
    # en passant
    if board.ep_square is None:
        toks.append("EP_NONE")
    else:
        file_idx = chess.square_file(board.ep_square)  # 0..7
        toks.append(f"EP_{'ABCDEFGH'[file_idx]}")
    # halfmove clock
    toks.append(f"HMC_{board.halfmove_clock}")

    assert len(toks) == 72, f"Expected 72 tokens, got {len(toks)}"
    return toks

# ---------- move encoding (matches your parser) ----------
def encode_move(board: chess.Board, move: chess.Move) -> str:
    """
    normal: "<P>{from}{to}"            e.g., "Pe2e4"
    promo : "<P>{from}{Q/R/B/N}{to}"   e.g., "Pe7Qe8" (promo letter at index 3)
    """
    piece = board.piece_at(move.from_square)
    p = (piece.symbol().upper() if piece else "P")
    fs = chess.square_name(move.from_square)
    ts = chess.square_name(move.to_square)
    if move.promotion:
        L = chess.piece_symbol(move.promotion).upper()
        return f"{p}{fs}{L}{ts}"
    return f"{p}{fs}{ts}"

def legal_moves_list(board: chess.Board) -> list[str]:
    return [encode_move(board, mv) for mv in board.legal_moves]

# ---------- labels & split ----------
def wdl_from_white_pov(result_tag: str) -> int:
    r = (result_tag or "").strip()
    if r == "1-0": return 1
    if r == "0-1": return -1
    return 0  # draw/unknown

def stable_game_bucket(game: chess.pgn.Game) -> int:
    parts = [game.headers.get(k, "") for k in ("Event","Site","Date","Round","White","Black","UTCDate","UTCTime")]
    h = hashlib.sha1("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % 10  # 0..9

def _open_writers() -> Dict[int, io.TextIOBase]:
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    ws: Dict[int, io.TextIOBase] = {}
    for i in range(VAL_SHARD_IDX + 1):
        p = SHARDS_DIR / f"shard_{i:02d}.jsonl"
        ws[i] = open(p, "w", encoding="utf-8", newline="\n")
    return ws

def _close_writers(writers: Dict[int, io.TextIOBase]):
    for f in writers.values():
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            pass
        f.close()

# ---------- PGN iteration ----------
def _iter_pgn_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    # include .txt because your per-game PGNs are saved as .txt files
    for ext in ("*.txt", "*.TXT", "*.pgn", "*.PGN", "*.pgn.gz", "*.PGN.GZ"):
        for p in root.rglob(ext):
            yield p

def _open_text_stream(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")

# ---------- main ----------
def main():
    writers = _open_writers()
    games_seen = 0
    positions_emitted = 0
    files_read = 0

    pgn_sources = list(_iter_pgn_files(INPUT_DIR))
    if not pgn_sources:
        print(f"[error] no PGN files found under {INPUT_DIR}")
        _close_writers(writers)
        return

    pbar = tqdm(total=TARGET_GAMES, desc="Sharding games (to 100k)", unit="game")

    for pgn_path in pgn_sources:
        if games_seen >= TARGET_GAMES:
            break
        files_read += 1
        try:
            with _open_text_stream(pgn_path) as fin:
                while games_seen < TARGET_GAMES:
                    game = chess.pgn.read_game(fin)
                    if game is None:
                        break

                    wdl = wdl_from_white_pov(game.headers.get("Result", ""))
                    bucket = stable_game_bucket(game)
                    shard_idx = VAL_SHARD_IDX if bucket == VAL_SHARD_IDX else (bucket % N_TRAIN_SHARDS)

                    board = game.board()
                    for mv in game.mainline_moves():
                        rec = {
                            "position_1d": tokens_from_board(board),
                            "legal_moves": legal_moves_list(board),
                            "target": encode_move(board, mv),
                            "wdl": int(wdl)
                        }
                        writers[shard_idx].write(json.dumps(rec, ensure_ascii=False) + "\n")
                        positions_emitted += 1
                        board.push(mv)

                    games_seen += 1
                    pbar.update(1)
        except Exception:
            # Skip problematic file and continue
            continue

    pbar.close()
    _close_writers(writers)

    manifest = {
        "source_dir": str(INPUT_DIR.resolve()),
        "files_read": files_read,
        "games_seen": games_seen,
        "positions_emitted": positions_emitted,
        "split": "shard_00..08=train, shard_09=val (per-game stable split)",
        "wdl_perspective": "White"
    }
    with open(SHARDS_DIR / "_pgn_to_shards_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
