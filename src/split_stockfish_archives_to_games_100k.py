#!/usr/bin/env python3
# split_stockfish_archives_to_games_100k.py
#
# Split downloaded *.pgn.gz archives (newest-first) into ONE-PGN-PER-TXT under
# data/games/stockfish/, with progress bar & ETA to exactly 100,000 games.

from __future__ import annotations
from pathlib import Path
import io
import re
import json
import gzip
from datetime import datetime
from typing import Tuple

import chess.pgn
from tqdm import tqdm

RAW_DIR = Path("data/stockfish_ltc_raw")
MANIFEST = RAW_DIR / "_archives_manifest.jsonl"
OUT_DIR = Path("data/games/stockfish")
TARGET_GAMES = 100_000

DATE_RE = re.compile(r".*/(?P<yy>\d{2})-(?P<mm>\d{2})-(?P<dd>\d{2})/(?P<testid>[^/]+)/[^/]+\.pgn\.gz$")

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")

def _white_utc_timestamp(game: chess.pgn.Game, fallback_date: Tuple[int,int,int]) -> str:
    udate = (game.headers.get("UTCDate") or "").strip()
    utime = (game.headers.get("UTCTime") or "").strip()
    try:
        if udate and utime:
            dt = datetime.strptime(udate + " " + utime, "%Y.%m.%d %H:%M:%S")
            return dt.strftime("%Y-%m-%dT%H-%M-%SZ")
    except Exception:
        pass
    y, m, d = fallback_date
    return f"{y:04d}-{m:02d}-{d:02d}T00-00-00Z"

def _build_filename(game: chess.pgn.Game, day_tuple: Tuple[int,int,int], test_id: str, idx: int) -> Path:
    ts = _white_utc_timestamp(game, day_tuple)
    name = f"{ts}_ltc_chess_{day_tuple[0]%100:02d}-{day_tuple[1]:02d}-{day_tuple[2]:02d}_{_safe(test_id)}_g{idx}.txt"
    return OUT_DIR / name

def _pgn_string(game: chess.pgn.Game) -> str:
    sio = io.StringIO()
    print(game, file=sio, end="\n")
    return sio.getvalue()

def _iter_manifest_records():
    # Manifest is newest-first by the downloader; preserve order
    with MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                yield rec["hf_path"], rec["local_path"]
            except Exception:
                continue

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume: count already-saved per-game files
    existing = sum(1 for _ in OUT_DIR.glob("*.txt"))
    total = existing
    written = 0

    pbar = tqdm(
        total=TARGET_GAMES,
        initial=min(total, TARGET_GAMES),
        unit="game",
        desc="Splitting (to exactly 100,000)",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for hf_path, local_path in _iter_manifest_records():
        if total >= TARGET_GAMES:
            break

        m = DATE_RE.match(local_path.replace("\\", "/"))
        if not m:
            continue
        yy, mm, dd = int(m["yy"]), int(m["mm"]), int(m["dd"])
        year = 2000 + yy  # kept if you ever want year elsewhere
        test_id = m["testid"]

        try:
            with gzip.open(local_path, "rt", encoding="utf-8", errors="replace") as fin:
                idx = 0
                while total < TARGET_GAMES:
                    game = chess.pgn.read_game(fin)
                    if game is None:
                        break
                    idx += 1
                    out_path = _build_filename(game, (year, mm, dd), test_id, idx)

                    if out_path.exists() and out_path.stat().st_size > 0:
                        total += 1
                        pbar.update(1)
                        continue

                    try:
                        pgn = _pgn_string(game)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp = out_path.with_suffix(out_path.suffix + ".part")
                        with tmp.open("w", encoding="utf-8", newline="\n") as fw:
                            fw.write(pgn.strip() + "\n")
                        tmp.replace(out_path)
                        written += 1
                        total += 1
                        pbar.update(1)
                    except Exception:
                        continue
        except Exception:
            continue

    pbar.close()
    summary = {
        "target_games": TARGET_GAMES,
        "existing_at_start": existing,
        "newly_written": written,
        "total_games_present": total,
        "output_dir": str(OUT_DIR.resolve())
    }
    with (OUT_DIR / "_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] wrote {written} new games; total now {total} at {OUT_DIR}")

if __name__ == "__main__":
    main()
