#!/usr/bin/env python3
# download_stockfish_archives_to_100k.py
#
# Download most-recent Stockfish Fishtest LTC archives from HF until they contain
# at least 100,000 games, with a progress bar and ETA to 100k.

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import re
import json
import gzip
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

REPO_ID = "official-stockfish/fishtest_pgns"
RAW_DIR = Path("data/stockfish_ltc_raw")
MANIFEST = RAW_DIR / "_archives_manifest.jsonl"
TARGET_GAMES = 100_000

DATE_RE = re.compile(r"^(?P<yy>\d{2})-(?P<mm>\d{2})-(?P<dd>\d{2})/[^/]+/[^/]+\.pgn\.gz$")

def list_descending_files() -> List[str]:
    api = HfApi()
    paths = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    paths = [p for p in paths if p.endswith(".pgn.gz") and DATE_RE.match(p)]
    def key(p: str) -> Tuple[int,int,int,str]:
        m = DATE_RE.match(p)
        yy, mm, dd = int(m["yy"]), int(m["mm"]), int(m["dd"])
        year = 2000 + yy
        return (-year, -mm, -dd, p[::-1])  # newest date first
    paths.sort(key=key)
    return paths

def fast_count_games_gz(local_path: str) -> int:
    # Count occurrences of PGN header "[Event " — fast & robust.
    cnt = 0
    with gzip.open(local_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("[Event "):
                cnt += 1
    return cnt

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = list_descending_files()
    if not files:
        print("[error] No archives found on Hugging Face")
        return

    # Resume from existing manifest if present
    seen = set()
    current = 0
    if MANIFEST.exists():
        with MANIFEST.open("r", encoding="utf-8") as mf:
            for line in mf:
                try:
                    rec = json.loads(line)
                    seen.add(rec["hf_path"])
                    current += int(rec.get("games_in_archive", 0))
                except Exception:
                    pass

    pbar = tqdm(
        total=TARGET_GAMES,
        initial=min(current, TARGET_GAMES),
        unit="game",
        desc="Downloading (to ≥100,000 games)",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    with MANIFEST.open("a", encoding="utf-8") as mf:
        for relpath in files:
            if current >= TARGET_GAMES:
                break
            if relpath in seen:
                continue

            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=relpath,
                local_dir=str(RAW_DIR),
                local_dir_use_symlinks=False
            )

            try:
                games_in_archive = fast_count_games_gz(local_path)
            except Exception:
                games_in_archive = 0

            # Clamp progress to target to keep ETA sane
            remaining = max(TARGET_GAMES - current, 0)
            pbar.update(min(games_in_archive, remaining))
            current += games_in_archive

            rec = {
                "hf_path": relpath,
                "local_path": str(Path(local_path).resolve()),
                "games_in_archive": int(games_in_archive)
            }
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if current >= TARGET_GAMES:
                break

    pbar.close()
    print(f"[done] Archives sufficient for ≥ {TARGET_GAMES} games (counted={current}).")
    print(f"[info] Manifest: {MANIFEST}")

if __name__ == "__main__":
    main()
