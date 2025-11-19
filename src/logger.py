# logger.py
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def get_logger(name: str, logs_dir: str, level: int = logging.INFO) -> logging.Logger:
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    lg.setLevel(level)
    lg.propagate = False
    if not lg.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"))
        lg.addHandler(ch)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(logs / f"{name}_{ts}.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        lg.addHandler(fh)
    return lg


class CSVLogger:
    def __init__(self, out_csv: Path, fieldnames: List[str]):
        self.out_csv = Path(out_csv)
        self.fieldnames = list(fieldnames)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def append(self, row: Dict[str, float]):
        with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            safe_row = {k: row.get(k, "") for k in self.fieldnames}
            writer.writerow(safe_row)
