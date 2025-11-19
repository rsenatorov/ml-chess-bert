# plots.py
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _plot_series(x, y, label):
    plt.plot(x, y, label=label)


def plot_recent_train(recent: Dict[str, List[float]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    if recent.get("loss"):
        _plot_series(range(len(recent["loss"])), recent["loss"], "loss")
    if recent.get("edge_acc"):
        _plot_series(range(len(recent["edge_acc"])), recent["edge_acc"], "edge_acc")
    if recent.get("promo_acc"):
        _plot_series(range(len(recent["promo_acc"])), recent["promo_acc"], "promo_acc")
    if recent.get("wdl_acc"):
        _plot_series(range(len(recent["wdl_acc"])), recent["wdl_acc"], "wdl_acc")
    plt.xlabel("recent steps")
    plt.ylabel("value")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_val_history(history: Dict[str, List[float]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for k in ("edge_acc", "promo_acc", "wdl_acc"):
        if k in history and history[k]:
            _plot_series(range(1, len(history[k]) + 1), history[k], k)
    if "loss" in history and history["loss"]:
        _plot_series(range(1, len(history["loss"]) + 1), history["loss"], "val_loss")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
