# metrics.py
import torch


def batch_metrics(batch, outputs):
    """
    Returns dict of edge_top1, promo_acc (on promo targets only), wdl_acc, promo_n
    """
    if len(outputs) == 4:
        edge_logits, promo_logits, value_logits, _ = outputs
    else:
        edge_logits, promo_logits, value_logits = outputs

    B = edge_logits.size(0)
    mask = batch["edge_mask"].reshape(B, -1)  # [B,4096]
    masked = edge_logits.masked_fill(~mask, float("-inf"))
    pred_edge = masked.argmax(dim=-1)         # [B]
    edge_acc = (pred_edge == batch["edge_target"]).float().mean().item()

    # promo accuracy
    is_promo = batch["is_promo"].bool()
    promo_acc = 0.0
    promo_n = int(is_promo.sum().item())
    if promo_n > 0:
        idxs = torch.nonzero(is_promo, as_tuple=True)[0]
        pe = batch["edge_target"].index_select(0, idxs)
        logits_pe = promo_logits.index_select(0, idxs)   # [Bp,4096,4]
        logits_pe = logits_pe[torch.arange(logits_pe.size(0), device=logits_pe.device), pe, :]  # [Bp,4]
        pmask = batch["promo_mask"].index_select(0, idxs).reshape(len(idxs), 64*64, 4)
        legal = pmask[torch.arange(len(idxs), device=logits_pe.device), pe, :]
        logits_pe = logits_pe.masked_fill(~legal, float("-inf"))
        pred_p = logits_pe.argmax(dim=-1)
        promo_acc = (pred_p == batch["promo_target"].index_select(0, idxs)).float().mean().item()

    # WDL accuracy
    wdl_acc = (value_logits.argmax(dim=-1) == batch["wdl_target"]).float().mean().item()

    return {
        "edge_acc": edge_acc,
        "promo_acc": promo_acc,
        "promo_n": promo_n,
        "wdl_acc": wdl_acc,
    }
