# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgePromoValueLoss(nn.Module):
    """
    Combined loss:
      - Edge CE on legal edges only (mask -inf for illegal)
      - Promo CE gated to promo targets only
      - Value CE (3-way WDL)
      - DeCov regularizer on [CLS] representation
      - Orthogonality + attention head diversity regularizers
    """
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.smooth = float(cfg.loss.label_smoothing)
        self.edge_w = float(cfg.loss.edge_weight)
        self.promo_w = float(cfg.loss.promo_weight)
        self.value_w = float(cfg.loss.value_weight)
        self.decov_lambda = float(cfg.loss.decov_lambda)
        self.orth_lambda = float(cfg.loss.orth_lambda)

    def _decov_loss(self, cls_h):
        if cls_h is None or cls_h.ndim != 2:
            return next(self.model.parameters()).new_tensor(0.0)
        B, D = cls_h.shape
        if B < 2:
            return cls_h.new_tensor(0.0)
        z = cls_h.float()
        z = z - z.mean(dim=0, keepdim=True)
        C = (z.t() @ z) / (B - 1)  # [D,D]
        diag = torch.diagonal(C)
        off = C - torch.diag(diag)
        off_term = (off ** 2).sum()
        alpha = 0.1
        diag_term = ((diag - 1.0) ** 2).sum()
        return off_term + alpha * diag_term

    def _orth_loss(self):
        tot = 0.0
        count = 0
        for lin in (getattr(self.model, "from_proj", None), getattr(self.model, "to_proj", None)):
            if isinstance(lin, nn.Linear):
                W = lin.weight  # [out,in]
                G = W @ W.t()   # [out,out]
                I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
                tot = tot + ((G - I) ** 2).sum()
                count += 1
        if count == 0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        return tot

    def _head_diversity_loss(self):
        total = 0.0
        device = next(self.model.parameters()).device
        for blk in getattr(self.model, "blocks", []):
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            H = getattr(attn, "nh", None)
            dh = getattr(attn, "dh", None)
            if H is None or dh is None:
                continue
            for lin in (attn.q, attn.k, attn.v):
                W = lin.weight  # [d_model, d_model] => out = H*dh
                if W.shape[0] != H * dh:
                    continue
                V = W.view(H, dh, -1).reshape(H, -1)  # safe reshape
                V = F.normalize(V, p=2, dim=1)
                G = V @ V.t()  # [H,H]
                I = torch.eye(H, device=G.device, dtype=G.dtype)
                off = G - I
                total = total + (off ** 2).sum()
        return torch.as_tensor(total, device=device)

    def forward(self, batch, outputs):
        # outputs: edge_logits, promo_logits, value_logits, cls_hidden
        edge_logits, promo_logits, value_logits, cls_hidden = outputs
        B = edge_logits.size(0)
        mask = batch["edge_mask"].reshape(B, -1)               # [B,4096]
        masked_edge = edge_logits.masked_fill(~mask, float("-inf"))

        # Edge CE
        tgt = batch["edge_target"]                             # [B]
        edge_loss = F.cross_entropy(masked_edge, tgt, label_smoothing=self.smooth)

        # Promo CE (gated to promos only and only legal promo types at target edge)
        is_promo = batch["is_promo"].bool()                   # [B]
        promo_loss = torch.tensor(0.0, device=edge_logits.device)
        if is_promo.any():
            idxs = torch.nonzero(is_promo, as_tuple=True)[0]  # [Bp]
            pe = tgt.index_select(0, idxs)                    # flattened edge indices
            logits_pe = promo_logits.index_select(0, idxs)    # [Bp,4096,4]
            # Gather the correct edge positions
            logits_pe = logits_pe[torch.arange(logits_pe.size(0), device=logits_pe.device), pe, :]  # [Bp,4]
            ptarget = batch["promo_target"].index_select(0, idxs)                                   # [Bp]
            pmask = batch["promo_mask"].index_select(0, idxs).reshape(logits_pe.size(0), 64*64, 4)
            legal = pmask[torch.arange(logits_pe.size(0), device=logits_pe.device), pe, :]          # [Bp,4] bool
            logits_pe = logits_pe.masked_fill(~legal, float("-inf"))
            promo_loss = F.cross_entropy(logits_pe, ptarget)

        # Value CE
        vloss = F.cross_entropy(value_logits, batch["wdl_target"])

        # Regularizers
        reg = torch.tensor(0.0, device=edge_logits.device)
        if self.decov_lambda > 0.0 and cls_hidden is not None:
            reg = reg + self.decov_lambda * self._decov_loss(cls_hidden)
        if self.orth_lambda > 0.0:
            reg = reg + self.orth_lambda * (self._orth_loss() + self._head_diversity_loss())

        total = self.edge_w * edge_loss + self.promo_w * promo_loss + self.value_w * vloss + reg
        return total, {
            "edge_ce": float(edge_loss.detach().cpu().item()),
            "promo_ce": float(promo_loss.detach().cpu().item()),
            "value_ce": float(vloss.detach().cpu().item()),
            "total": float(total.detach().cpu().item()),
        }
