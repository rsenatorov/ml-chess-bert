# model.py
# Edge-head Transformer with RoPE, Value head, and Promo head (bidirectional attention)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w2 = nn.Linear(d_in, d_hidden, bias=False)
        self.w3 = nn.Linear(d_hidden, d_in, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 512):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv)
        t = torch.arange(max_seq_len).type_as(inv)
        freqs = torch.einsum("i,j->ij", t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :])

    def _apply_rope(self, x):
        B, T, H, D = x.shape
        D2 = D // 2
        xv = x.reshape(B, T, H, D2, 2)
        cos = self.cos_cached[:, :T, :, :D2].to(x.dtype)
        sin = self.sin_cached[:, :T, :, :D2].to(x.dtype)
        x0, x1 = xv[..., 0], xv[..., 1]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        return torch.stack([out0, out1], dim=-1).flatten(-2)

    def forward(self, x):
        D = x.shape[-1]
        if D % 2 != 0:
            rot = self._apply_rope(x[..., :-1])
            return torch.cat([rot, x[..., -1:]], dim=-1)
        return self._apply_rope(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.nh = n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, rope: RotaryEmbedding):
        B, T, C = x.shape
        q = self.q(x).reshape(B, T, self.nh, self.dh)
        k = self.k(x).reshape(B, T, self.nh, self.dh)
        v = self.v(x).reshape(B, T, self.nh, self.dh)
        q = rope(q)
        k = rope(k)
        q = q.transpose(1, 2)  # [B,H,T,D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q.float() @ k.float().transpose(-2, -1)) * (self.dh ** -0.5)
        att = F.softmax(att, dim=-1, dtype=torch.float32).to(v.dtype)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.o(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, rope):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)
        self.drop = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, x):
        h = self.attn(self.ln1(x), self.rope)
        x = x + self.drop(h)
        h = self.mlp(self.ln2(x))
        x = x + self.drop(h)
        return x


class ChessEdgeModel(nn.Module):
    """
    Inputs: tokens [B,72] (encoded from position_1d)
    Outputs:
      edge_logits_flat: [B,4096]
      promo_logits_grid: [B,4096,4]  (meaningful only on promo-legal edges)
      value_logits: [B,3]
      cls_hidden: [B,d_model]
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.model.d_model

        self.emb = nn.Embedding(128, d)
        self.drop = nn.Dropout(cfg.model.dropout)

        head_dim = d // cfg.model.n_heads
        self.rope = RotaryEmbedding(head_dim, cfg.model.rope_theta, cfg.model.max_seq_len)
        self.blocks = nn.ModuleList([
            Block(d, cfg.model.n_heads, cfg.model.d_ff, cfg.model.dropout, self.rope)
            for _ in range(cfg.model.n_layers)
        ])
        self.ln_f = RMSNorm(d)

        # Edge head: bilinear + displacement bias
        pd = cfg.model.edge_proj_dim
        self.from_proj = nn.Linear(d, pd, bias=False)
        self.to_proj   = nn.Linear(d, pd, bias=False)
        self.bias_table = nn.Parameter(torch.zeros(15, 15)) if cfg.model.displacement_bias else None

        # Promo head MLP on concatenated from/to projections
        self.promo_mlp = nn.Sequential(
            nn.Linear(pd * 2, 128), nn.GELU(), nn.Dropout(cfg.model.dropout),
            nn.Linear(128, 4)
        )

        # Value head from [CLS]
        self.value_head = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Dropout(cfg.model.dropout),
            nn.Linear(256, 3)
        )

    def forward(self, tokens):
        B, T = tokens.shape
        assert T == self.cfg.model.max_seq_len, f"Expected {self.cfg.model.max_seq_len} tokens, got {T}"
        x = self.emb(tokens)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)          # [B,72,d]

        cls = x[:, 0, :]          # [B,d]
        squares = x[:, 1:65, :]   # [B,64,d]

        # Project from/to
        Sf = self.from_proj(squares)   # [B,64,pd]
        St = self.to_proj(squares)     # [B,64,pd]

        # Bilinear scores per edge
        pd = Sf.shape[-1]
        E = torch.matmul(Sf, St.transpose(1, 2)) / math.sqrt(pd)  # [B,64,64]

        if self.bias_table is not None:
            device = E.device
            idxs = torch.arange(64, device=device)
            fi = idxs % 8
            ri = idxs // 8
            fj = idxs % 8
            rj = idxs // 8
            DX = (fj[None, :] - fi[:, None] + 7).clamp(0, 14)  # [64,64]
            DY = (rj[None, :] - ri[:, None] + 7).clamp(0, 14)
            Bdisp = self.bias_table[DX, DY]                    # [64,64]
            E = E + Bdisp.unsqueeze(0)

        edge_logits = E.reshape(B, 64 * 64)                    # [B,4096]

        # Promo logits per edge (compute for all, gate later)
        Sf_exp = Sf.unsqueeze(2).expand(B, 64, 64, pd)
        St_exp = St.unsqueeze(1).expand(B, 64, 64, pd)
        cat = torch.cat([Sf_exp, St_exp], dim=-1)              # [B,64,64,2pd]
        promo_logits = self.promo_mlp(cat).reshape(B, 64 * 64, 4)  # [B,4096,4]

        value_logits = self.value_head(cls)                    # [B,3]

        return edge_logits, promo_logits, value_logits, cls
