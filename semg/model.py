"""EMG -> character CTC model: strided conv front-end (1000 Hz -> 125 Hz) + small Transformer."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text import CHARS


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 5, stride, 2)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 5, 1, 2)
        self.bn2 = nn.BatchNorm1d(cout)
        self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride), nn.BatchNorm1d(cout))
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        return self.act(self.bn2(self.conv2(y)) + self.skip(x))


class Block(nn.Module):
    """Pre-norm Transformer block written with plain ops so ONNX export keeps a dynamic length."""

    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        attn_mask = None if mask is None else ~mask[:, None, None, :]   # True = attend
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class EMGCTC(nn.Module):
    def __init__(self, n_channels=8, d_model=256, n_layers=6, n_heads=4, ff=1024,
                 dropout=0.2, n_vocab=len(CHARS), n_modes=2):
        super().__init__()
        self.config = dict(n_channels=n_channels, d_model=d_model, n_layers=n_layers,
                           n_heads=n_heads, ff=ff, dropout=dropout, n_vocab=n_vocab, n_modes=n_modes)
        self.frontend = nn.Sequential(ResBlock(n_channels, 128, 2), ResBlock(128, 192, 2),
                                      ResBlock(192, d_model, 2))
        self.mode_emb = nn.Embedding(n_modes, d_model)  # 0 = silent, 1 = voiced
        self.pos_conv = nn.Sequential(nn.Conv1d(d_model, d_model, 31, padding=15, groups=d_model), nn.GELU())
        self.blocks = nn.ModuleList(Block(d_model, n_heads, ff, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_vocab)

    @staticmethod
    def output_lengths(lengths):
        for _ in range(3):
            lengths = (lengths + 1) // 2
        return lengths

    def forward(self, emg, lengths=None, mode=None):
        """emg: (B, T, C) at 1000 Hz. Returns log-probs (B, T/8, V) and output lengths."""
        x = self.frontend(emg.transpose(1, 2))           # (B, D, T')
        x = x + self.pos_conv(x)
        x = x.transpose(1, 2)                            # (B, T', D)
        if mode is None:
            mode = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        x = x + self.mode_emb(mode)[:, None]
        mask = None
        out_lens = None
        if lengths is not None:
            out_lens = self.output_lengths(lengths)
            mask = torch.arange(x.shape[1], device=x.device)[None] >= out_lens[:, None]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.norm(x)).log_softmax(-1), out_lens


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EMGCTC(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt


def adapt_channels(model, n_channels):
    """Return a copy of `model` that takes `n_channels` inputs (e.g. a 4-electrode $50 rig).

    Every weight is kept except the first conv layer (and its skip path), which is
    re-initialised because its input width changes. Calibration then retrains it.
    """
    if n_channels == model.config["n_channels"]:
        return model
    cfg = dict(model.config, n_channels=n_channels)
    new = EMGCTC(**cfg).to(next(model.parameters()).device)
    own = new.state_dict()
    keep = {k: v for k, v in model.state_dict().items() if k in own and own[k].shape == v.shape}
    new.load_state_dict(keep, strict=False)
    return new
