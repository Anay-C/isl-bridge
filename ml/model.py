"""Sign encoder (small Transformer over keypoint frames) + cosine classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignModel(nn.Module):
    def __init__(self, feat_dim, n_classes, d=192, layers=4, heads=4, emb=256, t_max=64,
                 scale=16.0, margin=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(feat_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.1))
        self.pos = nn.Parameter(torch.zeros(1, t_max, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.2, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, emb))
        self.weight = nn.Parameter(torch.randn(n_classes, emb) * 0.01)
        self.scale, self.margin = scale, margin
        nn.init.trunc_normal_(self.pos, std=0.02)

    def embed(self, x):
        """x (B, T, feat_dim) -> L2-normalised embedding (B, emb)."""
        h = self.inp(x) + self.pos[:, : x.shape[1]]
        h = torch.cat([self.cls.expand(len(x), -1, -1), h], 1)
        h = self.enc(h)
        z = self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1))
        return F.normalize(z, dim=-1)

    def logits(self, z, margin_labels=None):
        cos = z @ F.normalize(self.weight, dim=-1).t()
        if margin_labels is not None:  # CosFace margin during training -> tighter, prototype-friendly clusters
            cos = cos - self.margin * F.one_hot(margin_labels, cos.shape[1])
        return self.scale * cos
