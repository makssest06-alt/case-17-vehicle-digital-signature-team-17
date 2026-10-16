"""Обучаемая голова поверх замороженных признаков CLIP.

Backbone не трогаем вообще: он посчитан один раз и лежит в .npy.
Голова учится на CPU за минуты, потому что видит уже готовые векторы 768D.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReidHead(nn.Module):
    """Linear(in->hidden) -> BN -> ReLU -> Dropout -> Linear(hidden->out) -> BNNeck.

    BNNeck: для triplet-лосса берём вектор ДО BN, для классификационного — ПОСЛЕ.
    Это стандартный приём ReID: два лосса тянут признак в разные стороны,
    и BN между ними снимает конфликт.
    """

    def __init__(self, in_dim: int = 768, hidden: int = 1024, out_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.bnneck = nn.BatchNorm1d(out_dim)
        self.bnneck.bias.requires_grad_(False)  # как в strong baseline

    def forward(self, x):
        h = self.drop(F.relu(self.bn1(self.fc1(x))))
        feat = self.fc2(h)          # до BNNeck — для triplet
        feat_bn = self.bnneck(feat)  # после BNNeck — для классификации и инференса
        return feat, feat_bn


class MarginHead(nn.Module):
    """ArcFace или CosFace поверх нормированных признаков."""

    def __init__(self, dim: int, num_classes: int, kind: str = "arcface", scale: float = 30.0, margin: float = 0.25):
        super().__init__()
        self.kind, self.s, self.m = kind, scale, margin
        self.weight = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feat_bn, labels):
        cos = F.linear(F.normalize(feat_bn), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        if self.kind == "cosface":
            target = cos - self.m
        else:
            theta = torch.acos(cos)
            target = torch.cos(theta + self.m)
        onehot = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1.0)
        return F.cross_entropy(self.s * (onehot * target + (1 - onehot) * cos), labels)


class SubcenterMarginHead(nn.Module):
    """ArcFace classifier with multiple prototypes per identity.

    Reduce each class' prototype cosine scores by max before applying the
    target-only angular margin. One center is numerically equivalent to
    MarginHead for ArcFace.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        num_subcenters: int = 2,
        scale: float = 30.0,
        margin: float = 0.25,
    ) -> None:
        super().__init__()
        if num_subcenters < 1:
            raise ValueError("num_subcenters must be >= 1")
        self.s = scale
        self.m = margin
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters
        self.weight = nn.Parameter(torch.empty(num_classes, num_subcenters, dim))
        nn.init.xavier_uniform_(self.weight)

    def class_logits(self, feat_bn: torch.Tensor) -> torch.Tensor:
        batch_size = feat_bn.shape[0]
        features = F.normalize(feat_bn, dim=1)
        centers = F.normalize(self.weight, dim=2).reshape(
            self.num_classes * self.num_subcenters, -1
        )
        cosine = F.linear(features, centers).reshape(
            batch_size, self.num_classes, self.num_subcenters
        )
        return cosine.amax(dim=2).clamp(-1 + 1e-7, 1 - 1e-7)

    def forward(self, feat_bn: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = self.class_logits(feat_bn)
        target = torch.cos(torch.acos(cosine) + self.m)
        onehot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        logits = onehot * target + (1.0 - onehot) * cosine
        return F.cross_entropy(self.s * logits, labels)


def batch_hard_triplet(feat, labels, margin: float = 0.3):
    """Batch-hard triplet: для каждого якоря самый далёкий свой и самый близкий чужой."""
    f = F.normalize(feat)
    dist = torch.cdist(f, f, p=2)
    same = labels.view(-1, 1) == labels.view(1, -1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=f.device)
    pos = dist.masked_fill(~same | eye, -math.inf).max(dim=1).values
    neg = dist.masked_fill(same, math.inf).min(dim=1).values
    return F.softplus(pos - neg) .mean() if margin <= 0 else F.relu(pos - neg + margin).mean()
