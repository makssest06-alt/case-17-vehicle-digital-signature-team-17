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


def batch_hard_triplet(feat, labels, margin: float = 0.3):
    """Batch-hard triplet: для каждого якоря самый далёкий свой и самый близкий чужой."""
    f = F.normalize(feat)
    dist = torch.cdist(f, f, p=2)
    same = labels.view(-1, 1) == labels.view(1, -1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=f.device)
    pos = dist.masked_fill(~same | eye, -math.inf).max(dim=1).values
    neg = dist.masked_fill(same, math.inf).min(dim=1).values
    return F.softplus(pos - neg) .mean() if margin <= 0 else F.relu(pos - neg + margin).mean()
