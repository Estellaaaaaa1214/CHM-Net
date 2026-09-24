import torch
from torch import nn
from torch.nn import functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, first, second):
        first = F.normalize(first, dim=1)
        second = F.normalize(second, dim=1)
        logits = first @ second.t() / self.temperature
        labels = torch.arange(first.size(0), device=first.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def heatmap_smoothness_loss(heatmap):
    differences = ((heatmap[:, :, 1:] - heatmap[:, :, :-1]).abs().mean(), (heatmap[:, :, :, 1:] - heatmap[:, :, :, :-1]).abs().mean(), (heatmap[:, :, :, :, 1:] - heatmap[:, :, :, :, :-1]).abs().mean())
    return sum(differences) / 3


def heatmap_sparsity_loss(heatmap):
    return heatmap.mean()
