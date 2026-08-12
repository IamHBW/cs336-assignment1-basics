import torch
import torch.nn as nn
from math import sqrt


def cross_entropy_loss(logits: torch.Tensor,target: torch.Tensor):
    logits -= logits.max(dim=-1,keepdim=True).values
    row_idx = torch.arange(logits.shape[0])
    return (-logits[row_idx,target] + logits.exp().sum(dim=-1).log()).mean()