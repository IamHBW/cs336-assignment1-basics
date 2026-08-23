import torch
import torch.nn as nn
from math import sqrt,cos,pi
from collections.abc import Callable, Iterable
from typing import Optional
import numpy.typing as npt
import numpy as np

def cross_entropy_loss(logits: torch.Tensor,target: torch.Tensor):
    logits -= logits.max(dim=-1,keepdim=True).values
    row_idx = torch.arange(logits.shape[0])
    return (-logits[row_idx,target] + logits.exp().sum(dim=-1).log()).mean()

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, weight_decay, lr=1e-3, betas=(0.9,0.999), eps=1e-8):
        beta1 = betas[0]
        beta2 = betas[1]
        defaults = {"lr": lr,"weight_decay": weight_decay,"beta1": beta1,"beta2": beta2,"eps": eps}
        super().__init__(params, defaults)
        

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            weight_decay = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 0.
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data  # Get the gradient of loss with respect to p.

                lr_t = lr * sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr * weight_decay * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                p.data -= lr_t * m / (v.sqrt() + eps)  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss

def lr_cosine_scheduler(t,lr_max,lr_min,T_w,T_c):
    if t < T_w:
        return t / T_w * lr_max
    elif t <= T_c:
        return lr_min + (1 + cos((t - T_w) * pi / (T_c - T_w))) * (lr_max - lr_min) / 2
    else:
        return lr_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    grad_l2_norm_sqared = 0
    for param in parameters:
        if param.grad is None:
            continue
        grad_l2_norm_sqared += param.grad.norm(p=2) ** 2

    if grad_l2_norm_sqared >= max_l2_norm ** 2:
        for param in parameters:
            if param.grad is not None:
                param.grad *= max_l2_norm / (grad_l2_norm_sqared.sqrt() + eps)

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    input_seq = torch.zeros(batch_size,context_length,dtype=torch.long,device=device)
    targets = torch.zeros(batch_size,context_length,dtype=torch.long,device=device)
    for i in range(batch_size):
        start = np.random.randint(0,n - context_length)
        chunk = dataset[start:start + context_length + 1]
        input_seq[i] = torch.as_tensor(chunk[:-1],dtype=torch.long,device=device)
        targets[i] = torch.as_tensor(chunk[1:],dtype=torch.long,device=device)
    return (input_seq,targets)