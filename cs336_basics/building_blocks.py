import torch
import torch.nn as nn
from math import sqrt
from einops import einsum
class Linear(nn.Module):
    def __init__(self, 
                 in_features, out_features, 
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features,in_features,device=device,dtype=dtype))
        std = sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight,mean = 0,std = std,a = -3 * std,b = 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x,self.weight,"... din, dout din -> ... dout")
