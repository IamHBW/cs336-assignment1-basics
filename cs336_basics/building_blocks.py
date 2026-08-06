import torch
import torch.nn as nn
from math import sqrt
from einops import einsum


class Linear(nn.Module):
    """
    y = x @ W.t
    """
    def __init__(self, 
                 in_features, out_features, 
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features,in_features,device=device,dtype=dtype))
        std = sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight,mean = 0,std = std,a = -3 * std,b = 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x,self.weight,"... din, dout din -> ... dout")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weights = torch.zeros(num_embeddings,embedding_dim,device=device,dtype=dtype)
        nn.init.trunc_normal_(self.weights,mean=0,std=1,a=-3,b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, 
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model,dtype = dtype,device = device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Your code here performing RMSNorm
        rms = ((x ** 2).mean(dim=-1,keepdim=True) + self.eps).sqrt()
        result = x * self.gain / rms
        # Return the result in the original dtype
        return result.to(in_dtype)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid()

class SwiGLU(nn.Module):
    def __init__(self, 
                     d_model: int, d_ff: int | None = None,
                     device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        if not d_ff:
            d_ff = ((8 * d_model / 3 + 63) // 64) * 64 #round up

        self.W1 = Linear(d_model,d_ff,dtype=dtype,device=device)
        self.W3 = Linear(d_model,d_ff,dtype=dtype,device=device)
        self.W2 = Linear(d_ff,d_model,dtype=dtype,device=device)
        self.silu = SiLU()

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.silu(self.W1(x)) * self.W3(x))
