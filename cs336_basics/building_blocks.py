import torch
import torch.nn as nn
from math import sqrt
from einops import einsum, rearrange


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

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        position = torch.arange(max_seq_len)
        k = torch.arange(d_k // 2)
        freq = 1 / (theta ** (2 * k / d_k))
        angles = einsum(position, freq,"i, k -> i k")

        cos = angles.cos()
        sin = angles.sin()

        self.register_buffer("cos_table",cos,persistent=False)
        self.register_buffer("sin_table",sin,persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos
        out = torch.stack([out_even,out_odd],dim=-1)
        out = rearrange(out,"... even odd -> ... (even odd)")
        return out

def softmax(x: torch.Tensor,dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim,keepdim=True).values
    x_shifted = x - x_max
    exp_x = x_shifted.exp()
    return exp_x / exp_x.sum(dim=dim,keepdim=True)

def scaled_dot_product_attention(query: torch.Tensor,key: torch.Tensor,value: torch.Tensor,mask: torch.Tensor | None = None) -> torch.Tensor:
    QK = einsum(query,key,"batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k")
    if mask is not None:
        QK = QK.masked_fill(~mask, float("-inf"))
    QK_softmax = softmax(QK / sqrt(query.shape[-1]),dim=-1)
    return einsum(QK_softmax,value,"batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v")

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,max_seq_len: int | None = None, theta: float | None = None,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        d_k = d_model // num_heads
        d_v = d_model // num_heads
        self.device = device
        self.dtype = dtype
        self.num_heads = num_heads
        self.W_q = Linear(d_model,num_heads * d_k,device=device,dtype=dtype)
        self.W_k = Linear(d_model,num_heads * d_k,device=device,dtype=dtype)
        self.W_v = Linear(d_model,num_heads * d_v,device=device,dtype=dtype)
        self.W_o = Linear(num_heads * d_v,d_model,device=device,dtype=dtype)
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta,d_k,max_seq_len,device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)
        queries = rearrange(queries,"b seq_len (head d_q) -> b head seq_len d_q",head = self.num_heads)
        keys = rearrange(keys,"b seq_len (head d_k) -> b head seq_len d_k",head = self.num_heads)
        values = rearrange(values,"b seq_len (head d_v) -> b head seq_len d_v",head = self.num_heads)
        if self.rope is not None:
            queries = self.rope(queries,token_positions)
            keys = self.rope(keys,token_positions)

        seq_len_q = queries.shape[-2]
        seq_len_k = keys.shape[-2]

        mask = torch.tril(torch.ones(seq_len_q,seq_len_k,device=self.device,dtype=torch.bool))
        out = scaled_dot_product_attention(queries,keys,values,mask)
        return self.W_o(rearrange(out,"b head seq_len d_v -> b seq_len (head d_v)",head = self.num_heads))

class TransformerBlock(nn.Module):
    def __init__(self,d_model: int,
                    num_heads: int,
                    d_ff: int,
                    max_seq_len: int,
                    theta: float,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.attn = CausalMultiHeadSelfAttention(d_model,num_heads,max_seq_len,theta,device,dtype)
        self.ln1 = RMSNorm(d_model,device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model,device=device,dtype=dtype)
        self.ff = SwiGLU(d_model,d_ff,device,dtype)

    def forward(self, x: torch.Tensor):
        token_positions = torch.arange(x.shape[-2],device=self.device,dtype=self.dtype)
        y = x + self.attn(self.ln1(x),token_positions)
        return y + self.ff(self.ln2(y))

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float):
        super().__init__()
        self.input_embed = Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(*[TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta) for i in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.output_embed = Linear(d_model,vocab_size)


    def forward(self, x: torch.Tensor):
        output = self.layers(self.input_embed(x))
        return self.output_embed(self.ln_final(output))