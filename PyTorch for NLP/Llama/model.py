import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class LlamaConfig:
    vocab_size: int = 128256
    context_length: int = 8192
    embed_dim: int = 2048
    mlp_ratio: int = 4 
    n_layers: int = 16
    n_heads: int = 32
    n_kv_groups: int = 4
    rope_base: int = 500000

class RMSNorm(nn.Module):

    def __init__(self,
                 embed_dim, 
                 eps=1e-6):
        
        super(RMSNorm, self).__init__()
        
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(axis=-1, keepdims=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).astype_as(x)
        return output * self.weight
    
class RotaryEmbeddings(nn.Module):
    """
    Direct implementation of Rotary Position Embeddings in the paper RoFormer - https://arxiv.org/pdf/2104.09864.pdf

    Args:
        head_dim: Embedding dimension per head of multiheaded attention
        max_positional_embeddings: Total sequence length we can handle, though because this isnt trained we can just make it longer
        base: Base of theta formula from paper -> theta = base ** (-2i/d) where i in [1, 2, ... dim/2]
    """
    def __init__(self, head_dim=64, max_position_embeddings=4096, base=10000):
        super(RotaryEmbeddings, self).__init__()
        assert head_dim % 2 == 0, f"Make sure your head dimension {head_dim} is an even number!"

        self.dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freqs = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freqs", inv_freqs, persistent=False)

        ### Build Cache for indexing ###
        self._build_sin_cos_cache()

    def _build_sin_cos_cache(self, seq_len=None):
        """
        Helper function to make the the sin/cosine cache which stores all positional information
        """
        t = torch.arange(self.max_position_embeddings,
                         device=self.inv_freqs.device, 
                         dtype=self.inv_freqs.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freqs)
        emb = freqs.unsqueeze(-1).repeat(1,1,2).flatten(start_dim=1)
        self.cos_cached = nn.Parameter(emb.cos()[None, None, ...], requires_grad=False)
        self.sin_cached = nn.Parameter(emb.sin()[None, None, ...], requires_grad=False)
    
    def rotate_half(self, x):
        """
        Converts an embedding sequence like [1,2,3,4] to [-2,1,-4,3] for each sample, and for each head of attention
        """
        b,h,s,d = x.shape
        x = x.reshape(b,h,s,d//2,2).flip(-1)
        x[...,0] = x[...,0]*-1
        x = x.flatten(start_dim=-2)
        return x
        
    def forward(self, x, start_index=0, seq_len=None):
        """
        Args:
            q: Queries matrix in the shape [batch_size x num_heads x seq_len x head_dim]
            k: Keys matrix in the shape [batch_size x num_heads x seq_len x head_dim] (None for Cross Attention)
            start_index: Position of token index w.r.t the entire sequence
            seq_len: Number of tokens in the sample
        """

        ### Double check the sequence length ###
        if seq_len is None:
            seq_len = x.shape[-2]
        elif x.shape[-2] < self.max_position_embeddings:
            seq_len = x.shape[-2]

        ### Grab Sin and Cos Cache at Wanted Index ###
        sin, cos = self.sin_cached[:,:,start_index:start_index+seq_len,:], \
            self.cos_cached[:,:,start_index:start_index+seq_len,:]

        x_rot = (x*cos) + (self.rotate_half(x) * sin)

        return x_rot