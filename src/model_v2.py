import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 512
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    vocab_size: int = 11
    hidden_dim: Optional[int] = 2048
    norm_eps: float = 1e-5
    max_seq_len: int = 1800


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

class AttentionHead(nn.Module):
    def __init__(self, dim: int, head_size: int):
        super().__init__()
        self.query = nn.Linear(dim, head_size, bias=False)
        self.key = nn.Linear(dim, head_size, bias=False)
        self.key_pos = nn.Linear(dim, head_size, bias=False)
        self.value = nn.Linear(dim, head_size, bias=False)
        
    def forward(self, x, rel_pos, attn_mask: Optional[torch.Tensor] = None):
        bsz, seqlen, dim = x.shape

        q = self.query(x) # [bsz, seqlen, head_size]

        k = self.key(x) # [bsz, seqlen, head_size]
        k_pos = self.key_pos(rel_pos) # [bsz, seqlen, head_size]
        k = k * k_pos

        wei = q @ k.transpose(-2, -1) / (dim ** 0.5) # [bsz, seqlen, seqlen]

        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).expand(bsz, -1, -1)

        if attn_mask is not None:
            combined_mask = torch.logical_and(causal_mask, attn_mask.bool())
        else:
            combined_mask = causal_mask

        wei = wei.masked_fill(combined_mask, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x) # [bsz, seqlen, head_size]

        out = wei @ v # [bsz, seqlen, head_size]
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, dim, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(dim, head_size) for _ in range(num_head)])
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rel_pos, attn_mask: Optional[torch.Tensor] = None):
        out = torch.cat([h(x, rel_pos, attn_mask) for h in self.heads], dim=-1)
        return self.proj(out)
    

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = MultiHeadAttention(self.n_heads, self.dim, self.head_dim)
        self.mlp = MLP(self.dim, args.hidden_dim)
        self.attention_norm = RMSNorm(self.dim, args.norm_eps)
        self.mlp_norm = RMSNorm(self.dim, args.norm_eps)

    def forward(self, x, rel_pos, attn_mask):
        h = x + self.attention(self.attention_norm(x), rel_pos, attn_mask)
        out = h + self.mlp(self.mlp_norm(h))
        return out
    

class Model(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.rel_pos_embeddings = nn.Linear(2, params.dim)
        self.layers = nn.ModuleList([TransformerBlock(params) for _ in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size)

    def forward(self, tokens, x_pos, y_pos, attn_mask, targets: Optional[torch.Tensor] = None):
        h = self.tok_embeddings(tokens)
        rel_pos = self.rel_pos_embeddings(torch.stack([x_pos, y_pos], dim=-1).to(torch.float))

        for layer in self.layers:
            h = layer(h, rel_pos, attn_mask)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=10)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits