import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 128
    n_attention_layers: int = 16
    n_mlp_layers: int = 16
    mlp_depths = [3] * 4 + [4] * 4 + [5] * 4 + [6] * 4
    max_layer_passes: int = 64
    n_heads: int = 4
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 2049
    vocab_size: int = 11


class Router(nn.Module):
    def __init__(self, args: ModelArgs, dim: int, num_options: int):
        super().__init__()
        self.norm = RMSNorm(dim, args.norm_eps)
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.fc2 = nn.Linear(2 * dim, num_options)

    def forward(self, x):
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)
    

class HaltRouter(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.max_seq_len + 1, 2 * (args.max_seq_len + 1))
        self.fc2 = nn.Linear(2 * (args.max_seq_len + 1), 1)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


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
    

class Attention2D(nn.Module):
    def _init__(self, args: ModelArgs):
        self.query = nn.Linear(args.dim, args.dim // args.n_heads, bias=False)
        self.key = nn.Linear(args.dim, args.dim // args.n_heads, bias=False)
        self.key_x = nn.Linear(args.dim, args.dim // args.n_heads, bias=False)
        self.key_y = nn.Linear(args.dim, args.dim // args.n_heads, bias=False)
        self.value = nn.Linear(args.dim, args.dim // args.n_heads, bias=False)

    def forward(self, x, x_pos_emb, y_pos_emb, attn_mask):
        bsz, seqlen, dim = x.shape

        q = self.query(x)

        k = self.key(x)
        k_x = self.key_x(x_pos_emb)
        k_y = self.key_y(y_pos_emb)
        k = k * k_x * k_y

        wei = q @ k.transpose(-2,-1) * (dim ** -0.5)
        # TODO: add mask (combin attn_mask and causal_mask)
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out
    

class MultiHeadAttention2D(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.heads = nn.ModuleList([Attention2D(args) for _ in range(args.n_heads)])
        self.proj = nn.Linear(args.dim, args.dim)

    def forward(self, x, x_pos_emb, y_pos_emb, attn_mask):
        out = torch.cat([attn(x, x_pos_emb, y_pos_emb, attn_mask) for attn in self.heads], dim = -1)
        return self.proj(out)
    

class MLP(nn.Module):
    def __init__(self, args: ModelArgs, hidden_mult: int, depth: int):
        super().__init__()
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.fc1 = nn.Linear(args.dim, hidden_mult * args.dim)
        self.fcn = nn.ModuleList([nn.Linear(hidden_mult * args.dim, hidden_mult * args.dim) for _ in range(depth - 2)])
        self.fc2 = nn.Linear(hidden_mult * args.dim, args.dim)

    def forward(self, x):
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        for fc in self.fcn:
            x = F.gelu(fc(x))
        return self.fc2(x)
    

class Model(nn.Module):
    last_importance: Optional[torch.Tensor]
    last_loss: Optional[torch.Tensor]

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_attention_layers = args.n_attention_layers
        self.n_mlp_layers = args.n_mlp_layers
        self.max_layer_passes = args.max_layer_passes
        self.vocab_size = args.vocab_size

        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.x_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.y_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.router = Router(args, args.dim, args.n_attention_layers + args.n_mlp_layers) 
        self.halt_router = HaltRouter(args)
        global_attention_blocks = nn.ModuleList([MultiHeadAttention2D(args) for _ in range(args.n_attention_layers)])
        global_mlp_blocks = nn.ModuleList([MLP(args, 4, args.mlp_depths[i]) for i in range(args.n_mlp_layers)])
        self.blocks = nn.ModuleList([global_attention_blocks, global_mlp_blocks])
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.token_importance = nn.Linear(args.max_seq_len + 1, 1, bias=False)

        self.last_loss = None

    def get_output_token(self, x):
        with torch.no_grad():
            token_logits = self.output(x)
            return token_logits.argmax(dim=-1)

    def forward(self, tokens, x_pos, y_pos, attn_mask, targets = None):
        bsz, seqlen = tokens.shape

        x = self.tok_embeddings(tokens)
        x_pos_emb = self.x_embeddings(x_pos)
        y_pos_emb = self.y_embeddings(y_pos)

        layer_passes = 0
        while layer_passes < self.max_layer_passes:
            layer_passes += 1

            # Pass over selected blocks
            router_logits = self.router(x)
            weights, selected_layers = torch.topk(router_logits, 1, dim=-1)
            weights = F.softmax(weights, dim=-1)
            results = torch.zeros_like(x)
            for i, option in enumerate(self.blocks):
                batch_idx, nth_block = torch.where(selected_layers == i)
                results[batch_idx] = weights[batch_idx, nth_block, None] * option(x[batch_idx], x_pos_emb[batch_idx], y_pos_emb[batch_idx], attn_mask[batch_idx])

            # Check if we should halt
            if (layer_passes + 1) % 4 == 0:
                predicted_token = self.get_output_token(x)
                new_sequence = torch.cat((tokens, predicted_token), dim=1)
                halt_logit = self.halt_router(new_sequence)
                halt_prob = torch.sigmoid(halt_logit)
                if halt_prob.item() > 0.9:
                    break

        x = self.norm(x)

        if targets is not None:
            logits = self.output(x)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=10)
            self.last_importance = self.token_importance(torch.cat((tokens, logits.argmax(dim=-1)), dim=1))
        else:
            logits = self.output(x[:, [-1], :])
            self.last_loss = None

        return logits
        


            
