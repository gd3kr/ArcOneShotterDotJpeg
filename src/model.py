import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 1
    n_attention_layers: int = 8
    n_mlp_layers: int = 8
    mlp_depths = [3] * 4 + [4] * 4 + [5] * 4 + [6] * 4
    max_layer_passes: int = 32
    n_heads: int = 1
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 1800
    vocab_size: int = 11


# class Router(nn.Module):
#     def __init__(self, args: ModelArgs, num_options: int):
#         super().__init__()
#         self.norm = RMSNorm(args.max_seq_len, args.norm_eps)
#         self.fc1 = nn.Linear(args.max_seq_len, 2 * args.max_seq_len)
#         self.fc2 = nn.Linear(2 * args.max_seq_len, num_options)
#         self.num_options = num_options
#         self.exploration_rate = 0.1  # Adjust this value to control exploration

#     def gumbel_softmax(self, logits, tau=1, hard=False, eps=1e-10, dim=-1):
#         gumbels = -torch.empty_like(logits).exponential_().log()
#         gumbels = (logits + gumbels) / tau
#         y_soft = F.softmax(gumbels, dim=dim)
#         if hard:
#             y_hard = torch.zeros_like(logits)
#             _, ind = y_soft.max(dim=dim, keepdim=True)
#             y_hard.scatter_(dim, ind, 1)
#             y = y_hard - y_soft.detach() + y_soft
#         else:
#             y = y_soft
#         return y

#     def forward(self, x):
#         logits = self.fc2(F.gelu(self.fc1(x)))  # Shape: [1, num_options]
        
#         # Add Gaussian noise to logits
#         noise = torch.randn_like(logits) * 0.1
#         noisy_logits = logits + noise

#         # Adjust temperature for exploration
#         exploration_temperature = 1.0 + self.exploration_rate

#         gumbel_out = self.gumbel_softmax(noisy_logits, tau=exploration_temperature, hard=False)

#         # Epsilon-greedy strategy
#         if torch.rand(1).item() < self.exploration_rate:
#             top_k_indices = torch.randint(0, self.num_options, (1, 1))
#         else:
#             _, top_k_indices = torch.topk(gumbel_out, k=1, dim=-1)

#         # Create a one-hot mask
#         mask = torch.zeros_like(gumbel_out)
#         mask.scatter_(-1, top_k_indices, 1.0)

#         masked_selection = mask * gumbel_out
#         masked_selection = masked_selection / masked_selection.sum(dim=-1, keepdim=True)

#         return masked_selection, top_k_indices


class Router(nn.Module):
    def __init__(self, args: ModelArgs, num_options: int):
        super().__init__()
        self.norm = RMSNorm(args.max_seq_len, args.norm_eps)
        self.fc1 = nn.Linear(args.max_seq_len, 2 * args.max_seq_len, bias=False)
        self.fc2 = nn.Linear(2 * args.max_seq_len, num_options, bias=False)
        self.num_options = num_options

    def forward(self, x):
        logits = self.fc2(F.gelu(self.fc1(x)))
        probs = F.softmax(logits, dim=-1)
        return probs
    

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
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.query = nn.Linear(args.max_seq_len, args.max_seq_len, bias=False)
        self.key = nn.Linear(args.max_seq_len, args.max_seq_len, bias=False)
        self.key_x = nn.Linear(args.max_seq_len, args.max_seq_len, bias=False)
        self.key_y = nn.Linear(args.max_seq_len, args.max_seq_len, bias=False)
        self.value = nn.Linear(args.max_seq_len, args.max_seq_len, bias=False)

    def forward(self, x, x_pos_emb, y_pos_emb, attn_mask):
        bsz, seqlen = x.shape

        # Reshape inputs to [bsz, 1, seqlen] for linear layers
        x = x.unsqueeze(1)
        x_pos_emb = x_pos_emb.unsqueeze(1)
        y_pos_emb = y_pos_emb.unsqueeze(1)

        q = self.query(x)
        k = self.key(x)
        k_x = self.key_x(x_pos_emb)
        k_y = self.key_y(y_pos_emb)
        k = k * k_x * k_y

        # Reshape q and k to [bsz, seqlen, seqlen] for attention computation
        q = q.squeeze(1)
        k = k.squeeze(1)

        wei = q @ k.transpose(-2, -1) * (seqlen ** -0.5)
        
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).expand(bsz, -1, -1)

        if attn_mask is not None:
            combined_mask = torch.logical_and(causal_mask, attn_mask.bool())
        else:
            combined_mask = causal_mask

        wei = wei.masked_fill(combined_mask, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x).squeeze(1)  # [bsz, seqlen]
        out = wei @ v.unsqueeze(-1)  # [bsz, seqlen, 1]
        return out.squeeze(-1)  # [bsz, seqlen]
    

class MultiHeadAttention2D(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.heads = nn.ModuleList([Attention2D(args) for _ in range(args.n_heads)])
        self.proj = nn.Linear(args.max_seq_len, args.max_seq_len)

    def forward(self, x, x_pos_emb, y_pos_emb, attn_mask):
        out = torch.cat([attn(x, x_pos_emb, y_pos_emb, attn_mask) for attn in self.heads], dim = -1)
        return self.proj(out)
    

class MLP(nn.Module):
    def __init__(self, args: ModelArgs, hidden_mult: int, depth: int):
        super().__init__()
        self.norm = RMSNorm(args.max_seq_len, args.norm_eps)
        self.fc1 = nn.Linear(args.max_seq_len, hidden_mult * args.max_seq_len)
        self.fcn = nn.ModuleList([nn.Linear(hidden_mult * args.max_seq_len, hidden_mult * args.max_seq_len) for _ in range(depth - 2)])
        self.fc2 = nn.Linear(hidden_mult * args.max_seq_len, args.max_seq_len)

    def forward(self, x):
        # x = self.norm(x)
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

        self.norm = RMSNorm(args.max_seq_len, args.norm_eps)
        # self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # self.x_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        # self.y_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.router = Router(args, args.n_attention_layers + args.n_mlp_layers) 
        self.halt_router = HaltRouter(args)
        global_attention_blocks = nn.ModuleList([MultiHeadAttention2D(args) for _ in range(args.n_attention_layers)])
        global_mlp_blocks = nn.ModuleList([MLP(args, 2, args.mlp_depths[i]) for i in range(args.n_mlp_layers)])
        self.blocks = nn.ModuleList([global_attention_blocks, global_mlp_blocks])
        self.output = nn.Linear(1, args.vocab_size, bias=False)
        self.token_importance = nn.Linear(args.max_seq_len + 1, 1, bias=False)

        self.last_loss = None

    def get_output_token(self, x):
        with torch.no_grad():
            token_logits = self.output(x)
            return token_logits.argmax(dim=-1)

    def forward(self, tokens, x_pos, y_pos, attn_mask, targets = None):
        bsz, seqlen = tokens.shape

        x = tokens.type(torch.float) / self.vocab_size
        x_pos_emb = x_pos.type(torch.float) / 30
        y_pos_emb = y_pos.type(torch.float) / 30

        layer_passes = 0
        while layer_passes < self.max_layer_passes:
            layer_passes += 1

            # Get block weights from router
            block_weights = self.router(x)  # shape: [batch_size, num_blocks]
            
            output = x
            for i in range(self.n_attention_layers + self.n_mlp_layers):
                if i < self.n_attention_layers:
                    block_output = self.blocks[0][i](output, x_pos_emb, y_pos_emb, attn_mask)
                else:
                    block_output = self.blocks[1][i - self.n_attention_layers](output)
                
                # Weight the block's output by the router's selection probability
                output = output + block_weights[:, i].unsqueeze(1) * block_output

            x = output

            # Check if we should halt
            # if (layer_passes + 1) % 4 == 0:
            #     predicted_token = self.get_output_token(x)
            #     new_sequence = torch.cat((tokens, predicted_token), dim=1)
            #     halt_logit = self.halt_router(new_sequence)
            #     halt_prob = torch.sigmoid(halt_logit)
            #     if halt_prob.item() > 0.9:
            #         break

        x_reshaped = x.transpose(0, 1).unsqueeze(-1)

        if targets is not None:
            logits = self.output(x_reshaped)
            logits = logits.transpose(0, 1)

            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=10)
            # self.last_importance = torch.sigmoid(self.token_importance(torch.cat((tokens, logits.argmax(dim=-1)), dim=1)))
        else:
            logits = self.output(x[:, [-1], :])
            self.last_loss = None

        return logits
        


            
