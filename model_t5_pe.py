# 10M parameters LLaMa model with T5 Positional Encoding
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from data import encode

class LayerNorm(nn.Module):

    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-6)

class T5RelativePositionBias(nn.Module):
    def __init__(self, scale, num_heads, bidirectional=False, num_buckets=32, max_distance=128):
        super().__init__()
        self.scale = scale
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact + 1e-10)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def forward(self, query_length, key_length, device):
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size, bias=True):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by the number of heads."

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.block_size = block_size

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # T5 Relative Position Bias
        self.rel_pos_bias = T5RelativePositionBias(
            scale=block_size, 
            num_heads=n_head, 
            bidirectional=False,  # Set to False for causal/autoregressive models
            num_buckets=32,
            max_distance=128
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            )

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Compute T5 relative position bias
        rel_bias = self.rel_pos_bias(T, T, x.device)  # (1, num_heads, T, T)

        # Manual attention with T5 relative position bias and causal masking
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + rel_bias  # Apply T5 relative positional bias
        
        if not self.flash:
            # Apply causal mask for non-flash attention
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        else:
            # Create causal mask for flash attention
            causal_mask = torch.tril(torch.ones(T, T, device=att.device, dtype=torch.bool))
            att = att.masked_fill(~causal_mask, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class SwiGLUFFN(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        d_ff = int((8/3) * n_embd)
        self.fc1 = nn.Linear(n_embd, 2 * d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.fc1(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        swish = x1 * torch.sigmoid(x1)
        x = swish * x2
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size, bias=True):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size, bias=bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = SwiGLUFFN(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, dropout, block_size, bias=bias) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_embd, bias=bias),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            logits = self.lm_head(x)

            equal_token_id = 10             
            masked_targets = targets.clone()
            for i in range(masked_targets.shape[0]):
                res = (masked_targets[i] == equal_token_id).nonzero(as_tuple=True)[0]
                
                if res.nelement() > 0:
                    eq_idx = res[0]
                    masked_targets[i, :eq_idx + 1] = encode("*")[0]

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                masked_targets.view(-1), 
                                ignore_index=encode("*")[0])

        return logits, loss