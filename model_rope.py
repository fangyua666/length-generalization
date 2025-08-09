import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-6)


def rotate_half(x):
    """
    Rotates half the hidden dimensions of the input.
    This is used to apply the RoPE transformation.
    Input x has shape [..., seq_len, num_heads, head_dim] or [..., num_heads, seq_len, head_dim]
    or [..., seq_len, head_dim]
    """
    # Get the last dimension (head_dim)
    last_dim = x.shape[-1]
    # Split the last dimension into two halves
    x1 = x[..., : last_dim // 2]
    x2 = x[..., last_dim // 2 :]
    # Concatenate with the second half negated and then the first half
    # This corresponds to the rotation matrix application:
    # [x_j, x_{j+d/2}] -> [-x_{j+d/2}, x_j]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary positional embedding to query and key tensors.

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        cos: Cosine values of shape (1, 1, seq_len, head_dim)
        sin: Sine values of shape (1, 1, seq_len, head_dim)

    Returns:
        Tuple of rotated query and key tensors
    """
    # Apply rotary embedding
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation.
    """

    def __init__(self, head_dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute the rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin values for maximum sequence length
        self._precompute_cos_sin(max_seq_len)

    def _precompute_cos_sin(self, seq_len):
        """Precompute cosine and sine values for given sequence length."""
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim // 2)

        # Create the full frequency tensor by repeating each frequency
        freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)

        cos = freqs.cos()
        sin = freqs.sin()

        # Add batch and head dimensions: (1, 1, seq_len, head_dim)
        self.register_buffer('cos_cached', cos[None, None, :, :])
        self.register_buffer('sin_cached', sin[None, None, :, :])

    def forward(self, seq_len):
        """
        Return cosine and sine values for the given sequence length.

        Args:
            seq_len: Sequence length

        Returns:
            Tuple of (cos, sin) tensors of shape (1, 1, seq_len, head_dim)
        """
        if seq_len > self.max_seq_len:
            # Recompute for longer sequences
            self._precompute_cos_sin(seq_len)
            self.max_seq_len = seq_len

        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size, bias=True):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by the number of heads."

        # Store hyperparameters
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.block_size = block_size

        # Key, Query, Value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Rotary Positional Embedding
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=block_size)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Check for Flash Attention availability
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask for slow attention
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            )

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # Split into Q, K, V (B, T, n_embd)

        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)

        # Apply Rotary Positional Embedding
        cos, sin = self.rope(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Flash Attention or fallback to manual implementation
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual attention with causal masking
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # Scaled dot product
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # Apply causal mask
            att = F.softmax(att, dim=-1)  # Normalize attention scores
            att = self.attn_dropout(att)
            y = att @ v  # Apply attention weights to values (B, n_head, T, head_size)

        # Reshape back to original format
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Reassemble heads

        # Output projection and residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


# SwiGLU used in llama
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
        # LayerNorm and CausalSelfAttention with explicit parameters
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size, bias=bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = SwiGLUFFN(n_embd, dropout)

    def forward(self, x):
        # Apply residual connection and pre-normalization
        x = x + self.attn(self.ln_1(x))  # Apply LayerNorm before attention
        x = x + self.mlp(self.ln_2(x))  # Apply LayerNorm before MLP
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
            wte = nn.Embedding(vocab_size, n_embd),  # token embeddings
            # Note: No positional embeddings (wpe) since we're using RoPE
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, dropout, block_size, bias=bias) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_embd, bias=bias),  # final layer norm
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # init all weights
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

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        # No need to add positional embeddings since RoPE is applied in attention
        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # Assuming encode function exists and returns the padding token index
            # You'll need to adjust this based on your actual encode function
            padding_token_idx = 12  # As defined in your original code
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=padding_token_idx)
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss