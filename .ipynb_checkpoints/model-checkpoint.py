"""
GPT model components following Raschka's
"Build a Large Language Model From Scratch" conventions.

Expected cfg dict keys:
    vocab_size, context_length, emb_dim, n_heads,
    n_layers, drop_rate, qkv_bias
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# GELU activation
# ---------------------------------------------------------------------------

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (approximate tanh version).
    Used instead of ReLU inside the feed-forward sublayer because it is
    smooth and non-zero for negative inputs, which empirically trains better
    for transformer language models.
    """

    def forward(self, x):
        return 0.5 * x * (
            1.0 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi))
                * (x + 0.044715 * x ** 3)
            )
        )


# ---------------------------------------------------------------------------
# Custom LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """
    Layer normalisation with learnable scale and shift (no dependency on
    batch size).  Written from scratch so every parameter is explicit;
    eps guards against division by zero on near-constant activations.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        # Learnable affine parameters, one per embedding dimension
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift


# ---------------------------------------------------------------------------
# FeedForward sublayer
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Position-wise two-layer MLP with a 4× hidden expansion.
    The bottleneck expand-then-contract design is standard in all GPT models.
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------------------------
# Multi-Head Causal Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with a causal (autoregressive) mask so each
    token can only attend to positions at or before itself.

    Architecture notes
    ------------------
    * Q/K/V projections use bias=False (common in modern GPTs).
    * The causal mask is stored as a non-parameter buffer so it travels with
      the model across device moves without being part of state_dict training.
    * W_o is the final linear that mixes the concatenated head outputs.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads  # dimension per attention head

        # Separate projections keep the code readable and match the book
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection combines all head outputs back to d_out
        self.W_o = nn.Linear(d_out, d_out, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Upper-triangular mask (ones above diagonal) registered as a buffer
        # so it is serialised and moved to GPU automatically with the model
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Project and reshape: (b, num_tokens, d_out) -> (b, num_heads, num_tokens, head_dim)
        Q = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores
        scale = self.head_dim ** -0.5
        attn_scores = Q @ K.transpose(-2, -1) * scale  # (b, num_heads, T, T)

        # Apply causal mask: future positions are set to -inf so softmax zeros them
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens].bool(), float("-inf")
        )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum over values, then merge heads back
        context = (attn_weights @ V)              # (b, num_heads, T, head_dim)
        context = context.transpose(1, 2)         # (b, T, num_heads, head_dim)
        context = context.contiguous().view(b, num_tokens, self.d_out)  # (b, T, d_out)

        return self.W_o(context)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Single transformer layer using pre-norm (LayerNorm before each sublayer).
    Pre-norm trains more stably than post-norm for deep networks.

    Structure per block:
        x = x + Attention(LayerNorm(x))   # residual over attention
        x = x + FeedForward(LayerNorm(x)) # residual over MLP
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in           = cfg["emb_dim"],
            d_out          = cfg["emb_dim"],
            context_length = cfg["context_length"],
            dropout        = cfg["drop_rate"],
            num_heads      = cfg["n_heads"],
            qkv_bias       = cfg["qkv_bias"],
        )
        self.ff    = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention sublayer with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Feed-forward sublayer with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class GPTModel(nn.Module):
    """
    Full GPT-style decoder-only transformer.

    Forward pass input:  idx  — integer token indices, shape (batch, seq_len)
    Forward pass output: logits over vocab, shape (batch, seq_len, vocab_size)
    """

    def __init__(self, cfg):
        super().__init__()

        # Token and position embeddings are summed to give the input to the stack
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack of N identical transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final layer norm applied before the output projection (pre-norm GPT-2 style)
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Linear head projects each token's embedding to a score over the full vocab;
        # weight tying with tok_emb is common but left optional here
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx):
        # idx: (batch, seq_len) integer token ids
        batch, seq_len = idx.shape

        tok_embeds = self.tok_emb(idx)  # (batch, seq_len, emb_dim)

        # Position indices created on the same device as the input so the model
        # works correctly regardless of whether it is on CPU or GPU
        pos_indices  = torch.arange(seq_len, device=idx.device)
        pos_embeds   = self.pos_emb(pos_indices)  # (seq_len, emb_dim) — broadcast over batch

        x = self.drop_emb(tok_embeds + pos_embeds)  # (batch, seq_len, emb_dim)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)  # (batch, seq_len, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# Greedy text generation
# ---------------------------------------------------------------------------

def generate(model, idx, max_new_tokens, context_length):
    """
    Autoregressively extend a sequence of token ids using greedy decoding
    (always pick the highest-probability next token).

    Parameters
    ----------
    model           : GPTModel in eval mode
    idx             : (batch, seq_len) LongTensor of seed token ids
    max_new_tokens  : how many tokens to generate
    context_length  : model's maximum context window; older tokens are cropped

    Returns
    -------
    idx : (batch, seq_len + max_new_tokens) LongTensor
    """

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop the running sequence so it never exceeds the model's context window
            idx_cond = idx[:, -context_length:]

            logits = model(idx_cond)          # (batch, seq_len, vocab_size)
            logits = logits[:, -1, :]         # only the last token's logits: (batch, vocab_size)

            # Greedy: take the token with the highest logit
            idx_next = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
            idx = torch.cat([idx, idx_next], dim=-1)        # (batch, seq_len + 1)

    return idx
