"""Transformers: stacking attention into the backbone of modern AI.

A transformer block is attention plus a small per-position feed-forward network,
wrapped with residual connections and layer normalization. Stack many blocks and
you get the architecture behind GPT (language), ViT (vision), and the backbones
of VLMs and VLAs. The same block, used three ways:

    causal self-attention   -> a language model (GPT) predicting the next token
    bidirectional attention -> an encoder (BERT, the ViT body) building features
    cross-attention         -> fusing one modality into another (VLM, decoder)

This file builds the block, a positional encoding, and a small but complete GPT
you can actually train on a few sentences.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dl.attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    """Inject order into the sequence.

    Attention is permutation invariant: by itself it does not know token 1 comes
    before token 2. We add a position signal so the model can use order. This is
    the classic fixed sinusoidal encoding; many modern models learn it instead
    (we use learned positions in the GPT below for simplicity).
    """

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Module):
    """Per-position MLP: expand, apply a nonlinearity, project back.

    Attention mixes information across positions; this network processes each
    position independently and gives the block its capacity to transform
    features. The expansion factor is usually 4x.
    """

    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_mult * dim),
            nn.GELU(),
            nn.Linear(hidden_mult * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One transformer layer using the modern pre-norm design.

    pre-norm means we normalize before each sub-layer and add a residual:
        x = x + attention(norm(x))
        x = x + feedforward(norm(x))
    The residual connections let gradients flow through deep stacks; layer norm
    keeps activations well scaled. Set ``causal=True`` for a language model.
    """

    def __init__(self, dim: int, num_heads: int, causal: bool = False, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, causal=causal, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """A stack of bidirectional transformer blocks (no causal mask).

    This is the body of BERT and of a Vision Transformer: every position can see
    every other position, producing context-aware features for the whole input.
    """

    def __init__(self, dim: int, depth: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, causal=False, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.norm(x)


class GPT(nn.Module):
    """A small decoder-only language model (GPT style), complete and trainable.

    Pipeline: token ids -> token embeddings + position embeddings -> a stack of
    causal transformer blocks -> a linear head producing a probability over the
    vocabulary for the next token. This is, in miniature, exactly the
    architecture of the large language models that back VLMs and VLAs.
    """

    def __init__(self, vocab_size: int, dim: int = 128, depth: int = 4,
                 num_heads: int = 4, max_len: int = 256, dropout: float = 0.0):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, causal=True, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """``idx`` is ``(batch, seq)`` of token ids. If ``targets`` is given,
        also returns the next-token cross-entropy loss used for training."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """Autoregressive sampling: repeatedly predict and append the next token."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
