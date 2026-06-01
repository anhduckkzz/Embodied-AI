"""Attention: the single mechanism behind every modern model.

Attention answers one question: "for each position in a sequence, which other
positions should I look at, and how much?" It then builds a weighted summary of
those positions. This is what lets a model relate distant words, link an
instruction to a region of an image, or connect a past observation to the
current action.

The vocabulary, made concrete:
    Query  (Q): what the current position is looking for.
    Key    (K): what each position offers, used for matching against queries.
    Value  (V): the actual content each position will contribute if attended to.

We compute how well each query matches each key, turn those scores into weights
with softmax, and use the weights to average the values. That is all attention
is. Everything else (multiple heads, masks, stacking into transformers) is
structure around this core operation.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The heart of attention.

    Shapes: q, k, v are ``(batch, heads, seq, head_dim)``. Returns the attended
    output ``(batch, heads, seq, head_dim)`` and the attention weights
    ``(batch, heads, seq, seq)`` (useful for visualizing what attends to what).

    Steps:
      1. scores = Q . K^T            (how well each query matches each key)
      2. scale by 1/sqrt(head_dim)   (keeps scores from growing with dimension)
      3. optional mask               (e.g. hide future positions in a language model)
      4. softmax over keys           (turn scores into a probability distribution)
      5. weighted sum of values      (the output: a mix of the relevant content)
    """
    head_dim = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    if mask is not None:
        # Where mask == 0, set the score to -inf so softmax gives it ~0 weight.
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights


def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """Lower-triangular mask so position t can only attend to positions <= t.

    This is what makes a language model autoregressive: when predicting the next
    token it must not peek at future tokens. Returns a ``(seq, seq)`` matrix of
    1s (allowed) and 0s (blocked).
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device))


class MultiHeadAttention(nn.Module):
    """Run several attention operations in parallel, then combine them.

    Why multiple "heads"? A single attention pattern can only capture one kind of
    relationship. With ``num_heads`` heads, each can specialize (one tracks
    syntax, another tracks subject-object links, and so on). We split the model
    dimension across heads, attend independently, concatenate, and project back.

    Set ``causal=True`` for decoder/language-model self-attention.
    """

    def __init__(self, dim: int, num_heads: int, causal: bool = False, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        # One linear layer produces Q, K, V for efficiency, then we split.
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x`` is ``(batch, seq, dim)``.

        If ``context`` is given, this is **cross-attention**: queries come from
        ``x`` while keys/values come from ``context`` (for example, text tokens
        attending to image tokens inside a VLM). If ``context`` is None it is
        **self-attention**: a sequence attending to itself.
        """
        B, T, _ = x.shape
        if context is None:
            context = x
        S = context.size(1)

        # Project. For cross-attention, Q from x, K/V from context.
        q = self.qkv(x)[:, :, : self.dim]
        kv = self.qkv(context)[:, :, self.dim:]
        k, v = kv[:, :, : self.dim], kv[:, :, self.dim:]

        # Reshape (batch, seq, dim) -> (batch, heads, seq, head_dim).
        def split_heads(t, seq):
            return t.view(B, seq, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(q, T), split_heads(k, S), split_heads(v, S)

        if mask is None and self.causal:
            mask = causal_mask(T, x.device)

        out, _ = scaled_dot_product_attention(q, k, v, mask)
        # Recombine heads: (batch, heads, seq, head_dim) -> (batch, seq, dim).
        out = out.transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.dropout(self.out_proj(out))
