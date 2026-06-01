"""The end-to-end VLA model, built on real PyTorch framework modules.

This is the stage-2/3 counterpart to the from-scratch ``dl`` library. Where
``dl`` exists to *teach* attention, here we *use* the framework: PyTorch's own
``nn.MultiheadAttention`` and ``nn.TransformerEncoder``. The point of the learning
code was to understand these; the point of an application is to build with the
production primitives, so we do.

Architecture (a small but real VLA):

    image  -> CNN vision encoder -> grid of vision tokens (+ position)
    text   -> embedding + Transformer encoder -> instruction tokens
    proprio-> linear -> one state token
                 \\
                  -> cross-attention: instruction tokens attend over
                     [vision tokens, state token] -> fused tokens
                  -> pool -> MLP action head -> action logits

The cross-attention is the fusion mechanism: the language decides what in the
image and state matters. This is the same idea as a real VLM backbone with an
action head, at a size that trains on a laptop CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Small CNN that turns an image into a grid of tokens.

    Each token is augmented with its explicit normalized (row, col) coordinate.
    Giving the model the literal position of every cell makes the relational task
    ("the red cell is below me, so move down") learnable by a small head, instead
    of forcing it to decode position from an abstract embedding.
    """

    def __init__(self, image_size: int, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, dim, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        grid = image_size // 8                    # three /2 pools
        self.num_tokens = grid * grid
        # Precompute the (row, col) coordinate of each token, normalized to [0, 1].
        rows = torch.arange(grid).repeat_interleave(grid)
        cols = torch.arange(grid).repeat(grid)
        coords = torch.stack([rows, cols], dim=-1).float() / max(grid - 1, 1)
        self.register_buffer("coords", coords.unsqueeze(0))   # (1, N, 2)
        self.proj = nn.Linear(dim + 2, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.num_tokens, dim))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.conv(image)                   # (B, dim, g, g)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, dim)
        coords = self.coords.expand(tokens.size(0), -1, -1)
        tokens = self.proj(torch.cat([tokens, coords], dim=-1))
        return tokens + self.pos


class TextEncoder(nn.Module):
    """Embed instruction tokens and contextualize them with a Transformer."""

    def __init__(self, vocab_size: int, max_len: int, dim: int, num_heads: int = 4,
                 num_layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=4 * dim,
                                           dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, instruction: torch.Tensor) -> torch.Tensor:
        x = self.emb(instruction) + self.pos[:, : instruction.size(1)]
        return self.encoder(x)                    # (B, L, dim)


class VLAPolicy(nn.Module):
    """Vision-Language-Action policy: (image, instruction, proprio) -> action."""

    def __init__(self, image_size: int, vocab_size: int, max_instr_len: int,
                 n_actions: int, dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.vision = VisionEncoder(image_size, dim)
        self.text = TextEncoder(vocab_size, max_instr_len, dim, num_heads)
        self.proprio = nn.Linear(2, dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0,
                                                batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim + 2, dim), nn.ReLU(), nn.Linear(dim, n_actions))

    def forward(self, image: torch.Tensor, instruction: torch.Tensor,
                proprio: torch.Tensor) -> torch.Tensor:
        vis = self.vision(image)                           # (B, Nv, dim)
        txt = self.text(instruction)                       # (B, L, dim)
        state = self.proprio(proprio).unsqueeze(1)         # (B, 1, dim)
        context = torch.cat([vis, state], dim=1)           # (B, Nv+1, dim)
        # Instruction tokens attend over the visual and proprioceptive context.
        fused, _ = self.cross_attn(txt, context, context)  # (B, L, dim)
        fused = self.norm(txt + fused)
        # The target word is the last instruction token ("go to the <target>");
        # use its fused representation, which has attended to the relevant cell.
        target_token = fused[:, -1]                         # (B, dim)
        # Concatenate raw proprio so the head can compare to the attended target.
        return self.head(torch.cat([target_token, proprio], dim=-1))
