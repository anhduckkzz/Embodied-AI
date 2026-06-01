"""CLIP from scratch: contrastive language-image pretraining, end to end.

This builds on ``dl.multimodal`` (which has the contrastive loss and the fusion
block) and wires real small encoders into a complete CLIP that can do zero-shot
classification. It exists to teach the mechanism. Application and framework code
should use a real CLIP (open_clip, Hugging Face transformers); the lesson and the
guides cover those.

The idea in one sentence: train an image encoder and a text encoder so that an
image and its caption land at the same point in a shared space. Once trained you
can classify an image with no labelled training for the task: embed the image,
embed candidate label texts ("a photo of a cat", "a photo of a dog"), and pick
the nearest. That is zero-shot classification, and it is why CLIP became the
default vision backbone for VLMs and VLAs.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dl.multimodal import clip_contrastive_loss
from dl.transformer import TransformerEncoder
from dl.vision import ViT

# A tiny vocabulary for the teaching demos: captions are "a <color>".
CLIP_COLORS = ["red", "green", "blue", "yellow"]
_RGB = {"red": (1.0, 0.1, 0.1), "green": (0.1, 0.8, 0.2),
        "blue": (0.2, 0.3, 1.0), "yellow": (1.0, 0.85, 0.1)}
CLIP_VOCAB = ["<pad>", "a"] + CLIP_COLORS
CLIP_STOI = {w: i for i, w in enumerate(CLIP_VOCAB)}
CLIP_MAX_LEN = 2


def caption_ids(color: str) -> List[int]:
    return [CLIP_STOI["a"], CLIP_STOI[color]]


def class_token_ids() -> torch.Tensor:
    """Token ids for every class caption, shape (num_colors, CLIP_MAX_LEN)."""
    return torch.tensor([caption_ids(c) for c in CLIP_COLORS], dtype=torch.long)


def synthetic_color_dataset(n: int, image_size: int = 16, seed: int = 0):
    """Images each showing one colored shape on gray, plus the color label and
    the caption token ids. Used to train and zero-shot-test MiniCLIP."""
    rng = np.random.default_rng(seed)
    H = image_size
    images = np.full((n, 3, H, H), 0.5, dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    tokens = np.zeros((n, CLIP_MAX_LEN), dtype=np.int64)
    yy, xx = np.ogrid[:H, :H]
    for i in range(n):
        ci = int(rng.integers(len(CLIP_COLORS)))
        color = np.array(_RGB[CLIP_COLORS[ci]], dtype=np.float32)
        if rng.integers(2) == 0:                              # circle
            r = int(rng.integers(H // 6, H // 3))
            cy, cx = rng.integers(r, H - r, size=2)
            m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        else:                                                  # rectangle
            hh, ww = rng.integers(H // 4, H // 2, size=2)
            cy, cx = int(rng.integers(0, H - hh)), int(rng.integers(0, H - ww))
            m = np.zeros((H, H), dtype=bool)
            m[cy:cy + hh, cx:cx + ww] = True
        images[i, :, m] = color              # broadcasts (3,) over the masked pixels
        labels[i] = ci
        tokens[i] = caption_ids(CLIP_COLORS[ci])
    return (torch.from_numpy(images), torch.from_numpy(tokens), torch.from_numpy(labels))


class TextTower(nn.Module):
    """Token ids -> a single text embedding, via a small Transformer encoder."""

    def __init__(self, vocab_size: int, max_len: int, dim: int, depth: int = 2,
                 num_heads: int = 4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.encoder = TransformerEncoder(dim, depth, num_heads)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(token_ids) + self.pos[:, : token_ids.size(1)]
        x = self.encoder(x)
        return x.mean(dim=1)            # mean-pool the token features


class MiniCLIP(nn.Module):
    """A complete, tiny CLIP: ViT image tower + Transformer text tower.

    Both towers project into a shared ``embed_dim`` space. ``logit_scale`` is the
    learned temperature from the CLIP paper. Train with ``contrastive_loss`` on a
    batch of matched (image, text) pairs; classify zero-shot with
    ``zero_shot_logits``.
    """

    def __init__(self, image_size: int, patch_size: int, vocab_size: int,
                 max_text_len: int, dim: int = 128, embed_dim: int = 64):
        super().__init__()
        self.image_tower = ViT(image_size, patch_size, num_classes=1, dim=dim,
                               depth=3, num_heads=4)
        self.text_tower = TextTower(vocab_size, max_text_len, dim)
        self.image_proj = nn.Linear(dim, embed_dim)
        self.text_proj = nn.Linear(dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.659))   # log(1 / 0.07)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.image_tower(image, return_features=True)
        return F.normalize(self.image_proj(feats), dim=-1)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        feats = self.text_tower(token_ids)
        return F.normalize(self.text_proj(feats), dim=-1)

    def contrastive_loss(self, image: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        img_e = self.encode_image(image)
        txt_e = self.encode_text(token_ids)
        # Reuse the InfoNCE loss; temperature comes from the learned logit scale.
        return clip_contrastive_loss(img_e, txt_e,
                                     temperature=1.0 / self.logit_scale.exp().clamp(max=100))

    @torch.no_grad()
    def zero_shot_logits(self, image: torch.Tensor, class_token_ids: torch.Tensor) -> torch.Tensor:
        """Score one image batch against a set of candidate label texts.

        ``class_token_ids`` is ``(num_classes, max_text_len)``. Returns
        ``(batch, num_classes)`` similarity logits; argmax is the prediction.
        """
        img_e = self.encode_image(image)                 # (B, E)
        txt_e = self.encode_text(class_token_ids)        # (C, E)
        return self.logit_scale.exp() * img_e @ txt_e.t()
