"""Promptable segmentation from scratch: the core mechanism behind SAM.

The Segment Anything Model (SAM) answers a precise question: given an image and a
prompt (a point, a box, a rough mask), produce the mask of the object the prompt
refers to. The architecture is three parts:

    1. an image encoder that runs once and produces a dense feature map,
    2. a prompt encoder that turns the prompt into a few tokens,
    3. a lightweight mask decoder that fuses prompt tokens with image features
       (by attention) and outputs a mask.

This file implements a small but faithful version, ``MiniSAM``, that you can
train in under a minute on synthetic shapes. It captures the essential idea:
the same image plus a different prompt yields a different mask. SAM 2 adds a
memory to extend this to video; SAM 3 adds text/exemplar concept prompts. Those
are explained in the lesson; the mechanism you must understand first is here.

This is teaching code. Real applications use the released SAM/SAM2/SAM3 models;
see the framework guide.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dl.attention import MultiHeadAttention


class ImageEncoder(nn.Module):
    """CNN that produces a dense feature map at 1/4 resolution."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, dim, 3, padding=1), nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)          # (B, dim, H/4, W/4)


class PromptEncoder(nn.Module):
    """Turn a point prompt into tokens, plus a learned output (mask) token.

    The point coordinate is lifted to a higher-dimensional embedding (a small
    positional encoding) so the decoder can use it precisely. The output token is
    a learned query whose decoded representation becomes the mask predictor, just
    as in SAM.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.output_token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, 2) normalized (x, y) in [0, 1].
        point_tok = self.point_mlp(points).unsqueeze(1)        # (B, 1, dim)
        out_tok = self.output_token.expand(points.size(0), -1, -1)
        return torch.cat([out_tok, point_tok], dim=1)          # (B, 2, dim)


class MaskDecoder(nn.Module):
    """Fuse prompt tokens with image features by attention, then predict a mask.

    Each layer lets the tokens attend to the image (cross-attention) and to
    themselves (self-attention). The decoded output token is turned into a
    dynamic per-pixel classifier (a hypernetwork), which is applied to the
    upscaled image features to produce the mask, exactly the structure SAM uses.
    """

    def __init__(self, dim: int, grid: int, image_size: int, num_heads: int = 4,
                 depth: int = 2):
        super().__init__()
        self.grid = grid
        self.image_size = image_size
        self.img_pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        self.self_attn = nn.ModuleList([MultiHeadAttention(dim, num_heads) for _ in range(depth)])
        self.cross_attn = nn.ModuleList([MultiHeadAttention(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.upscale = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1))
        self.hypernet = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, image_feats: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        B, C, h, w = image_feats.shape
        img_tokens = image_feats.flatten(2).transpose(1, 2) + self.img_pos  # (B, N, C)
        for self_a, cross_a in zip(self.self_attn, self.cross_attn):
            tokens = tokens + self_a(tokens)
            tokens = tokens + cross_a(tokens, context=img_tokens)
        tokens = self.norm(tokens)
        mask_vector = self.hypernet(tokens[:, 0])               # (B, C) dynamic classifier

        # Upscale image features to full resolution and apply the dynamic classifier.
        feat = F.interpolate(image_feats, size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)
        feat = self.upscale(feat)                               # (B, C, H, W)
        mask_logits = torch.einsum("bchw,bc->bhw", feat, mask_vector)
        return mask_logits


class MiniSAM(nn.Module):
    """A complete promptable segmentation model: image + point -> object mask."""

    def __init__(self, image_size: int = 32, dim: int = 64):
        super().__init__()
        self.image_size = image_size
        self.grid = image_size // 4
        self.image_encoder = ImageEncoder(dim)
        self.prompt_encoder = PromptEncoder(dim)
        self.mask_decoder = MaskDecoder(dim, self.grid, image_size)

    def forward(self, image: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Return per-pixel mask logits ``(B, H, W)`` for the prompted object."""
        feats = self.image_encoder(image)
        tokens = self.prompt_encoder(points)
        return self.mask_decoder(feats, tokens)

    @torch.no_grad()
    def predict(self, image: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(self(image, points)) > 0.5).float()


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Segmentation loss combining pixel BCE and a soft Dice term.

    BCE supervises every pixel; Dice directly optimizes overlap and handles the
    foreground/background imbalance typical of masks.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = (1, 2)
    inter = (prob * target).sum(dims)
    dice = 1 - (2 * inter + 1) / (prob.sum(dims) + target.sum(dims) + 1)
    return bce + dice.mean()


def mask_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean intersection-over-union between binary predicted and target masks."""
    inter = (pred * target).sum((1, 2))
    union = ((pred + target) > 0).float().sum((1, 2))
    return float((inter / union.clamp(min=1)).mean())


def synthetic_shapes(n: int, image_size: int = 32, seed: int = 0):
    """Teaching dataset: images with a few colored shapes, a point, and the mask
    of the shape that the point falls inside.

    The target shape is painted last so it is fully visible, and the prompt point
    is sampled inside it. This is exactly the promptable-segmentation task: the
    same image with a different point would select a different shape.
    Returns torch tensors: images (n,3,H,H), points (n,2) in [0,1], masks (n,H,H).
    """
    rng = np.random.default_rng(seed)
    H = image_size
    images = np.full((n, 3, H, H), 0.5, dtype=np.float32)
    points = np.zeros((n, 2), dtype=np.float32)
    masks = np.zeros((n, H, H), dtype=np.float32)
    yy, xx = np.ogrid[:H, :H]
    for i in range(n):
        shapes = []
        for _ in range(int(rng.integers(2, 4))):
            color = rng.uniform(0.2, 1.0, 3).astype(np.float32)
            if rng.integers(2) == 0:                          # circle
                r = int(rng.integers(H // 8, H // 4))
                cy, cx = rng.integers(r, H - r, size=2)
                m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
            else:                                              # rectangle
                hh, ww = rng.integers(H // 6, H // 3, size=2)
                cy, cx = int(rng.integers(0, H - hh)), int(rng.integers(0, H - ww))
                m = np.zeros((H, H), dtype=bool)
                m[cy:cy + hh, cx:cx + ww] = True
            shapes.append((m, color))
        target_idx = int(rng.integers(len(shapes)))
        # Paint non-target shapes first, then the target on top (fully visible).
        order = [j for j in range(len(shapes)) if j != target_idx] + [target_idx]
        for j in order:
            m, color = shapes[j]
            images[i, :, m] = color          # broadcasts (3,) over the masked pixels
        tm = shapes[target_idx][0]
        ys, xs = np.where(tm)
        k = int(rng.integers(len(xs)))
        points[i] = [xs[k] / (H - 1), ys[k] / (H - 1)]
        masks[i] = tm.astype(np.float32)
    return (torch.from_numpy(images), torch.from_numpy(points), torch.from_numpy(masks))
