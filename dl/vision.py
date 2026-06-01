"""Vision models: turning images into features, two ways.

To understand a vision-language model you first need to understand how a model
"sees". There are two dominant designs:

    CNN (convolutional network): slide small learned filters over the image.
        Early layers detect edges and textures, deeper layers detect parts and
        objects. Strong built-in assumption that nearby pixels are related
        (locality) and that a feature is useful anywhere (translation
        invariance). The classic backbone (LeNet, ResNet).

    ViT (Vision Transformer): cut the image into fixed patches, treat each patch
        as a token, add position information, and run a transformer encoder
        (dl.transformer). It reuses the exact same machinery as a language model,
        which is precisely why vision and language can later be unified in one
        transformer. This is the backbone most modern VLMs and VLAs use
        (often via SigLIP or DINOv2 style pretraining).

Both produce a feature vector (or a sequence of patch features) that downstream
heads use for classification, detection, or multimodal fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dl.transformer import TransformerEncoder


class SimpleCNN(nn.Module):
    """A small convolutional classifier, the traditional vision backbone.

    Conv -> ReLU -> Pool blocks progressively shrink the spatial size while
    growing the channel count (the "feature" dimension), then a linear head
    classifies. Good intuition for what convolution buys you before ViT.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                    # halve spatial size
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 4 * width, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                            # global average pool
        )
        self.head = nn.Linear(4 * width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).flatten(1)
        return self.head(z)


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and embed each as a token.

    Implemented as a single strided convolution: a Conv2d with kernel and stride
    equal to the patch size turns each patch into one vector. Output is a
    sequence of patch tokens, ready for a transformer, exactly like word tokens.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, dim: int):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                 # (B, dim, H/p, W/p)
        return x.flatten(2).transpose(1, 2)   # (B, num_patches, dim)


class ViT(nn.Module):
    """A Vision Transformer: patches as tokens through a transformer encoder.

    Pipeline: image -> patch tokens -> prepend a learnable [CLS] token -> add
    position embeddings -> transformer encoder -> classify from the [CLS] token.
    The [CLS] token aggregates information from all patches and serves as the
    image's summary vector, which is also what gets aligned with text in a VLM.
    """

    def __init__(self, image_size: int = 32, patch_size: int = 4, in_channels: int = 3,
                 num_classes: int = 10, dim: int = 128, depth: int = 4, num_heads: int = 4):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.encoder = TransformerEncoder(dim, depth, num_heads)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        B = x.size(0)
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos_emb
        x = self.encoder(x)
        cls_out = x[:, 0]                 # the [CLS] summary token
        if return_features:
            return cls_out               # the image embedding used by a VLM
        return self.head(cls_out)
