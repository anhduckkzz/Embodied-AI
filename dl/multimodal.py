"""Multimodal models: how vision and language are joined.

This is the conceptual bridge from single-modality models to VLMs and then VLAs.
Two complementary mechanisms, both implemented here:

  1. Contrastive alignment (CLIP): train an image encoder and a text encoder so
     that matching image-text pairs land close together in a shared vector space
     and non-matching pairs land far apart. This teaches the model to "ground"
     language in vision. The result is the kind of vision encoder (CLIP, SigLIP)
     used as the eyes of most VLMs.

  2. Cross-attention fusion: let language tokens attend to image tokens (or
     vice versa) inside a transformer, so the two streams genuinely interact
     rather than just sitting side by side. This is the fusion that turns a
     vision encoder plus a language model into a true VLM, and a VLM plus an
     action head into a VLA.

Understanding these two ideas is exactly the "how Vision-Language is unified"
step. A VLA then adds an action head on top of a VLM backbone (see Part 10/16).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dl.attention import MultiHeadAttention


def clip_contrastive_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor,
                          temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE loss used by CLIP.

    Given a batch of paired (image, text) embeddings, the matched pairs are the
    diagonal of the similarity matrix and should score highest. We L2-normalize,
    compute all pairwise cosine similarities scaled by 1/temperature, and apply
    cross-entropy in both directions (image->text and text->image), with the
    correct match being the diagonal index.
    """
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    logits = image_embeds @ text_embeds.t() / temperature   # (B, B) similarities
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2


class DualEncoder(nn.Module):
    """A minimal CLIP-style model: two encoders projecting to a shared space.

    The encoders here are simple MLPs over pre-extracted features so the example
    stays small and fast; in practice the image encoder is a ViT (dl.vision) and
    the text encoder is a transformer (dl.transformer). The projection heads and
    the contrastive loss are exactly as in real CLIP.
    """

    def __init__(self, image_feat_dim: int, text_feat_dim: int, embed_dim: int = 64):
        super().__init__()
        self.image_proj = nn.Sequential(
            nn.Linear(image_feat_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.text_proj = nn.Sequential(
            nn.Linear(text_feat_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, image_feats: torch.Tensor, text_feats: torch.Tensor):
        return self.image_proj(image_feats), self.text_proj(text_feats)


class CrossAttentionFusion(nn.Module):
    """Fuse two modalities by letting one attend to the other.

    Queries come from the ``x`` stream (for example language tokens); keys and
    values come from the ``context`` stream (for example image patch tokens). The
    language tokens are updated with the image information most relevant to them.
    This is the fusion operation at the core of many VLMs (and VLAs, where the
    fused tokens then drive an action head).
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads, causal=False)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_attn(self.norm_x(x), context=self.norm_ctx(context))
        x = x + self.ff(self.norm_ff(x))
        return x
