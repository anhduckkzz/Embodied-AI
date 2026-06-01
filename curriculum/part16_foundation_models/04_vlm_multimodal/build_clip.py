"""Build and train a small CLIP-style dual encoder from scratch.

Run:  python curriculum/part16_foundation_models/04_vlm_multimodal/build_clip.py

Demonstrates contrastive image-text alignment, the mechanism that grounds
language in vision and produces the encoders used by VLMs and VLAs. We use
pre-extracted toy features so it runs instantly on a CPU; the loss and the
training loop are exactly those of real CLIP. We measure image-to-text retrieval
accuracy before and after training.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from dl.multimodal import DualEncoder, clip_contrastive_loss


def retrieval_accuracy(model, img_feats, txt_feats):
    """Top-1 image-to-text retrieval: does each image's nearest text match it?"""
    with torch.no_grad():
        ie, te = model(img_feats, txt_feats)
        sim = F.normalize(ie, dim=-1) @ F.normalize(te, dim=-1).t()
        match = (sim.argmax(dim=1) == torch.arange(sim.size(0))).float().mean()
    return match.item()


def main():
    torch.manual_seed(0)
    n, idim, tdim = 64, 48, 32
    # Build paired features with a real underlying correspondence to discover:
    # each caption feature is a fixed linear map of its image feature, plus noise.
    img_feats = torch.randn(n, idim)
    W = torch.randn(idim, tdim)
    txt_feats = img_feats @ W + 0.1 * torch.randn(n, tdim)

    model = DualEncoder(idim, tdim, embed_dim=48)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    print(f"Contrastive image-text alignment on {n} pairs")
    print(f"retrieval accuracy before training: {retrieval_accuracy(model, img_feats, txt_feats):.2f} "
          f"(chance = {1/n:.2f})\n")

    for step in range(1, 401):
        ie, te = model(img_feats, txt_feats)
        loss = clip_contrastive_loss(ie, te)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"step {step:>3}  loss {loss.item():.3f}  "
                  f"retrieval_acc {retrieval_accuracy(model, img_feats, txt_feats):.2f}")

    print("\nMatching image-text pairs are now closest in the shared space. This "
          "is how a VLM grounds language in vision, and where a VLA's eyes come from.")


if __name__ == "__main__":
    main()
