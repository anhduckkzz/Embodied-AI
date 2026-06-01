"""Build and train a small Vision Transformer from scratch.

Run:  python curriculum/part16_foundation_models/03_vision_models/build_vit.py

Trains the ViT from dl/vision.py to classify synthetic images as having either
horizontal or vertical stripes. The task is simple on purpose so it trains in
seconds on a CPU, while exercising the full pipeline: patchify, embed, attend,
read the [CLS] token, classify.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from dl.vision import ViT

IMG = 16


def make_batch(n, device):
    """Half the batch has horizontal stripes (label 0), half vertical (label 1)."""
    x = torch.rand(n, 3, IMG, IMG) * 0.1
    rows = torch.arange(IMG).view(1, 1, IMG, 1)
    cols = torch.arange(IMG).view(1, 1, 1, IMG)
    horiz = ((rows % 4) < 2).float().expand(n // 2, 3, IMG, IMG)
    vert = ((cols % 4) < 2).float().expand(n - n // 2, 3, IMG, IMG)
    x = torch.cat([x[: n // 2] + horiz, x[n // 2:] + vert])
    y = torch.cat([torch.zeros(n // 2), torch.ones(n - n // 2)]).long()
    return x.to(device), y.to(device)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViT(image_size=IMG, patch_size=4, num_classes=2, dim=64, depth=3,
                num_heads=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    n_patches = model.patch_embed.num_patches
    print(f"ViT: {IMG}x{IMG} image -> {n_patches} patch tokens + 1 CLS | "
          f"params: {sum(p.numel() for p in model.parameters()):,} (device {device})\n")

    for step in range(1, 301):
        x, y = make_batch(64, device)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            xt, yt = make_batch(200, device)
            acc = (model(xt).argmax(-1) == yt).float().mean().item()
            print(f"step {step:>3}  loss {loss.item():.3f}  test_acc {acc:.2f}")

    print("\nThe ViT learned to read patch patterns through self-attention. The "
          "[CLS] feature it produces is exactly what a VLM aligns with text next.")


if __name__ == "__main__":
    main()
