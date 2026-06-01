"""Build and train MiniSAM: promptable segmentation from scratch.

Run:  python curriculum/part6_perception/build_minisam.py
      (saves build_minisam.png if matplotlib is available)

Trains the from-scratch MiniSAM (dl/segmentation.py) on synthetic shapes to learn
the core SAM mechanism: given an image and a point prompt, segment the object the
point falls on. It then shows promptability: the same image with two different
points produces two different masks. Runs on CPU in about a minute.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dl.segmentation import (MiniSAM, dice_bce_loss, mask_iou, synthetic_shapes)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training MiniSAM (promptable segmentation) on synthetic shapes (device: {device})\n")

    images, points, masks = synthetic_shapes(1500, image_size=32, seed=0)
    images, points, masks = images.to(device), points.to(device), masks.to(device)
    model = MiniSAM(image_size=32, dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    n = len(images)
    for step in range(1, 1501):
        idx = torch.randint(0, n, (32,), device=device)
        logits = model(images[idx], points[idx])
        loss = dice_bce_loss(logits, masks[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0:
            print(f"step {step:>4}  loss {loss.item():.3f}")

    test_i, test_p, test_m = synthetic_shapes(200, image_size=32, seed=7)
    iou = mask_iou(model.predict(test_i.to(device), test_p.to(device)).cpu(), test_m)
    print(f"\nHeld-out segmentation IoU: {iou:.2f}  (1.0 is perfect overlap)")

    _demonstrate_promptability(model, device)
    print("\nSame image, different prompt -> different mask: that is promptable "
          "segmentation, the core idea of SAM. SAM 2 adds memory for video; SAM 3 "
          "adds text/exemplar concept prompts. See segment_anything.md.")


def _demonstrate_promptability(model, device):
    """Render one scene and segment it from two different point prompts."""
    import numpy as np
    img, pts, _ = synthetic_shapes(1, image_size=32, seed=3)
    # Find two distinct foreground points (on two different shapes if possible).
    image = img[0].numpy()
    fg = np.argwhere(np.abs(image - 0.5).sum(0) > 0.2)
    if len(fg) < 2:
        return
    p1 = fg[0]
    p2 = fg[len(fg) // 2]
    prompts = torch.tensor([[p1[1] / 31, p1[0] / 31], [p2[1] / 31, p2[0] / 31]],
                           dtype=torch.float32)
    masks = model.predict(img.repeat(2, 1, 1, 1).to(device), prompts.to(device)).cpu()
    print(f"  prompt A at pixel {(int(p1[1]), int(p1[0]))} -> mask area {int(masks[0].sum())} px")
    print(f"  prompt B at pixel {(int(p2[1]), int(p2[0]))} -> mask area {int(masks[1].sum())} px")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(11, 4))
        ax[0].imshow(image.transpose(1, 2, 0)); ax[0].set_title("scene")
        for a, m, p, name in zip(ax[1:], masks, (p1, p2), ("prompt A", "prompt B")):
            a.imshow(image.transpose(1, 2, 0))
            a.imshow(m.numpy(), alpha=0.5, cmap="Reds")
            a.scatter([p[1]], [p[0]], c="lime", s=80, marker="*")
            a.set_title(name)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_minisam.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"  saved visualization -> {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
