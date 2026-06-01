"""Build a from-scratch CLIP and use it for zero-shot classification.

Run:  python curriculum/part16_foundation_models/04_vlm_multimodal/clip_zeroshot.py

This trains MiniCLIP (dl/clip.py): a ViT image tower and a Transformer text tower
aligned by the contrastive loss. After training only on (image, caption) pairs,
it classifies new images with NO classifier head, by comparing each image to the
candidate label captions. That is zero-shot classification, CLIP's signature
capability and the reason it is the default vision encoder for VLMs and VLAs.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from dl.clip import (CLIP_COLORS, CLIP_MAX_LEN, CLIP_VOCAB, MiniCLIP, class_token_ids,
                    synthetic_color_dataset)


def zero_shot_accuracy(model, images, labels):
    logits = model.zero_shot_logits(images, class_token_ids())
    return (logits.argmax(-1) == labels).float().mean().item()


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training MiniCLIP on (image, 'a <color>') pairs (device: {device})\n")

    images, tokens, labels = synthetic_color_dataset(1500, image_size=16, seed=0)
    model = MiniCLIP(image_size=16, patch_size=4, vocab_size=len(CLIP_VOCAB),
                     max_text_len=CLIP_MAX_LEN, dim=96, embed_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    images, tokens, labels = images.to(device), tokens.to(device), labels.to(device)

    test_i, _, test_l = synthetic_color_dataset(200, image_size=16, seed=99)
    test_i, test_l = test_i.to(device), test_l.to(device)
    print(f"zero-shot accuracy before training: {zero_shot_accuracy(model, test_i, test_l):.2f} "
          f"(chance = {1/len(CLIP_COLORS):.2f})\n")

    n = len(images)
    for step in range(1, 301):
        idx = torch.randint(0, n, (64,), device=device)
        loss = model.contrastive_loss(images[idx], tokens[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 75 == 0:
            print(f"step {step:>3}  contrastive_loss {loss.item():.3f}  "
                  f"zero_shot_acc {zero_shot_accuracy(model, test_i, test_l):.2f}")

    print("\nExample zero-shot predictions (compare image to label captions):")
    logits = model.zero_shot_logits(test_i[:5], class_token_ids())
    for k in range(5):
        pred = CLIP_COLORS[int(logits[k].argmax())]
        true = CLIP_COLORS[int(test_l[k])]
        print(f"  image {k}: predicted '{pred}'  (true '{true}')")
    print("\nNo task labels were used to build the classifier: the labels are just "
          "text the model already aligned with images. Swap the toy encoders for a "
          "real ViT and text transformer and this is CLIP.")


if __name__ == "__main__":
    main()
