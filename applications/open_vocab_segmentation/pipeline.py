"""Open-vocabulary segmentation: segment objects named by free text.

This is an application-layer component built on the real frameworks (SAM and CLIP,
or SAM 3 natively). It does not use the from-scratch teaching code in dl/; that
code is for understanding the mechanism, this code is for building the system.

Two paths are provided:

  1. SAM + CLIP (works with the original SAM or SAM 2). SAM proposes candidate
     masks for everything in the image; CLIP labels each masked region by
     comparing it to the text queries. The combination yields open-vocabulary
     segmentation even though neither model alone is both localized and
     language-aware.

  2. SAM 3 (native). SAM 3 accepts a text prompt directly and returns all
     instances of the concept, so the CLIP step is unnecessary.

The frameworks need downloaded checkpoints and a GPU, so this is reference code
for your own machine. It degrades gracefully and explains what to install if the
models are absent.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def segment_with_sam3(image: np.ndarray, text_queries: List[str],
                      checkpoint: str = "sam3_base"):
    """Path 2: SAM 3 native promptable concept segmentation (text prompt).

    Returns a list of (query, mask) for every detected instance. Requires the
    `sam3` package and a checkpoint. The exact API may change; check the repo.
    """
    try:
        from sam3 import SAM3
    except ImportError:
        print("SAM 3 not installed. `pip install sam3` and download a checkpoint, "
              "or use the SAM + CLIP path. See sam_framework_guide.md.")
        return None
    model = SAM3.from_pretrained(checkpoint)
    results = []
    for query in text_queries:
        out = model.predict(image, text_prompt=query)
        for mask in out.masks:                 # one mask per matching instance
            results.append((query, mask))
    return results


def segment_with_sam_and_clip(image: np.ndarray, text_queries: List[str],
                              sam_checkpoint: str = "sam_vit_b.pth",
                              clip_name: str = "openai/clip-vit-base-patch32",
                              score_threshold: float = 0.25,
                              device: str = "cuda"):
    """Path 1: SAM proposes masks, CLIP labels each region by text.

    For each SAM mask we crop the masked region, embed it with CLIP, and match it
    to the text queries. Masks whose best query score exceeds the threshold are
    returned as (query, score, mask).
    """
    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        print("This path needs `segment-anything` and `transformers` plus model "
              "checkpoints. See sam_framework_guide.md and the CLIP lesson.")
        return None

    # 1. SAM proposes masks for everything in the image.
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    proposals = mask_generator.generate(image)         # list of dicts with 'segmentation'

    # 2. CLIP scores each masked crop against the text queries.
    clip = CLIPModel.from_pretrained(clip_name).to(device)
    proc = CLIPProcessor.from_pretrained(clip_name)
    text_inputs = proc(text=[f"a photo of a {q}" for q in text_queries],
                       return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_feats = clip.get_text_features(**text_inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    results = []
    for p in proposals:
        mask = p["segmentation"]
        crop = _masked_crop(image, mask)
        if crop is None:
            continue
        with torch.no_grad():
            img_inputs = proc(images=crop, return_tensors="pt").to(device)
            img_feat = clip.get_image_features(**img_inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ text_feats.t()).squeeze(0)   # (num_queries,)
        best = int(sims.argmax())
        score = float(sims[best])
        if score >= score_threshold:
            results.append((text_queries[best], score, mask))
    return results


def _masked_crop(image: np.ndarray, mask: np.ndarray, pad: int = 4):
    """Crop the bounding box of a mask, zeroing out non-mask pixels."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, image.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, image.shape[1])
    crop = image[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1]] = 0
    return crop


def open_vocabulary_segment(image: np.ndarray, text_queries: List[str],
                            prefer: str = "auto", **kwargs):
    """Segment objects named by free text. Tries SAM 3 first if requested or
    available, otherwise falls back to SAM + CLIP."""
    if prefer in ("auto", "sam3"):
        out = segment_with_sam3(image, text_queries, **{k: v for k, v in kwargs.items()
                                                        if k == "checkpoint"})
        if out is not None:
            return out
        if prefer == "sam3":
            return None
    return segment_with_sam_and_clip(image, text_queries,
                                     **{k: v for k, v in kwargs.items()
                                        if k in ("sam_checkpoint", "clip_name",
                                                 "score_threshold", "device")})


def main():
    print("Open-vocabulary segmentation (SAM + CLIP, or SAM 3).\n")
    print("This application runs on a machine with the SAM/CLIP (or SAM 3) models")
    print("installed plus their checkpoints and a GPU. See README.md and the")
    print("framework guide in curriculum/part6_perception/sam_framework_guide.md.\n")
    print("Usage sketch:")
    print("  import numpy as np, cv2")
    print("  image = cv2.cvtColor(cv2.imread('scene.jpg'), cv2.COLOR_BGR2RGB)")
    print("  from applications.open_vocab_segmentation.pipeline import open_vocabulary_segment")
    print("  results = open_vocabulary_segment(image, ['mug', 'keyboard', 'person'])")


if __name__ == "__main__":
    main()
