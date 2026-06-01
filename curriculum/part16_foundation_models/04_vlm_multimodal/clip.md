# CLIP: contrastive language-image pretraining

Three perspectives on CLIP, the model that taught vision to understand language
and became the default eyes of VLMs and VLAs. This complements
[the VLM module README](README.md) and the build-it demo
[`build_clip.py`](build_clip.py) with a focused, end-to-end treatment.

Code: from-scratch mechanism in [`dl/clip.py`](../../../dl/clip.py) and
[`dl/multimodal.py`](../../../dl/multimodal.py); zero-shot demo
[`clip_zeroshot.py`](clip_zeroshot.py).

## 1. Mechanism: how CLIP works (build it)

CLIP has two encoders and one training objective.

- Image tower: a vision encoder (a ViT) maps an image to a vector.
- Text tower: a text transformer maps a caption to a vector.
- Shared space: both vectors are projected to the same dimension and
  L2-normalized.
- Contrastive objective (InfoNCE): for a batch of N matched (image, caption)
  pairs, compute the N by N matrix of cosine similarities. The correct pairs are
  the diagonal. Cross-entropy in both directions (image to text and text to
  image) pulls matched pairs together and pushes mismatched pairs apart. A learned
  temperature (the logit scale) sharpens the distribution.

That is the whole idea: do not predict a fixed label, learn a shared space where
an image and its description coincide. The complete mechanism is in
[`dl/clip.py`](../../../dl/clip.py) (`MiniCLIP`), with the loss in
[`dl/multimodal.py`](../../../dl/multimodal.py) (`clip_contrastive_loss`).

### Zero-shot classification (the payoff)

Once the space is learned you can classify with no task-specific training. Embed
the image, embed candidate label captions ("a photo of a cat", "a photo of a
dog"), and pick the nearest caption. The runnable
[`clip_zeroshot.py`](clip_zeroshot.py) trains MiniCLIP only on (image, caption)
pairs and then classifies fresh images by comparison to label captions, with no
classifier head.

```bash
python curriculum/part16_foundation_models/04_vlm_multimodal/clip_zeroshot.py
```

## 2. Framework: running real CLIP

You will not pretrain CLIP yourself (it took hundreds of millions of image-text
pairs). Use a pretrained one. Two common options, for your own machine (the
sandbox here has no model weights):

Hugging Face transformers:

```python
# pip install transformers pillow torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("scene.jpg")
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
inputs = proc(text=labels, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(**inputs).logits_per_image          # image-to-text scores
probs = logits.softmax(dim=1)                           # zero-shot class probabilities
```

open_clip (more models, including SigLIP-style and larger trainings):

```python
# pip install open_clip_torch
import open_clip, torch
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
# encode_image / encode_text, then cosine similarity, as above.
```

On the 4060 these base models run comfortably for inference. To adapt CLIP to a
domain, fine-tune with LoRA (Part 16.6) rather than full training.

## 3. Real system: where CLIP is used

CLIP is rarely the end product; it is the grounding component inside larger
systems.

- Vision encoder for VLMs and VLAs. CLIP and its successor SigLIP are the
  standard image encoders whose features already understand language, which is
  why a VLM can connect words to image regions (Module 16.4) and a VLA can follow
  language instructions (Part 10).
- Open-vocabulary perception. CLIP turns any text into a classifier, enabling
  open-vocabulary detection and segmentation. Paired with SAM (Part 6) it labels
  masks by free text, see
  [`applications/open_vocab_segmentation/`](../../../applications/open_vocab_segmentation/).
  This is also the language half that SAM 3 brings inside one model.
- Retrieval and data curation. Find images by text (and vice versa); filter and
  label massive datasets, including robot datasets.
- Reward and success signals. CLIP similarity between an image and a goal
  description can score whether a task looks done, a useful signal for RL and for
  evaluating policies.

## How CLIP and SAM relate

CLIP aligns whole images with text but does not localize. SAM localizes (masks)
but, before SAM 3, did not understand text. Combine them and you get
open-vocabulary segmentation: SAM proposes masks, CLIP labels each masked region
by comparing it to candidate texts. SAM 3 folds this language grounding into the
segmentation model directly via promptable concept segmentation.

## Project

1. Run [`clip_zeroshot.py`](clip_zeroshot.py); watch zero-shot accuracy rise from
   chance as the contrastive loss falls. Add a new color and a held-out test.
2. Replace the toy towers in `MiniCLIP` with the real `dl.vision.ViT` and a
   longer text vocabulary, and verify zero-shot still works.
3. On your machine, load a pretrained CLIP and reproduce a zero-shot
   classification on a few real photos with your own label list.

## Check your understanding

1. What does CLIP's contrastive objective pull together and push apart?
2. Why can a trained CLIP classify without any task-specific training data?
3. What is the learned temperature (logit scale) for?
4. Why are CLIP-style features a good vision encoder for a VLM or VLA?
5. How do CLIP and SAM combine for open-vocabulary segmentation, and what does
   SAM 3 change?

## Go deeper

- Radford et al. (2021), Learning Transferable Visual Models (CLIP).
- Zhai et al. (2023), SigLIP. Cherti et al. (2023), open_clip / reproducible scaling.
- Hugging Face transformers CLIP docs; the open_clip repository.
