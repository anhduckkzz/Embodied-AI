# Module 16.4 - Vision-Language Models: unifying the two modalities

Code: [`dl/multimodal.py`](../../../dl/multimodal.py),
[`dl/clip.py`](../../../dl/clip.py). Demos: [`build_clip.py`](build_clip.py)
(contrastive alignment) and [`clip_zeroshot.py`](clip_zeroshot.py) (zero-shot
classification). A focused three-perspective treatment of CLIP (mechanism,
framework, real system) is in [`clip.md`](clip.md).

This module is the answer to your question: to understand a VLA you must
understand a VLM, and to understand a VLM you must understand how a vision model
(Module 16.3) and a language model (Module 16.2) are joined. There are two core
mechanisms, and a VLM uses one or both.

## 1. What a VLM is

A vision-language model takes images and text together and reasons over both:
captioning an image, answering a question about it, or locating an object named
in text. It needs the image features (Module 16.3) and the text features (Module
16.2) to live in a shared space and to interact. The two mechanisms below provide
exactly that.

## 2. Mechanism 1: contrastive alignment (CLIP)

Goal: make matching image-text pairs close in a shared embedding space, and
non-matching pairs far apart. Recipe:

1. Encode each image to a vector (a ViT) and each caption to a vector (a text
   transformer).
2. Project both into a shared embedding space and L2-normalize.
3. For a batch of N pairs, compute the N x N matrix of cosine similarities. The
   correct matches are the diagonal.
4. Apply cross-entropy so each image picks its own caption and each caption picks
   its own image (the symmetric InfoNCE loss).

See [`clip_contrastive_loss`](../../../dl/multimodal.py). The result is a vision
encoder whose features are already aligned with language. This is why CLIP and
SigLIP are the standard eyes of modern VLMs and VLAs: the image features are
already grounded in words.

Contrastive alignment gives you retrieval and zero-shot classification (compare
an image embedding to embeddings of candidate label texts), but the two streams
do not deeply interact. For richer reasoning you add mechanism 2.

## 3. Mechanism 2: cross-attention fusion

Let one modality attend to the other inside a transformer. Concretely, language
tokens form the queries and image patch tokens form the keys and values, so each
text token is updated with the image content most relevant to it (or the reverse).
This makes the modalities genuinely interact rather than just sit side by side.
See [`CrossAttentionFusion`](../../../dl/multimodal.py); it is the cross-attention
of Module 16.1 with Q from one stream and K, V from the other.

## 4. How real VLMs are built (two common patterns)

- Dual-encoder, late fusion (CLIP family): separate encoders, compare embeddings.
  Great for retrieval and grounding, lightweight.
- Backbone fusion (LLaVA, Flamingo, Qwen-VL, PaliGemma): take a pretrained vision
  encoder, project its patch features into the token space of a pretrained large
  language model, and feed them in as if they were extra tokens. The LLM then
  attends over image tokens and text tokens together (self-attention over the
  combined sequence, or cross-attention layers as in Flamingo). This reuses a
  powerful language model and is how most instruction-following VLMs work today.

The common thread: project image features into a shared token space, then let
attention mix vision and language.

## 5. From VLM to VLA (the payoff)

A vision-language-action model is a VLM with an action output:

```
images + instruction  ->  VLM backbone (vision encoder + LLM + fusion)  ->  action head  ->  robot actions
```

The action head is either action tokenization (discretize actions into tokens the
LLM predicts, as in RT-2 and OpenVLA) or a continuous diffusion/flow head (Module
16.5, as in Octo and Pi0). Everything before the action head is exactly the VLM
you built understanding of in this module. That is why the dependency chain is
real: transformer -> vision and language models -> VLM -> VLA.

## 6. Build it yourself

[`build_clip.py`](build_clip.py) trains the dual encoder from
[`dl/multimodal.py`](../../../dl/multimodal.py) with the contrastive loss on
paired features, then measures image-to-text retrieval accuracy. It shows
alignment emerging: after training, each image's nearest text is its true caption.

```bash
python curriculum/part16_foundation_models/04_vlm_multimodal/build_clip.py
```

## Project
1. Run `build_clip.py`. Watch retrieval accuracy climb from chance to near
   perfect as the contrastive loss falls.
2. Replace the toy feature inputs with a real `ViT` image encoder (Module 16.3)
   and the `GPT`/encoder text features (Module 16.2). You will have built a small
   end-to-end CLIP.
3. Use [`CrossAttentionFusion`](../../../dl/multimodal.py) to let text tokens read
   image tokens, and inspect the cross-attention weights.

## Check your understanding
1. What does contrastive alignment optimize, and what is the correct match in the
   similarity matrix?
2. Why are CLIP/SigLIP features a good starting point for a VLM?
3. How does cross-attention fusion differ from contrastive alignment?
4. Describe the backbone-fusion pattern (LLaVA style) in one sentence.
5. What single component turns a VLM into a VLA?

## Go deeper
- Radford et al. (2021), CLIP; Zhai et al. (2023), SigLIP.
- Alayrac et al. (2022), Flamingo; Liu et al. (2023), LLaVA; Beyer et al. (2024),
  PaliGemma.

Next: [Module 16.5 - Diffusion and Action Heads](../05_diffusion_action_heads/),
then revisit [Part 10 - VLA Models](../../part10_vla_models/) with full context.
