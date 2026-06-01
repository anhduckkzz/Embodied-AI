# Module 16.3 - Vision Models: CNNs and Vision Transformers

Code: [`dl/vision.py`](../../../dl/vision.py). Demo:
[`build_vit.py`](build_vit.py).

To understand a vision-language model you first need to understand how a model
turns an image into features. There are two dominant designs, and modern VLMs use
the second.

## 1. Images as data

An image is a grid of pixels, typically three channels (red, green, blue). A
model needs to convert this grid into a vector (or a sequence of vectors) that
captures content: edges, textures, parts, objects. The two architectures below
differ in how they do this.

## 2. Convolutional Neural Networks (CNNs)

A CNN slides small learned filters (kernels) across the image. Each filter
detects a local pattern wherever it appears. Key properties:

- Locality: a filter only looks at a small neighborhood, matching the fact that
  nearby pixels are related.
- Weight sharing and translation invariance: the same filter is applied
  everywhere, so a feature learned in one location is detected anywhere. This is
  efficient and a strong, useful prior for images.
- Hierarchy: stacking conv layers with pooling builds from edges (early layers)
  to textures and parts (middle) to objects (deep), while spatial size shrinks
  and the channel (feature) dimension grows.

CNNs (LeNet, AlexNet, VGG, ResNet) dominated vision for a decade and are still
excellent, especially with limited data. See [`SimpleCNN`](../../../dl/vision.py).

## 3. Vision Transformers (ViT)

A ViT applies the transformer (Module 16.1) directly to images:

1. Split the image into fixed size patches (for example 16x16 pixels).
2. Flatten and linearly embed each patch into a vector. Now the image is a
   sequence of patch tokens, exactly like a sequence of word tokens. This is the
   key insight that unifies vision and language.
3. Prepend a learnable [CLS] token and add position embeddings.
4. Run a transformer encoder. Every patch attends to every other patch, so the
   model captures global relationships from the first layer.
5. Read the [CLS] token as the image summary, then classify or pass it to a
   downstream multimodal model.

See [`PatchEmbedding`](../../../dl/vision.py) (patching as a strided convolution)
and [`ViT`](../../../dl/vision.py).

## 4. CNN vs ViT

- CNNs bake in locality and translation invariance, so they learn well from less
  data. ViTs have weaker built-in priors, so they need more data (or strong
  pretraining) but then scale better and capture global context naturally.
- ViTs reuse the same machinery as language models, which is exactly why a single
  transformer can later process image patches and text tokens together. This is
  the architectural reason VLMs are possible.

## 5. How vision models are pretrained (what powers VLMs)

The vision encoder in a modern VLM is rarely trained from scratch on labels.
Instead it is pretrained with self-supervised or weakly-supervised objectives:

- Contrastive image-text (CLIP, SigLIP): align images with their captions
  (Module 16.4). Produces features that already understand semantics and
  language, which is why these are the default VLM encoders.
- Self-distillation / masked modeling (DINOv2, MAE): learn rich features from
  images alone, no labels.

These pretrained encoders are the eyes that a VLM and then a VLA build on.

## 6. Build it yourself

[`build_vit.py`](build_vit.py) trains a small ViT from
[`dl/vision.py`](../../../dl/vision.py) on a synthetic image classification task
(horizontal vs vertical stripe patterns) and reports accuracy. It runs on a CPU
in seconds and shows the full patch-to-prediction pipeline.

```bash
python curriculum/part16_foundation_models/03_vision_models/build_vit.py
```

## Project
1. Run `build_vit.py`. Confirm it reaches high accuracy. Change the patch size and
   observe the effect on the number of tokens and on accuracy.
2. Swap the ViT for `SimpleCNN` on the same task and compare. Then make the task
   harder (smaller patterns, more noise) and see which copes better.
3. Use `ViT(..., return_features=True)` to extract image embeddings; this is the
   exact output a VLM consumes in Module 16.4.

## Check your understanding
1. What two priors does a CNN build in, and why are they useful for images?
2. How does a ViT turn an image into a sequence of tokens?
3. Why does a ViT typically need more data than a CNN?
4. What architectural fact makes it possible to process images and text in one
   transformer?
5. How are the vision encoders used by VLMs usually pretrained?

## Go deeper
- Dosovitskiy et al. (2020), An Image Is Worth 16x16 Words (ViT).
- He et al. (2016), Deep Residual Learning (ResNet); He et al. (2022), MAE.
- Oquab et al. (2023), DINOv2; Zhai et al. (2023), SigLIP.

Next: [Module 16.4 - Vision-Language Models](../04_vlm_multimodal/).
