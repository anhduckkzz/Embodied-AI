# Part 16 — Foundation Models: transformers, diffusion, multimodality

> The deep-learning machinery **underneath** VLAs (Part 10). To understand,
> fine-tune, or build a Vision-Language-Action model you need transformers,
> attention, tokenization, diffusion, and multimodal fusion. This part teaches
> those from the embodied-AI angle.

## 1. The transformer & attention (the engine of modern AI)

- **Tokens & embeddings:** turn input (text, image patches, action bins) into a
  sequence of vectors. The model operates on sequences of tokens regardless of
  modality — which is *why one architecture serves language, vision, and action*.
- **Self-attention:** each token computes **queries, keys, values**; it attends to
  other tokens by `softmax(QKᵀ/√d)·V`, mixing information across the whole
  sequence. This models long-range dependencies that RNNs/CNNs struggle with.
- **Multi-head attention + feed-forward + residual + layernorm** = a transformer
  block; stack many. **Positional encodings** inject order.
- **Why it dominates:** parallelizable, scales with data/compute (scaling laws),
  and is **modality-agnostic**. (Karpathy's *Zero to Hero* builds one from scratch
  — the best way to truly get it; Part 0.)

## 2. Vision transformers & visual encoders

- **ViT:** split an image into patches → tokens → transformer. Now the default
  vision backbone.
- **Self-supervised visual encoders:** **DINOv2** (self-distillation), **MAE**
  (masked autoencoding), **SigLIP/CLIP** (image-text contrastive). These produce
  semantic image features **without task labels** — and are exactly the "eyes" of
  VLAs (Parts 6, 10). CLIP/SigLIP also align images and text in one space, the
  basis of language grounding.

## 3. Large Language Models (the "brain"/backbone)

- **Pretraining:** next-token prediction on web-scale text → broad knowledge +
  reasoning. **Instruction tuning + RLHF** (Module 14) make them follow commands.
- **In VLAs:** an LLM (or VLM) backbone provides semantic grounding ("what is a
  mug," "what does *left of* mean") and is adapted to emit **actions** — often via
  **action tokenization** (RT-2, OpenVLA: discretize actions into tokens the LLM
  "speaks").
- **Efficient fine-tuning:** **LoRA/QLoRA** adapt big models with tiny trainable
  adapters — *this is how you fine-tune a 7B VLA on your A100* (full fine-tune
  would need far more memory).

## 4. Diffusion & flow models (the modern action head)

- **Diffusion models:** learn to **denoise** — start from noise, iteratively
  refine to a sample from the data distribution. Power image generation; in
  robotics they model **multimodal action distributions** ("go left *or* right")
  that a single Gaussian can't (**Diffusion Policy**, Part 12).
- **Flow matching:** a newer, often faster continuous-generative formulation;
  **π0** uses it for high-frequency continuous robot actions (Part 10).
- These are the **continuous/diffusion action heads** contrasted with action
  tokenization in Part 10.

## 5. Multimodal fusion (vision + language + action + state)

- **How modalities meet:** project each (image tokens, text tokens, proprioception,
  action tokens) into a shared token space and let attention fuse them — the VLA
  architecture (Part 10).
- **Cross-attention vs concatenation;** modality-specific encoders + a shared
  backbone; **late vs early fusion** (echoes Part 6's sensor fusion).
- **Tokenizing actions/state:** discretization (bins), VAE/latent codes, or direct
  continuous heads — a key design choice for embodied models.

## 6. Why "foundation model" (and scaling) matters for robots
- **Transfer:** internet-pretrained vision/language knowledge bootstraps robots
  that have far less data than NLP/vision.
- **Generalization:** one big model across tasks/embodiments (Open X-Embodiment)
  beats many narrow ones — the bet behind VLAs.
- **Scaling laws:** more data/params/compute → predictably better; robotics is
  early on this curve (data is the bottleneck → teleop pipelines, Part 9).

## 7. On your hardware
- 💻 **RTX 4060 (8 GB):** run/inspect small models (ViT, CLIP, SmolVLM,
  quantized 7B with QLoRA inference), build a transformer from scratch, fine-tune
  small nets. Use **gradient checkpointing / 4-bit** to fit.
- 🅰️ **A100:** fine-tune 7B-class VLAs/LLMs (LoRA), train diffusion policies,
  larger ViT training. The Hugging Face stack (`transformers`, `peft`, `diffusers`,
  `trl`, `lerobot`) is your toolkit.

## 🛠️ Project
1. 💻 Build a tiny **transformer** from scratch (follow Karpathy) and train it on a
   toy sequence — *then* you'll truly understand attention.
2. 💻 Use a pretrained **CLIP/SigLIP** to score image–text similarity; see language
   grounding that a VLA reuses.
3. 🅰️ **LoRA-fine-tune** a small LLM or a VLA (via `peft`/`lerobot`) on a tiny
   dataset; observe how few parameters adapt the whole model.
4. 💻 Train a 1-D **Diffusion Policy** to fit a multimodal action distribution; see
   why diffusion beats a single Gaussian for imitation (Part 12).

## ✅ Check your understanding
1. What does self-attention compute, and why is it modality-agnostic?
2. What do self-supervised encoders (DINOv2/CLIP) give a VLA, and how trained?
3. Why does action tokenization let an LLM output robot actions?
4. Why are diffusion/flow heads good for robot actions vs a single Gaussian?
5. What is LoRA and why is it the practical way to fine-tune big models on a 4060/A100?

## 📖 Go deeper
- Vaswani et al. (2017) *Attention Is All You Need*; Karpathy *Zero to Hero* +
  nanoGPT. ViT (Dosovitskiy 2020), CLIP (Radford 2021), DINOv2 (2023).
- Ho et al. (2020) DDPM; Lipman et al. (2022) Flow Matching; Hu et al. (2021) LoRA.
- Hugging Face `transformers`/`diffusers`/`peft`/`trl` docs.

➡️ **Next:** [Part 17 — Legged Locomotion & Humanoids](../part17_legged_locomotion/),
where massively-parallel RL meets real dynamics — and revisit
[Part 10 — VLA](../part10_vla_models/) now that you know the machinery.
