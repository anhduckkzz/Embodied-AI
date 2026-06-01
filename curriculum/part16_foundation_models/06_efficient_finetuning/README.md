# Module 16.6 - Efficient Fine-Tuning on Limited Hardware

How to adapt large pretrained models (VLMs, VLAs, LLMs) on a single GPU. This is
the practical module: it tells you how to actually train on your RTX 4060 (8 GB)
and how to use the A100 well.

## 1. Why fine-tuning, not training from scratch

Foundation models cost enormous compute to pretrain. You almost never train one
from scratch. Instead you take a pretrained model and adapt it to your task or
robot with a small amount of data. The question is how to do this without needing
the memory of full training.

## 2. The memory problem

Full fine-tuning updates every weight, which requires storing, for each
parameter: the weight, its gradient, and optimizer state (Adam keeps two extra
values per parameter). For a 7B-parameter model that is tens of gigabytes,
impossible on 8 GB. The techniques below reduce this drastically.

## 3. Parameter-efficient fine-tuning (PEFT)

- LoRA (Low-Rank Adaptation): freeze the pretrained weights and inject small
  trainable low-rank matrices into each layer. You train only these adapters
  (often less than 1 percent of parameters), so gradients and optimizer state are
  tiny. Quality is close to full fine-tuning for most adaptation tasks. This is
  the default way to fine-tune large models, including VLAs like OpenVLA.
- QLoRA: load the frozen base model in 4-bit quantization (less memory) and train
  LoRA adapters on top. This is what lets a 7B model fine-tune on a single 8 to 24
  GB GPU.
- Adapters, prefix/prompt tuning: other PEFT variants in the same spirit.

## 4. Quantization

Represent weights in fewer bits (8-bit or 4-bit) instead of 16/32-bit floats.

- For inference: run a big model in 4-bit to fit it in limited VRAM, with small
  quality loss. This is how you can run a quantized 7B VLA on the 4060 for
  inference.
- For training: combine with LoRA (QLoRA) so only the small adapters are in full
  precision.

## 5. Memory-saving training techniques

- Gradient checkpointing: recompute activations during the backward pass instead
  of storing them, trading compute for memory.
- Mixed precision (bf16/fp16): use 16-bit math for most operations.
- Gradient accumulation: simulate a large batch by summing gradients over several
  small batches before stepping, so you fit in VRAM while keeping batch statistics.
- Smaller batch size, shorter context, frozen vision encoder: practical levers.

## 6. A concrete plan for your hardware

- RTX 4060 (8 GB): inference on small or 4-bit-quantized models (SmolVLA,
  small ViT/CLIP, quantized 7B). LoRA/QLoRA fine-tuning of small-to-medium models.
  Build-from-scratch demos in this part. Use bf16, gradient checkpointing, small
  batches with accumulation.
- A100 (Colab, 40/80 GB): LoRA fine-tune 7B-class VLAs and LLMs, train diffusion
  policies, larger ViT training, bigger batches. This is where the real VLA
  fine-tuning (Part 9 LeRobot guide, Part 10) happens.

## 7. The tooling

The Hugging Face stack implements all of the above:

- transformers: models and training loops.
- peft: LoRA/QLoRA and other PEFT methods.
- bitsandbytes: 4/8-bit quantization.
- accelerate: device placement, mixed precision, multi-GPU.
- trl: preference optimization (RLHF/DPO, Part 1 Module 14).
- diffusers: diffusion models (Module 16.5).
- lerobot: robot policies and VLAs (Part 9, Part 10).

## Project
1. LoRA-fine-tune a small pretrained language model on a tiny custom dataset with
   `peft`; count trainable versus total parameters and note the memory difference
   from full fine-tuning.
2. Load a model in 4-bit with `bitsandbytes` and confirm the VRAM reduction.
3. Plan a VLA fine-tune: which base model (SmolVLA fits modest hardware), which
   PEFT method, what batch size and precision for the 4060 versus the A100.

## Check your understanding
1. Why is full fine-tuning of a 7B model infeasible on 8 GB, in terms of what
   must be stored per parameter?
2. What does LoRA train, and why does that save so much memory?
3. What does QLoRA add on top of LoRA?
4. What does gradient checkpointing trade, and why is it useful?
5. Give a concrete fine-tuning plan for the 4060 versus the A100.

## Go deeper
- Hu et al. (2021), LoRA; Dettmers et al. (2023), QLoRA.
- Hugging Face peft, bitsandbytes, accelerate, and trl documentation.

Next: revisit [Part 10 - VLA Models](../../part10_vla_models/) and the
[LeRobot guide](../../part9_manipulation_teleop/lerobot_guide.md), now that you
understand every layer of the architecture.
