# Part 16 - Foundation Models: the architecture stack under VLA

Code: the [`dl/`](../../dl/) library (attention, transformers, vision, language,
multimodal), built from scratch to be read and run.

This part teaches the model architectures that modern robotics and autonomous
driving are built on, from the bottom up. The motivation is direct: a
vision-language-action model is a three-modality model. You cannot understand it
without understanding a vision-language model. You cannot understand a VLM without
understanding vision models, language models, and how the two are unified. And
all of these are built on the transformer. So we start at the transformer and
climb.

## The dependency chain

```
attention  ->  transformer  ->  language model  ----\
                            \                         \
                             \->  vision model  -------->  vision-language model (VLM)  ->  VLA
                                                  (multimodal fusion)        (add an action head)
```

Each module below builds on the previous one and is backed by runnable code in
[`dl/`](../../dl/) that you can train on a laptop in seconds to a minute.

## Modules

| # | Module | Build-it demo | Core code |
|---|--------|---------------|-----------|
| 16.1 | [Attention and Transformers](01_attention_transformers/) | `build_gpt.py` (train a tiny GPT) | [`dl/attention.py`](../../dl/attention.py), [`dl/transformer.py`](../../dl/transformer.py) |
| 16.2 | [Language Models](02_language_models/) | (uses `build_gpt.py`) | [`dl/tokenizer.py`](../../dl/tokenizer.py), `dl/transformer.py` |
| 16.3 | [Vision Models (CNN, ViT)](03_vision_models/) | `build_vit.py` (train a ViT) | [`dl/vision.py`](../../dl/vision.py) |
| 16.4 | [Vision-Language Models](04_vlm_multimodal/) | `build_clip.py` (train CLIP) | [`dl/multimodal.py`](../../dl/multimodal.py) |
| 16.5 | [Diffusion and Action Heads](05_diffusion_action_heads/) | (project) | (Part 12, Part 10) |
| 16.6 | [Efficient Fine-Tuning (LoRA, quantization)](06_efficient_finetuning/) | (project) | Hugging Face peft |

After this part, return to [Part 10 - VLA Models](../part10_vla_models/): it will
read as a straightforward combination of the pieces you built here.

## How to study this part

1. Do the modules in order, running each build-it demo. The demos are small on
   purpose so you see the whole mechanism, not a black box.
2. Read the corresponding file in [`dl/`](../../dl/) alongside each module. The
   code is the lesson; the README is the explanation.
3. Run the test suite to see each component verified:
   `pytest tests/test_dl.py -q`.

## Hardware notes

All build-it demos in this part run on CPU in seconds to a minute, and on the RTX
4060 trivially. The large-model work (running or LoRA fine-tuning real 7B-class
VLMs and VLAs) belongs on the A100; see Module 16.6 for the memory techniques
that make limited-VRAM training possible.

## What the from-scratch library covers

- [`dl/attention.py`](../../dl/attention.py): scaled dot-product attention,
  multi-head attention, causal masking, self vs cross attention.
- [`dl/transformer.py`](../../dl/transformer.py): positional encoding, the
  pre-norm transformer block, a bidirectional encoder, and a complete small GPT.
- [`dl/tokenizer.py`](../../dl/tokenizer.py): character tokenizer plus an
  explanation of BPE.
- [`dl/vision.py`](../../dl/vision.py): a small CNN and a Vision Transformer.
- [`dl/multimodal.py`](../../dl/multimodal.py): CLIP contrastive loss, a dual
  encoder, and cross-attention fusion.

Start at [Module 16.1 - Attention and Transformers](01_attention_transformers/).
