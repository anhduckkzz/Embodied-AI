# End-to-end multi-task VLA (application layer)

This is a small but complete Vision-Language-Action policy. It is the
application-layer counterpart to the from-scratch teaching code in
[`dl/`](../../dl/), and it follows a deliberate separation of concerns:

- The `dl/`, `rl/`, and `robotics/` libraries exist to learn the mechanisms. You
  read them to understand how attention, transformers, filters, and control work.
  They are not meant to be wired into applications.
- This `applications/` area is the real-system layer. It is built on the
  production framework (PyTorch's own `nn.MultiheadAttention` and
  `nn.TransformerEncoder`), exactly as you would build a real model. It does not
  import from `dl/`.

In other words: learn the mechanism from `dl/`, then build with the framework here.

## What it does

One policy handles two different families of language instruction in the same
world, which is what makes it multi-task rather than a single-task demo:

1. Object grounding: "go to the red". The policy must read the instruction to
   choose a color and read the image to find where that color is. Vision is
   required.
2. Spatial grounding: "go to the center". The policy must map a spatial word to a
   location and use its own position. Proprioception and language are required.

The observation is a real signal triple, the same shape a real VLA consumes:

```
image (rendered RGB) + instruction (tokenized language) + proprioception (state)
        -> VLA policy -> action (move or interact)
```

## Architecture

```
image  -> CNN encoder -> grid of vision tokens (each carries its cell coordinate)
text   -> embedding + Transformer encoder -> instruction tokens
state  -> linear -> one proprioception token
              \
               cross-attention: instruction tokens attend over [vision, state]
              /
   target word token + raw proprio -> MLP action head -> action logits
```

The cross-attention is the fusion step: the language selects what in the image
and state matters. This is the same idea as a real VLM backbone with an action
head (Part 16, Part 10), at a size that trains on a laptop CPU.

Files:
- [`env.py`](env.py): the multi-task, image-observation environment and a
  scripted expert.
- [`model.py`](model.py): the VLA policy, built on PyTorch framework modules.
- [`train.py`](train.py): behavioral-cloning training, per-family evaluation, and
  a language-conditioning demonstration.

## Run it

```bash
python -m applications.vla.train
```

This collects expert demonstrations across both task families, trains the policy,
reports success per family, and shows that the same scene yields different
behavior under different instructions (true language conditioning).

## How this connects to real VLAs

This is the imitation-learning recipe (Part 12) applied to a multimodal policy
(Part 16) that emits actions (Part 10). To go from here to a production VLA you
would:

- Replace the toy CNN and word embedding with pretrained encoders (a ViT or
  SigLIP vision encoder and a language model), adapted with LoRA (Part 16.6).
- Replace the grid world with a robot environment and continuous actions
  (Part 5, Part 9), and a diffusion or flow action head (Part 16.5).
- Scale the demonstration data via teleoperation and train with a framework such
  as LeRobot (see the [LeRobot guide](../../curriculum/part9_manipulation_teleop/lerobot_guide.md)).

The structure, image plus instruction plus state in and actions out trained by
imitation, is exactly the same.

## Extend it (project ideas)

1. Add a third task family, for example "avoid the red, go to the blue", which
   requires compositional language understanding.
2. Swap the toy CNN for a torchvision ResNet or a `timm` ViT as the vision
   encoder, and compare.
3. Add action chunking (predict several steps at once) as in ACT and Diffusion
   Policy (Part 12, Part 16.5).
4. Replace behavioral cloning with DAgger to fix covariate shift (Part 12).
