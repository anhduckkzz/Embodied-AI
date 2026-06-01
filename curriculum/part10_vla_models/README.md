# Part 10 — Vision-Language-Action (VLA) Models 🤖🧠

> The synthesis. A VLA takes **what the robot sees** (Part 5/6) + **a language
> instruction** and outputs **actions** (Part 3/9), trained with **imitation +
> RL** (Part 1). Everything in this curriculum converges here. A runnable
> toy-VLA lives in
> [`../part1_reinforcement_learning/07_vla_robotics/vla_minidemo.py`](../part1_reinforcement_learning/07_vla_robotics/vla_minidemo.py);
> this part is the full, current picture and a path to real models on your A100.

> Prerequisite for real understanding: work through
> [Part 16 - Foundation Models](../part16_foundation_models/) first. A VLA is a
> three-modality model; to understand it you must understand the transformer
> (16.1), language models (16.2), vision models (16.3), and how vision and
> language are unified into a VLM (16.4), plus the action heads (16.5). Then this
> part reads as "a VLM with an action head." To train one in practice, see the
> [LeRobot guide](../part9_manipulation_teleop/lerobot_guide.md).

---

## 1. What a VLA is, precisely

A single policy
```
   π( action_sequence | images, language_instruction, proprioception )
```
that maps perception + instruction to robot actions. It's the robotics analogue
of an instruction-tuned multimodal LLM: **one generalist, language-conditioned,
multi-task** controller instead of many narrow policies. The bet — proven in NLP
and vision — is that **scale + diversity → generalization** to new objects,
scenes, and instructions.

## 2. Architecture (and where each earlier part lives)

```
 images ─► vision encoder (ViT / SigLIP / DINOv2, Part 6) ─┐
                                                           ├─► transformer backbone ─► action head ─► actions (Part 3/9)
 instruction ─► text tokenizer (LLM, Part 10) ────────────┤        (a VLM/LLM)         (de)tokenizer    Δpose, gripper
 proprioception (joint state, Part 5) ─────────────────────┘
```

- **Vision encoder** — the perception backbones from Part 6, now as the "eyes."
- **Backbone** — usually a **pretrained VLM/LLM** (so the robot inherits semantic
  knowledge: what a "mug" is, what "left of" means).
- **Action head** — two dominant designs:
  - **Action tokenization** (RT-2, OpenVLA): discretize each action dimension
    into bins, predict them as *text tokens* — actions become "another language."
  - **Continuous / diffusion / flow** heads (Octo, π0, Diffusion Policy): output
    continuous **action chunks** (several future steps at once) via diffusion or
    flow-matching. Smoother, higher-frequency control.
- **Actions** are typically **end-effector deltas in SE(3)** + gripper — the
  exact objects from Part 3.

## 3. How VLAs are trained (this whole repo, in three stages)

1. **VLM pretraining** — start from an internet-scale vision-language model
   (transfer learning; semantic grounding).
2. **Imitation pretraining — the bulk.** Behavioral Cloning (Part 1 / Module 06)
   on **massive teleoperated demonstrations** (Part 9): next-action prediction on
   `(images, instruction, action)`. **Open X-Embodiment** (~1M+ trajectories, 22
   robots) is the canonical corpus.
3. **RL / preference fine-tuning — the refinement.** Improve beyond the demos via
   interaction + reward, using **PPO/actor-critic (Part 1 / Module 04)** and
   RLHF-style methods. Also fixes **covariate shift** (Part 1 / Module 06): the
   policy visits states the experts never showed, and RL teaches recovery.

> **The slogan again: imitate at scale, then refine with RL.** You learned both
> halves — Part 9 collects the data, Part 1 trains and refines the policy.

## 4. The field map (as of early 2026)

| Model | Who | Key idea (a concept you now own) |
|-------|-----|----------------------------------|
| **RT-1 / RT-2** | Google DeepMind | actions-as-tokens; VLM→robot transfer (BC at scale) |
| **Open X-Embodiment / RT-X** | cross-lab | one BC policy/dataset across many robot embodiments |
| **OpenVLA** | Stanford/Berkeley | open **7B** VLA, action tokenization, LoRA-fine-tunable |
| **Octo** | Berkeley | open transformer policy, **diffusion** action head |
| **π0 / π0.5** | Physical Intelligence | **flow-matching** continuous actions, cross-embodiment |
| **Diffusion Policy** | Columbia/MIT/TRI | imitation via conditional **diffusion** over action chunks |
| **GR00T N1** | NVIDIA | humanoid foundation model; sim + RL + imitation |
| **RDT-1B, TinyVLA, SmolVLA** | various | efficient/open VLAs runnable on modest GPUs |

Names churn; the **building blocks don't**: VLM backbone + BC on diverse demos +
an action head + RL/preference fine-tuning.

## 5. Run a real VLA on your hardware

- **Inference (💻 4060, 8 GB):** small/quantized VLAs (SmolVLA, TinyVLA, quantized
  OpenVLA) can run for inference. Great for understanding I/O and latency.
- **Fine-tuning (🅰️ A100):** LoRA / full fine-tune OpenVLA or Octo on a small task
  dataset (LIBERO sim or your own teleop). This is the realistic "train a VLA"
  project for you.
- **Easiest entry — Hugging Face LeRobot:** open VLAs + datasets + training code.
  Load a pretrained policy, run it in sim, fine-tune on a LeRobot/Open-X dataset.

```bash
# Sketch (see LeRobot docs for the current API):
pip install lerobot
# load a pretrained policy, evaluate in a sim env, then fine-tune on a dataset
```

## 6. Open challenges (each ties to an earlier part)
- **Action representation** — tokens vs diffusion/flow (Parts 3, 9).
- **Robustness / covariate shift** — DAgger + RL fine-tuning (Part 1).
- **3D & spatial reasoning** — depth/point clouds, BEV (Part 6).
- **Long-horizon tasks** — credit assignment, hierarchy, chunking (Part 1).
- **Sim-to-real & data scale** — domain randomization, teleop pipelines (Parts 4, 9).
- **Safety & evaluation** — hard, open; preference/constraint methods (Part 1).

## 7. VLA vs the modular stack (when to use which)
- **Modular** (Parts 5→6→7→8→9 as separate, interpretable modules): reliable,
  debuggable, certifiable — today's self-driving and industrial robots.
- **End-to-end VLA**: generalizes to novelty and language, less brittle to
  hand-engineering — the research frontier and likely future. In practice,
  **hybrids** are emerging (learned components inside a modular safety scaffold).

## 🛠️ Project (your capstone)
1. Re-run the toy VLA
   ([`vla_minidemo.py`](../part1_reinforcement_learning/07_vla_robotics/vla_minidemo.py))
   and articulate, line by line, which part of *this* curriculum each piece maps to.
2. On the A100: fine-tune **OpenVLA** or **Octo** (via LeRobot) on a **LIBERO**
   task; evaluate success rate; then try a harder instruction to see covariate
   shift.
3. Write a one-page design for a VLA for *your* target application (Part 11):
   sensors, action space, data-collection (teleop) plan, train recipe (BC→RL).

## ✅ Check your understanding
1. What three inputs does a VLA map to actions, and in what space are the actions?
2. Action tokenization vs diffusion/flow heads — trade-offs?
3. Which training stage is imitation, which is RL, and why are both needed?
4. How does each of Parts 3, 6, 9, and 1 show up inside a VLA?
5. When would you ship a modular stack instead of an end-to-end VLA?

## 📖 Go deeper
- Papers: Open X-Embodiment, RT-2, OpenVLA, Octo, π0, Diffusion Policy, GR00T.
- Hugging Face **LeRobot** (code + datasets + models); **LIBERO** benchmark.
- Surveys: "Vision-Language-Action Models" (2024–2025); "Foundation Models for Robotics."

➡️ **Next:** [Part 11 — Applications](../part11_applications/): VLA and the modular
stack applied to autonomous driving, drones, and humanoids.
