# Module 07 — VLA & robotics: the destination 🤖

> Code: [`vla_minidemo.py`](vla_minidemo.py) · Notebook: `notebook.ipynb`

You made it. This module connects everything — value functions, policy
gradients, continuous control, imitation — to **Vision-Language-Action (VLA)
models**: neural networks that take **camera images + a language instruction**
and output **robot actions**. It's a conceptual roadmap *plus* a runnable
mini-demo so the abstraction becomes concrete.

## 1. What is a VLA model?

A single policy
```
        π( action | image(s), language instruction, robot state )
```
that maps what the robot **sees** and what it's **told** to **what it does**.
"Pick up the red block and put it in the bowl" → a stream of end-effector poses
and gripper commands. VLAs are the robotics analogue of instruction-tuned LLMs:
**generalist, language-conditioned, multi-task** controllers.

They exist because narrow, single-task robot policies don't scale. The bet —
mirroring NLP and vision — is that **one big model trained on diverse data**
generalizes to new objects, scenes, and instructions better than many small ones.

## 2. The anatomy of a VLA

```
  ┌────────────┐   ┌──────────────┐
  │  camera(s) │──►│ vision encoder│──┐
  └────────────┘   │ (ViT/SigLIP) │  │   ┌───────────────┐   ┌──────────────┐
                   └──────────────┘  ├──►│  transformer  │──►│ action head  │──► actions
  ┌────────────┐   ┌──────────────┐  │   │  (LLM backbone)│   │ (de)tokenizer│   (Δpose, grip)
  │ "stack the │──►│ text tokenizer│──┘   └───────────────┘   └──────────────┘
  │  blocks"   │   └──────────────┘
  └────────────┘
```

- **Vision encoder** turns images into tokens (often SigLIP / DINO / ViT).
- **Language** is tokenized as usual.
- A **transformer backbone** (frequently a pretrained VLM/LLM) fuses them.
- An **action head** emits actions. Two dominant designs:
  - **Action tokenization** (RT-2, OpenVLA): discretize each action dimension
    into bins and predict them as *text tokens* — actions become "another
    language" the LLM speaks.
  - **Continuous/diffusion heads** (π0, Octo, diffusion policies): output
    continuous action chunks directly, often via flow-matching/diffusion.

## 3. How is a VLA trained? (here's where this whole repo pays off)

Three stages, each a concept you've already learned:

1. **VLM pretraining** — start from an internet-scale vision-language model
   (semantic knowledge of objects, words, relations). *Transfer learning.*
2. **Imitation pretraining (the bulk)** — **Behavioral Cloning (Module 06)** on
   massive robot-demonstration datasets (e.g. **Open X-Embodiment**, ~1M+
   trajectories across many robots). This is supervised next-action prediction.
3. **RL / preference fine-tuning (the refinement)** — improve beyond the
   demonstrations using interaction and reward. The machinery is **PPO /
   actor-critic (Module 04)** and **RLHF**-style preference optimization. RL also
   underpins **sim-to-real** locomotion and dexterity, where **PPO in massively
   parallel simulators** and **SAC (Module 05)** are standard.

> **The slogan:** *imitate at scale, then refine with RL.* You learned both
> halves in Modules 06 and 04.

## 4. A field map (as of early 2026)

| Model | Group | Idea you already know |
|-------|-------|------------------------|
| **RT-1 / RT-2** | Google DeepMind | actions-as-tokens; VLM → robot transfer (BC at scale) |
| **Open X-Embodiment / RT-X** | cross-lab | one BC dataset/policy across many robot embodiments |
| **OpenVLA** | Stanford/Berkeley | open 7B VLA, action tokenization, fine-tunable (BC) |
| **Octo** | Berkeley | open transformer policy, diffusion action head (BC) |
| **π0 / π0.5 (pi-zero)** | Physical Intelligence | flow-matching continuous actions; cross-embodiment |
| **Diffusion Policy** | Columbia/MIT | imitation via a conditional diffusion model |
| **GR00T** | NVIDIA | humanoid foundation model; sim + RL + imitation |

The names change fast; the **building blocks don't**: a VLM backbone, BC
pretraining on diverse demos, an action head, and RL/preference fine-tuning.

## 5. Key challenges (and which module each connects to)

- **Action representation** — tokenized vs continuous/diffusion → *policy
  parameterization* (Modules 03–05).
- **Covariate shift / robustness** — *the BC problem* (Module 06); fixed by
  DAgger-like data and RL fine-tuning.
- **Sample efficiency & safety** — *off-policy & sim-to-real* (Module 05);
  PPO/SAC, domain randomization.
- **Reward design / preferences** — *RLHF & inverse RL* (Modules 04, 06).
- **Long horizons & 3D reasoning** — *credit assignment, γ, GAE* (Modules 00, 04).

## 6. The mini-demo — a language-conditioned policy

[`vla_minidemo.py`](vla_minidemo.py) is a tiny, **dependency-light** "VLA in
spirit": a grid world where the instruction ("go to the red goal" vs "go to the
blue goal") changes the optimal action. The agent's input is
`[observation ⊕ instruction-embedding]` and it's trained two ways:

1. **Behavioral cloning** from a scripted expert (Module 06) — the VLA recipe.
2. *(Optional)* **PPO fine-tuning** with a reward (Module 04) — the refinement.

It deliberately strips away the heavy vision/language encoders so the **core
idea is unmistakable: the same state can demand different actions depending on
the instruction, and one conditioned policy learns to handle both.** Swapping
the toy instruction-embedding for a real text encoder and the grid for camera
images is, conceptually, the whole leap to a real VLA.

```bash
python curriculum/07_vla_robotics/vla_minidemo.py
```

## 7. Where to go from here (your project ideas)

- Fine-tune **OpenVLA** or **Octo** on a small custom dataset (HF + LIBERO sim).
- Reproduce a **Diffusion Policy** on a pick-and-place task.
- Train **PPO** locomotion in **Isaac Lab / MuJoCo**, then sim-to-real concepts.
- Implement **action tokenization** on the continuous LunarLander as a sandbox
  for the RT-2/OpenVLA action representation.

## ✅ Check your understanding
1. What three inputs does a VLA map to actions?
2. Contrast action *tokenization* with *continuous/diffusion* action heads.
3. Which training stage is BC, and which is RL? Why are both needed?
4. Why does the covariate-shift lesson from Module 06 matter for VLAs?
5. In the mini-demo, why must the policy take the instruction as input?

## 📖 Go deeper
- **Open X-Embodiment** (2023) · **RT-2** (2023) · **OpenVLA** (2024) ·
  **Octo** (2024) · **π0** (2024, Physical Intelligence) · **Diffusion Policy** (2023).
- LeRobot (Hugging Face) — open VLA/robot-learning library and datasets.
- Survey: "Foundation Models for Robotics" / "A Survey on Vision-Language-Action Models".

🎉 **You've completed the path from the Bellman equation to robot foundation
models.** The algorithms in `rl/` are the same ones these systems are built on —
now go build something.
