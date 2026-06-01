# LeRobot: train and fine-tune real robot policies and VLAs

LeRobot is Hugging Face's open library for end-to-end robot learning. It is the
most practical on-ramp to actually training the policies and VLAs this curriculum
builds toward, and it is the basis of the Hugging Face Robotics Course. This guide
explains what it provides and how to use it on your hardware.

References: LeRobot docs (https://huggingface.co/docs/lerobot), the GitHub repo
(https://github.com/huggingface/lerobot), and the HF Robotics Course
(https://huggingface.co/learn/robotics-course).

## 1. What LeRobot gives you

- Datasets: a standard `LeRobotDataset` format and many shared datasets on the
  Hub (teleoperated demonstrations, the raw material for imitation, Part 12).
- Policies: ready implementations of the algorithms in this curriculum, including
  ACT (action chunking transformer, Part 12), Diffusion Policy (Parts 12, 16.5),
  TD-MPC and VQ-BeT, and VLAs such as SmolVLA and the Pi0 family (Parts 10, 16).
- Training and evaluation: a unified training entry point, simulation
  environments, and tools to record and replay episodes.
- Hardware support: drivers for low-cost arms (for example SO-100/SO-101) so you
  can teleoperate, record data, and deploy on real hardware later.

## 2. Install

```bash
pip install lerobot
# optional extras for specific simulation environments / policies, see the docs
```

The library is built on PyTorch and the Hugging Face stack (Part 16.6).

## 3. The core workflow

The same loop you learned conceptually in Parts 9, 12, and 16, made concrete:

```
record demos (teleop)  ->  LeRobotDataset  ->  train a policy  ->  evaluate in sim  ->  deploy
```

1. Get data. Use an existing dataset from the Hub or record your own by
   teleoperating a robot (Part 9). Datasets store synchronized camera images,
   robot state, and actions, plus a natural-language task description.
2. Pick a policy. Start with ACT: it is lightweight, trains fast, needs no extra
   dependencies, and is the recommended first model. Move to Diffusion Policy for
   multimodal behavior (Part 16.5), or SmolVLA / Pi0 for language-conditioned VLA.
3. Train with the provided training script (configured by the policy and dataset).
4. Evaluate in the matching simulation environment, then on hardware.

## 4. Choosing a policy (maps to the curriculum)

| Policy | What it is | Curriculum link | Hardware |
|--------|------------|-----------------|----------|
| ACT | action chunking transformer, imitation | Part 12, Part 16.1 | trains on the 4060 |
| Diffusion Policy | diffusion action head, multimodal | Part 16.5 | 4060 small, A100 comfortable |
| SmolVLA | 450M vision-language-action model | Part 10, Part 16.4 | fine-tune on A100 (LoRA on 4060) |
| Pi0 / Pi0-FAST | flow-matching / autoregressive VLA | Part 10, Part 16.5 | A100 |

SmolVLA provides a pretrained base (about 450M parameters); fine-tuning it for
roughly 20k steps takes on the order of a few hours on a single A100, which is the
realistic "train your own VLA" project for you. On the 4060, prefer ACT for full
training and use LoRA/quantization (Part 16.6) for VLA work.

## 5. A minimal training sketch

The exact commands track the current docs, but the shape is:

```bash
# Train ACT on a dataset (CLI form; see docs for the current flags)
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=<hub-dataset-id> \
  --output_dir=outputs/act_run

# Fine-tune the SmolVLA base on your dataset
lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.repo_id=<your-dataset> \
  --output_dir=outputs/smolvla_finetune
```

Always check `https://huggingface.co/docs/lerobot` for the current API; LeRobot is
evolving quickly (recent releases added streaming dataset recording and loading
simulation environments from the Hub).

## 6. Suggested progression

1. Work through the Hugging Face Robotics Course units alongside Parts 1, 9, and
   12 of this curriculum.
2. Load an existing LeRobot dataset and inspect a few episodes: look at the image,
   state, action, and task fields. This is the `(observation, instruction,
   action)` tuple from Part 10 made real.
3. Train ACT on a small simulation dataset end to end. Evaluate it.
4. Fine-tune SmolVLA on the A100 for a language-conditioned task; compare success
   rates and observe covariate shift (Part 12) on harder instructions.
5. If you have hardware, record your own teleop dataset and train on it.

## 7. How this closes the loop with the curriculum

LeRobot is where the theory becomes a working robot policy:

- Imitation learning (Part 12) is the training objective.
- The policies are transformers with diffusion or tokenized action heads
  (Part 16).
- VLAs combine a VLM backbone with an action head (Parts 10, 16.4, 16.5).
- RL and preference fine-tuning (Part 1, Module 14) refine the imitation-trained
  policy.

Read the architecture first (Part 16), then use LeRobot to train the real thing.
