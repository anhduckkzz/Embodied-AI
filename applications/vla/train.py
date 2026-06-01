"""Train and evaluate the end-to-end multi-task VLA by behavioral cloning.

Run:  python -m applications.vla.train

Pipeline:
  1. Collect demonstrations from the scripted expert across BOTH task families
     (object grounding and spatial grounding), mixing many random scenes.
  2. Train the VLA policy (applications/vla/model.py, built on torch framework
     modules) to imitate the expert from image + instruction + proprioception.
  3. Evaluate success per task family, and show that the same scene produces
     different behavior under different instructions (true language conditioning).

This is the application-layer VLA. It does not import the from-scratch ``dl``
library; that code is for learning the mechanism, this code uses the framework.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from applications.vla.env import (GRID, IMG, MAX_INSTR_LEN, N_ACTIONS, VOCAB_SIZE,
                                  MultiTaskVLAEnv)
from applications.vla.model import VLAPolicy


def collect_dataset(n_episodes: int = 1200, seed: int = 0) -> Dict[str, np.ndarray]:
    """Roll out the expert across random multi-task episodes; record (obs, action)."""
    env = MultiTaskVLAEnv(seed=seed)
    images, instrs, proprios, actions = [], [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        for _ in range(2 * GRID + 1):
            a = env.expert_action()
            images.append(obs["image"])
            instrs.append(obs["instruction"])
            proprios.append(obs["proprio"])
            actions.append(a)
            obs, _, done, _ = env.step(a)
            if done:
                break
    return {
        "image": np.stack(images),
        "instruction": np.stack(instrs),
        "proprio": np.stack(proprios),
        "action": np.array(actions, dtype=np.int64),
    }


def train(model: VLAPolicy, data: Dict[str, np.ndarray], epochs: int = 8,
          batch_size: int = 256, lr: float = 3e-4, device: str = "cpu",
          verbose: bool = True) -> List[float]:
    img = torch.as_tensor(data["image"], device=device)
    ins = torch.as_tensor(data["instruction"], device=device)
    pro = torch.as_tensor(data["proprio"], device=device)
    act = torch.as_tensor(data["action"], device=device)
    n = len(act)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(img[idx], ins[idx], pro[idx])
            loss = F.cross_entropy(logits, act[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        history.append(total / n)
        if verbose:
            print(f"epoch {epoch + 1:>2}  bc_loss {history[-1]:.4f}")
    return history


@torch.no_grad()
def evaluate(model: VLAPolicy, task_type: str, n_episodes: int = 200,
             seed: int = 999, device: str = "cpu") -> float:
    """Greedy rollout success rate for one task family."""
    env = MultiTaskVLAEnv(seed=seed)
    model.eval()
    successes = 0
    for _ in range(n_episodes):
        obs = env.reset(task_type=task_type)
        for _ in range(2 * GRID + 1):
            a = _act(model, obs, device)
            obs, _, done, info = env.step(a)
            if done:
                successes += int(info["success"])
                break
    return successes / n_episodes


def _act(model: VLAPolicy, obs: Dict, device: str) -> int:
    image = torch.as_tensor(obs["image"], device=device).unsqueeze(0)
    instr = torch.as_tensor(obs["instruction"], device=device).unsqueeze(0)
    proprio = torch.as_tensor(obs["proprio"], device=device).unsqueeze(0)
    return int(model(image, instr, proprio).argmax(-1).item())


def show_language_conditioning(model: VLAPolicy, device: str = "cpu") -> None:
    """One fixed scene, several instructions: the policy should chase whichever
    target the language names, proving it is genuinely conditioned on the words."""
    from applications.vla.env import COLOR_NAMES
    env = MultiTaskVLAEnv(seed=7)
    env.reset(task_type="color")
    print("\nSame scene, different instruction -> different first move:")
    for color in COLOR_NAMES[:env.n_objects]:
        obs = env.set_task("color", color)
        a = _act(model, obs, device)
        name = {0: "up", 1: "down", 2: "left", 3: "right", 4: "interact"}[a]
        gr, gc = env.goal_cell
        ar, ac = env.agent
        print(f"  'go to the {color:6s}'  target at ({gr},{gc}), agent at ({ar},{ac})"
              f"  -> {name}")


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"End-to-end multi-task VLA (device: {device})\n")

    print("1) Collecting expert demonstrations across both task families ...")
    data = collect_dataset(n_episodes=2500)
    print(f"   {len(data['action'])} (observation, action) pairs\n")

    model = VLAPolicy(IMG, VOCAB_SIZE, MAX_INSTR_LEN, N_ACTIONS, dim=128).to(device)
    print(f"2) Training the VLA ({sum(p.numel() for p in model.parameters()):,} params)")
    train(model, data, epochs=25, lr=5e-4, device=device)

    print("\n3) Evaluating per task family:")
    color_sr = evaluate(model, "color", device=device)
    spatial_sr = evaluate(model, "spatial", device=device)
    print(f"   object grounding ('go to the <color>') : {color_sr:.0%}")
    print(f"   spatial grounding ('go to the <region>'): {spatial_sr:.0%}")

    show_language_conditioning(model, device)
    print("\nOne policy, two task families, image + language + state in -> action out. "
          "Scale the env, encoders, and data and this is a real VLA.")


if __name__ == "__main__":
    main()
