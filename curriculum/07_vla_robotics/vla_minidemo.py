"""A tiny "VLA in spirit": an instruction-conditioned grid-world policy.

This strips a Vision-Language-Action model down to its irreducible core so the
key idea is unmistakable:

    The SAME observation can require DIFFERENT actions depending on the
    language instruction, and a single conditioned policy learns both.

We replace the heavy parts of a real VLA with toy stand-ins:
  * "vision"   -> the agent's (x, y) position in a 5x5 grid
  * "language" -> a 1-bit instruction: go to the RED goal or the BLUE goal
  * "action"   -> one of {up, down, left, right}

Training follows the real VLA recipe in miniature:
  1. Behavioral Cloning from a scripted expert      (Module 06)
  2. (optional) PPO-style reward fine-tuning          (Module 04)

Run:  python curriculum/07_vla_robotics/vla_minidemo.py
Deps: numpy, torch only (no gym).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 5
ACTIONS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}  # up, down, left, right
RED_GOAL = (0, 0)
BLUE_GOAL = (GRID - 1, GRID - 1)


# --------------------------------------------------------------------------
# Environment: instruction decides which goal is rewarding.
# --------------------------------------------------------------------------
class InstructedGridWorld:
    """5x5 grid with two goals. The instruction picks the target goal.

    Observation = (x, y) normalized to [0,1]. Instruction = 0 (red) or 1 (blue).
    Reward = +1 on reaching the instructed goal, small step penalty otherwise.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, instruction: int = None):
        self.instruction = int(self.rng.integers(2)) if instruction is None else instruction
        # start somewhere that isn't a goal
        while True:
            self.pos = (int(self.rng.integers(GRID)), int(self.rng.integers(GRID)))
            if self.pos not in (RED_GOAL, BLUE_GOAL):
                break
        return self._obs(), self.instruction

    def _obs(self):
        return np.array([self.pos[0] / (GRID - 1), self.pos[1] / (GRID - 1)], dtype=np.float32)

    def goal(self):
        return RED_GOAL if self.instruction == 0 else BLUE_GOAL

    def step(self, action: int):
        dx, dy = ACTIONS[int(action)]
        x = min(max(self.pos[0] + dx, 0), GRID - 1)
        y = min(max(self.pos[1] + dy, 0), GRID - 1)
        self.pos = (x, y)
        if self.pos == self.goal():
            return self._obs(), 1.0, True
        return self._obs(), -0.05, False


def scripted_expert(obs, instruction):
    """Greedy expert: step toward the instructed goal (Manhattan)."""
    x, y = round(obs[0] * (GRID - 1)), round(obs[1] * (GRID - 1))
    gx, gy = RED_GOAL if instruction == 0 else BLUE_GOAL
    if x < gx:
        return 3  # right
    if x > gx:
        return 2  # left
    if y < gy:
        return 1  # down
    return 0      # up


# --------------------------------------------------------------------------
# The conditioned policy: input = [observation ; instruction embedding].
# This concatenation IS the heart of a VLA — vision + language -> action.
# --------------------------------------------------------------------------
class ConditionedPolicy(nn.Module):
    def __init__(self, obs_dim=2, n_instructions=2, n_actions=4, embed_dim=8, hidden=64):
        super().__init__()
        self.instr_embed = nn.Embedding(n_instructions, embed_dim)  # toy "language encoder"
        self.net = nn.Sequential(
            nn.Linear(obs_dim + embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs, instruction):
        e = self.instr_embed(instruction)
        return self.net(torch.cat([obs, e], dim=-1))  # logits over actions


# --------------------------------------------------------------------------
# Stage 1 — Behavioral Cloning from the scripted expert (the VLA pretraining).
# --------------------------------------------------------------------------
def collect_demos(n_episodes=400, seed=0):
    env = InstructedGridWorld(seed)
    obs_list, instr_list, act_list = [], [], []
    for _ in range(n_episodes):
        obs, instr = env.reset()
        for _ in range(GRID * GRID):
            a = scripted_expert(obs, instr)
            obs_list.append(obs)
            instr_list.append(instr)
            act_list.append(a)
            obs, _, done = env.step(a)
            if done:
                break
    return (np.array(obs_list), np.array(instr_list, dtype=np.int64),
            np.array(act_list, dtype=np.int64))


def train_bc(policy, epochs=60, seed=0):
    obs, instr, act = collect_demos(seed=seed)
    obs_t = torch.as_tensor(obs)
    instr_t = torch.as_tensor(instr)
    act_t = torch.as_tensor(act)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for ep in range(epochs):
        logits = policy(obs_t, instr_t)
        loss = F.cross_entropy(logits, act_t)   # imitation = classification
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            print(f"  BC epoch {ep + 1:>3}  loss={loss.item():.4f}")
    return policy


# --------------------------------------------------------------------------
# Evaluation — does ONE policy follow BOTH instructions from the SAME states?
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(policy, n_episodes=200, seed=123):
    env = InstructedGridWorld(seed)
    successes = 0
    for _ in range(n_episodes):
        obs, instr = env.reset()
        for _ in range(2 * GRID):
            logits = policy(torch.as_tensor(obs).unsqueeze(0),
                            torch.as_tensor([instr]))
            a = int(logits.argmax(-1).item())
            obs, _, done = env.step(a)
            if done:
                successes += 1
                break
    return successes / n_episodes


def demonstrate_conditioning(policy):
    """Show that from one start state, the chosen action flips with the instruction."""
    print("\nSame state, different instruction -> different action:")
    mid = np.array([0.5, 0.5], dtype=np.float32)  # center of the grid
    for instr, name in [(0, "RED  (corner 0,0)"), (1, "BLUE (corner 4,4)")]:
        with torch.no_grad():
            logits = policy(torch.as_tensor(mid).unsqueeze(0), torch.as_tensor([instr]))
        a = int(logits.argmax(-1).item())
        arrow = {0: "↑ up", 1: "↓ down", 2: "← left", 3: "→ right"}[a]
        print(f"  instruction={name:18s} -> action = {arrow}")


def main():
    torch.manual_seed(0)
    print("=== Tiny VLA mini-demo: instruction-conditioned policy ===\n")
    print("Stage 1: Behavioral Cloning from a scripted expert (the VLA recipe)")
    policy = ConditionedPolicy()
    train_bc(policy)

    acc = evaluate(policy)
    print(f"\nSuccess rate following instructions: {acc:.1%}")
    demonstrate_conditioning(policy)

    print("\nTakeaway: one conditioned policy maps (perception + instruction) -> action.")
    print("Swap the grid for camera images and the embedding for a text encoder,")
    print("scale the data, add RL fine-tuning -> that is a real VLA.")


if __name__ == "__main__":
    main()
