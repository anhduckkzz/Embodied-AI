"""A multi-task, language-conditioned environment for the end-to-end VLA.

This is a stage-3 "real system" component, not a from-scratch teaching toy. The
observation is an actual rendered RGB image plus a tokenized language instruction
plus proprioception (the agent position), which is exactly the input signature of
a real vision-language-action policy.

It is deliberately multi-task. One policy must handle two different families of
instruction in the same world:

  1. Object grounding ("go to the red"): navigate to a colored square. The agent
     must read the instruction to pick the color and read the image to find where
     that color is. Vision is required.
  2. Spatial grounding ("go to the center"): navigate to a named region of the
     grid. The agent must map a spatial word to a location and use its own
     position. Proprioception and language are required.

Success on both with a single network is what makes this a genuine multi-task VLA
rather than a single-task demo.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

GRID = 5
CELL = 8
IMG = GRID * CELL          # rendered image is IMG x IMG pixels

# Object colors (RGB in [0, 1]).
PALETTE = {
    "red": (1.0, 0.15, 0.15),
    "green": (0.15, 0.8, 0.2),
    "blue": (0.2, 0.35, 1.0),
    "yellow": (1.0, 0.85, 0.1),
}
COLOR_NAMES: List[str] = list(PALETTE)

# Named spatial regions, as (row, col) grid cells.
REGIONS = {
    "topleft": (0, 0),
    "topright": (0, GRID - 1),
    "bottomleft": (GRID - 1, 0),
    "bottomright": (GRID - 1, GRID - 1),
    "center": (GRID // 2, GRID // 2),
}
REGION_NAMES: List[str] = list(REGIONS)

# A small fixed vocabulary for the instruction language.
WORDS = ["<pad>", "go", "to", "the"] + COLOR_NAMES + REGION_NAMES
STOI = {w: i for i, w in enumerate(WORDS)}
VOCAB_SIZE = len(WORDS)
MAX_INSTR_LEN = 4          # "go to the <target>"

# Actions: four moves plus an interact action that ends the episode.
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}   # up, down, left, right
INTERACT = 4
N_ACTIONS = 5


def encode_instruction(words: List[str]) -> np.ndarray:
    ids = [STOI[w] for w in words][:MAX_INSTR_LEN]
    ids = ids + [STOI["<pad>"]] * (MAX_INSTR_LEN - len(ids))
    return np.array(ids, dtype=np.int64)


class MultiTaskVLAEnv:
    """Grid world with image observations and language-specified goals."""

    def __init__(self, seed: int = 0, n_objects: int = 4):
        self.rng = np.random.default_rng(seed)
        self.n_objects = min(n_objects, len(COLOR_NAMES))

    # ---- episode setup -----------------------------------------------------
    def reset(self, task_type: Optional[str] = None, target: Optional[str] = None) -> Dict:
        """Start a new episode. ``task_type`` is "color" or "spatial" (random if
        None); ``target`` optionally fixes the color or region."""
        cells = self._distinct_cells(self.n_objects + 1)
        self.object_cells = {COLOR_NAMES[i]: cells[i] for i in range(self.n_objects)}
        self.agent = list(cells[self.n_objects])
        self.task_type = task_type or self.rng.choice(["color", "spatial"])
        self._set_goal(target)
        self.done = False
        return self._obs()

    def set_task(self, task_type: str, target: str) -> Dict:
        """Keep the current scene but change the instruction. Used to show that the
        same image yields different behavior under different instructions."""
        self.task_type = task_type
        self._set_goal(target)
        self.done = False
        return self._obs()

    def _set_goal(self, target: Optional[str]) -> None:
        if self.task_type == "color":
            self.target = target or COLOR_NAMES[int(self.rng.integers(self.n_objects))]
            self.goal_cell = list(self.object_cells[self.target])
        else:
            self.target = target or REGION_NAMES[int(self.rng.integers(len(REGION_NAMES)))]
            self.goal_cell = list(REGIONS[self.target])
        self.instruction_words = ["go", "to", "the", self.target]
        self.instruction = encode_instruction(self.instruction_words)

    def _distinct_cells(self, n: int) -> List[List[int]]:
        all_cells = [(r, c) for r in range(GRID) for c in range(GRID)]
        idx = self.rng.choice(len(all_cells), size=n, replace=False)
        return [list(all_cells[i]) for i in idx]

    # ---- rendering and observation ----------------------------------------
    def _render(self) -> np.ndarray:
        """Return a (3, IMG, IMG) float32 image of the current scene."""
        img = np.full((IMG, IMG, 3), 0.5, dtype=np.float32)   # gray background
        for name, (r, c) in self.object_cells.items():
            img[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = PALETTE[name]
        # Draw the agent as a white border ring so any object underneath stays
        # visible in the cell interior (the model can still see its color).
        r, c = self.agent
        y0, y1, x0, x1 = r * CELL, (r + 1) * CELL, c * CELL, (c + 1) * CELL
        img[y0:y1, x0] = 1.0
        img[y0:y1, x1 - 1] = 1.0
        img[y0, x0:x1] = 1.0
        img[y1 - 1, x0:x1] = 1.0
        return np.transpose(img, (2, 0, 1)).copy()

    def _obs(self) -> Dict:
        proprio = np.array([self.agent[0] / (GRID - 1), self.agent[1] / (GRID - 1)],
                           dtype=np.float32)
        return {"image": self._render(),
                "instruction": self.instruction.copy(),
                "proprio": proprio}

    # ---- dynamics ----------------------------------------------------------
    def step(self, action: int):
        if action == INTERACT:
            success = list(self.agent) == list(self.goal_cell)
            self.done = True
            return self._obs(), float(success), True, {"success": bool(success)}
        dr, dc = MOVES[int(action)]
        self.agent[0] = int(min(max(self.agent[0] + dr, 0), GRID - 1))
        self.agent[1] = int(min(max(self.agent[1] + dc, 0), GRID - 1))
        return self._obs(), 0.0, False, {"success": False}

    def expert_action(self) -> int:
        """A scripted oracle that always solves the current task (used for BC)."""
        r, c = self.agent
        gr, gc = self.goal_cell
        if [r, c] == [gr, gc]:
            return INTERACT
        if r < gr:
            return 1   # down
        if r > gr:
            return 0   # up
        if c < gc:
            return 3   # right
        return 2       # left
