"""Tests for the end-to-end multi-task VLA application (applications/vla).

These verify the environment and expert are correct, the model wires together
with the right shapes, and behavioral cloning actually learns (the fast spatial
task is used as the learning check; full object-grounding success is reached by
running `python -m applications.vla.train`, which needs more epochs than a unit
test should run)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from applications.vla.env import (GRID, IMG, MAX_INSTR_LEN, N_ACTIONS, VOCAB_SIZE,  # noqa: E402
                                 MultiTaskVLAEnv)
from applications.vla.model import VLAPolicy  # noqa: E402


def test_expert_solves_both_task_families():
    env = MultiTaskVLAEnv(seed=2)
    for family in ("color", "spatial"):
        successes = 0
        for _ in range(200):
            obs = env.reset(task_type=family)
            for _ in range(2 * GRID + 1):
                obs, _, done, info = env.step(env.expert_action())
                if done:
                    successes += info["success"]
                    break
        assert successes == 200, f"expert failed on {family}: {successes}/200"


def test_observation_structure():
    env = MultiTaskVLAEnv(seed=0)
    obs = env.reset()
    assert obs["image"].shape == (3, IMG, IMG)
    assert obs["image"].dtype == np.float32
    assert obs["instruction"].shape == (MAX_INSTR_LEN,)
    assert obs["proprio"].shape == (2,)


def test_model_forward_shape():
    model = VLAPolicy(IMG, VOCAB_SIZE, MAX_INSTR_LEN, N_ACTIONS, dim=64)
    image = torch.randn(5, 3, IMG, IMG)
    instr = torch.randint(0, VOCAB_SIZE, (5, MAX_INSTR_LEN))
    proprio = torch.rand(5, 2)
    logits = model(image, instr, proprio)
    assert logits.shape == (5, N_ACTIONS)


def test_set_task_changes_instruction_same_scene():
    env = MultiTaskVLAEnv(seed=3)
    env.reset(task_type="color")
    img1 = env._render()
    obs = env.set_task("color", "blue")
    img2 = env._render()
    # Same scene (objects unchanged), only the instruction/goal changed.
    assert np.array_equal(img1, img2)
    assert env.target == "blue"


def test_bc_learns_spatial_task_quickly():
    """A short BC run should solve the fast spatial-grounding family and clearly
    reduce the imitation loss. This confirms the full pipeline learns."""
    from applications.vla.train import collect_dataset, evaluate, train
    torch.manual_seed(0)
    data = collect_dataset(n_episodes=500, seed=0)
    model = VLAPolicy(IMG, VOCAB_SIZE, MAX_INSTR_LEN, N_ACTIONS, dim=64)
    history = train(model, data, epochs=8, lr=5e-4, verbose=False)
    assert history[-1] < 0.7 * history[0]            # imitation loss fell clearly
    assert evaluate(model, "spatial", n_episodes=100) > 0.8
