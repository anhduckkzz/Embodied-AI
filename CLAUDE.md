# Repository guide for AI assistants

This repo is an **Embodied AI: Zero → Master** curriculum. It grows from a basic
LunarLander DQN to reinforcement learning, classical robotics, perception, SLAM,
navigation, manipulation, Vision-Language-Action (VLA) models, and applications
(autonomous driving, drones, humanoids).

## Structure
- `rl/` — importable RL library. Agents in `rl/agents/`, shared parts in
  `networks.py`, `buffers.py`, `envs.py`, `utils.py`. Unified trainer in `rl/train.py`.
- `robotics/` — importable NumPy robotics library: `transforms.py` (SO3/SE3/quat),
  `kinematics.py` (DH/FK/IK/Jacobian), `control.py` (PID), `filters.py` (KF/EKF),
  `planning.py` (A*/Dijkstra/RRT), `pointcloud.py` (voxel/RANSAC/ICP), `sensors.py`.
- `curriculum/partN_*/` — numbered lessons (11 parts). Each part has a `README.md`
  (theory) and many have runnable `*_demo.py` scripts and/or `notebook.ipynb`.
  Part 4 also has setup guides (MuJoCo/Isaac/ROS) as `*_guide.md`.
- `scripts/` — `play.py` / `record.py` entry points.
- `tests/` — `pytest` sanity checks (`test_core.py` for rl, `test_robotics.py`).
- `tools/build_notebooks.py` — regenerates the Part-1 Colab notebooks.

## Conventions
- Keep each module readable and aligned with its lesson's maths; comments cite the
  update rules / equations. Configs are `@dataclass`es.
- Standardize on **Gymnasium** (not old `gym`) and **LunarLander-v3**.
- Libraries are NumPy/PyTorch and dependency-light by design; heavy tools
  (MuJoCo, Isaac, ROS, Open3D, transformers) are optional extras or setup guides.
- New demos must be runnable from the repo root and degrade gracefully if an
  optional dependency (e.g. matplotlib) is missing.
- If you change Part-1 lesson code examples, regenerate notebooks:
  `python tools/build_notebooks.py`.

## Useful commands
```bash
pip install -e ".[box2d,dev]"          # install packages + deps
pytest -q                              # run all sanity tests (rl + robotics)
python -m rl.train --algo ppo --env LunarLander-v3 --save runs/ppo.pt
python curriculum/part6_perception/perception_demo.py     # a robotics demo
python curriculum/part4_simulation/mujoco_demo.py         # needs `pip install mujoco`
```

## Don't
- Don't reintroduce the old `gym` API or `v2` env ids.
- Don't add heavy deps to core `requirements.txt`; gate them behind extras.
- Don't break the runnable demos' "no heavy deps / graceful fallback" property.
