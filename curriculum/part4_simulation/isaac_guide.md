# Isaac Sim & Isaac Lab setup 🅰️ (NVIDIA RTX required)

NVIDIA's photorealistic simulator (**Isaac Sim**) and its GPU-parallel RL
framework (**Isaac Lab**, formerly Isaac Gym / OmniIsaacGymEnvs). Use these when
you need **realistic sensors** (RGB/depth/LiDAR) or **massively parallel RL**.

> Your **RTX 4060 (8 GB)** can run Isaac Sim for perception and modest RL.
> For thousands of parallel envs or big scenes, use a cloud **A100/L4**.

## Requirements
- NVIDIA RTX GPU + recent driver (RTX 30/40 series ideal), Linux or Windows.
- ~30+ GB disk. CUDA-capable. (8 GB VRAM works; 16 GB+ is comfortable.)

## Install (current recommended path: pip)
```bash
# In a fresh conda/venv with Python 3.10:
pip install isaacsim --extra-index-url https://pypi.nvidia.com   # Isaac Sim
# Then Isaac Lab (clone + install):
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install
```
(NVIDIA also ships an Omniverse-launcher/container route; check the current Isaac
Lab docs — the install path moves between releases.)

## Verify
```bash
# List built-in tasks and train a quadruped with thousands of parallel envs:
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4096 --headless
```
On a good GPU a locomotion policy trains in **minutes** — that speed is the whole
point of GPU-parallel sim.

## When to use what
- **Isaac Sim (alone):** generate photorealistic **perception datasets**
  (Part 6), test cameras/LiDAR, build digital twins, domain randomization.
- **Isaac Lab:** large-scale **RL** (locomotion, manipulation) with PPO/SAC at
  thousands of envs; export the policy, then sim-to-real with domain randomization.

## Workflow that matches this curriculum
1. Prototype the algorithm on MuJoCo/Gym (cheap, Part 1 + Part 4).
2. Move to Isaac Lab for scale + realistic sensors.
3. Add domain randomization → deploy → fine-tune on hardware (sim-to-real).

## Gotchas
- First launch compiles shaders — be patient.
- Headless (`--headless`) for training; GUI only to inspect.
- Match the Isaac Lab version to its required Isaac Sim version exactly.
- 8 GB VRAM: lower `--num_envs`, image resolution, and scene complexity.
