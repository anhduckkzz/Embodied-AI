# MuJoCo setup & first steps 💻 (great on your RTX 4060)

MuJoCo is the easiest high-quality robot simulator to start with. Pure pip, no
GPU required for the basics.

## Install
```bash
pip install mujoco                     # simulator + python bindings
pip install "gymnasium[mujoco]"        # standard RL robot tasks
# optional, DeepMind's control suite + nicer envs:
pip install dm_control
```

## Verify
```bash
python curriculum/part4_simulation/mujoco_demo.py        # PID-controls a pendulum
VIEWER=1 python curriculum/part4_simulation/mujoco_demo.py   # interactive 3D window
```

## The mental model
- A robot is an **MJCF** XML: `<body>` tree, `<joint>`s, `<geom>`s (collision/
  visual), `<actuator>`s (motors), `<sensor>`s. (See the string in `mujoco_demo.py`.)
- Code loop: `model = MjModel.from_xml_*`, `data = MjData(model)`, then each step
  `data.ctrl[:] = command; mujoco.mj_step(model, data)` and read `data.qpos`,
  `data.qvel`, sensors.

## What to do next
1. **Train RL on real physics:** your Part-1 agents work directly on MuJoCo Gym
   envs:
   ```bash
   python -m rl.train --algo sac --env Ant-v5 --continuous --steps 1000000
   python -m rl.train --algo ppo --env Humanoid-v5 --steps 5000000   # 🅰️ big
   ```
2. **Load a real robot:** grab MJCF models from **MuJoCo Menagerie**
   (github.com/google-deepmind/mujoco_menagerie) — Franka arm, Unitree Go2,
   Shadow Hand, etc.
3. **Scale up (optional):** **MJX** runs MuJoCo on GPU/TPU via JAX for parallel RL.

## Tips
- Headless render to images with `mujoco.Renderer` (for datasets / videos).
- `model.opt.timestep` is your `dt`; smaller = more accurate, slower.
- Contact unstable? Increase solver iterations, reduce timestep, check geom sizes.
