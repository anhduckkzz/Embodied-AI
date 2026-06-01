# Part 17 — Legged Locomotion & Humanoids

> The hardest classical-dynamics problem and the hottest robotics frontier
> (Unitree, Boston Dynamics, Figure, Tesla Optimus, NVIDIA GR00T). It's where
> **massively-parallel RL** (Part 1/4), **control** (Part 14), **estimation**
> (Part 7), and **sim-to-real** (Module 15) all converge on a real body.

## 1. Why legs are hard

- **Underactuated & hybrid dynamics:** the robot is only "connected" to the world
  through intermittent **foot contacts**; dynamics switch as feet land/lift
  (hybrid system). Contact is exactly what's hard to simulate (Part 2/4).
- **Unstable & dynamic balance:** unlike a wheeled robot, a biped can fall any
  instant; balance is *active*, every millisecond.
- **High DOF:** a humanoid has 20–40+ joints — huge action/state spaces.

## 2. Classical approach: model-based locomotion

- **Zero-Moment Point (ZMP):** keep the point where ground reaction forces net to
  zero inside the support polygon → static/quasi-static stability (classic
  bipedal walking, ASIMO-era).
- **Centroidal dynamics + MPC:** model the robot's center-of-mass momentum and
  solve **Model Predictive Control** (Part 14) over footstep plans and contact
  forces in real time (MIT Cheetah, ANYmal classical controllers).
- **Whole-body control (WBC):** instantaneously map task objectives (CoM, swing
  foot, posture) to joint torques via optimization, respecting contacts/limits.
- Reliable and interpretable; demands a good model and heavy real-time optimization.

## 3. Learned approach: RL locomotion (now state of the art)

The modern recipe that put robust walking on real quadrupeds/humanoids:
1. **Massively parallel sim:** thousands of robots in **Isaac Lab** (Part 4) step
   on the GPU; **PPO** (Part 1, Module 04) trains a policy in *minutes–hours*.
2. **Domain randomization** (Module 15): randomize mass, friction, terrain, motor
   delay, pushes → a **robust** policy.
3. **Sim-to-real deployment** with online **adaptation** — **RMA (Rapid Motor
   Adaptation)** infers terrain/dynamics from recent history and adapts in
   milliseconds; **teacher–student** distillation trains a deployable policy that
   only uses onboard sensors.
4. Result: blind quadrupeds traversing wild terrain (ETH ANYmal, Unitree), and
   increasingly dynamic humanoid walking/running — all learned.

Reward design (Module 14/15) is the craft: track commanded velocity + stay
upright + smoothness/energy penalties + foot-clearance + don't-fall.

## 4. Humanoids specifically
- **Why humanoids:** built for *our* world (stairs, doors, tools) and for learning
  from human data (teleop, video).
- **Locomotion + manipulation (loco-manipulation):** walk *and* use arms/hands —
  whole-body coordination (Part 9 + this part).
- **Foundation models:** **NVIDIA GR00T** and others aim at general humanoid
  policies trained on sim + human video + teleop — a **humanoid VLA** (Parts 10/16).
  This is the convergence point of the entire curriculum.

## 5. The full stack on a legged robot
```
 proprioception+IMU (Part 5) ─► state estimation (Part 7) ─┐
 commanded velocity / goal ─────────────────────────────►  RL policy or MPC (Parts 1/14)
 terrain perception (Parts 6/15, optional) ────────────────┘        │
                                                       joint targets/torques ─► actuators (Part 2)
```

## 6. On your hardware
- 💻 **RTX 4060:** train quadruped locomotion in Isaac Lab at modest `num_envs`,
  or use lighter sims (MuJoCo `Ant`/`Humanoid`, **gym** + your Part-1 SAC/PPO);
  run pretrained policies.
- 🅰️ **A100 / cloud:** large `num_envs` parallel RL, humanoid policies, GR00T-scale
  experiments.

## 🛠️ Project
1. 💻 Train **PPO/SAC** (Part 1) on MuJoCo **`Ant-v5`** then **`Humanoid-v5`** —
   your first learned locomotion. Add an **energy penalty** and see the gait change.
2. 🅰️ In **Isaac Lab**, train a **quadruped velocity-tracking** policy with domain
   randomization (Module 15); study the sim-to-real recipe.
3. Read the ANYmal (Hwangbo/Lee) and **RMA** papers; map each to Parts 1, 4, 7, 15.

## ✅ Check your understanding
1. Why are legged robots a *hybrid, underactuated* control problem?
2. What does the ZMP criterion ensure, and what's its limitation?
3. Describe the modern RL locomotion recipe (sim → randomize → adapt → deploy).
4. What does RMA add at deployment that a fixed policy lacks?
5. Why are humanoids the natural endpoint for VLA/foundation models?

## 📖 Go deeper
- Hwangbo et al. (2019) *Learning agile locomotion* (ANYmal); Lee et al. (2020)
  *Learning quadrupedal locomotion over challenging terrain* (Science Robotics);
  Kumar et al. (2021) **RMA**; Radosavovic et al. (2024) humanoid RL.
- Isaac Lab locomotion examples; MIT Underactuated Robotics (Tedrake); NVIDIA GR00T.

🎉 **End of the curriculum.** From the Bellman equation (Part 1) to humanoid
foundation models, you've covered the full embodied-AI stack — decision-making,
physics, math, simulation, ROS, sensing, perception, 3D vision, estimation,
planning, control, manipulation, imitation, foundation models, VLA, and
applications. Now pick a capstone (Part 11) and **build**.
