# Part 8 — Navigation & Planning: deciding where and how to move

> Code: [`robotics/planning.py`](../../robotics/planning.py), demo:
> [`planning_demo.py`](planning_demo.py). With a pose and a map (Part 7) and an
> understanding of obstacles (Part 6), the robot must **decide where to go and
> compute a safe path to get there.** This is the "act" side of autonomy.

The classic decomposition (used by ROS Nav2 and most autonomy stacks):

```
GLOBAL PLAN (where to go, roughly)  →  LOCAL PLAN (how to move now, safely)  →  CONTROL (Part 9)
   A* / Dijkstra on a map               dynamic-window / MPC, avoid obstacles      track the path
```

---

## 1. Configuration space (C-space) — the key abstraction

Plan not in the world, but in the space of robot **configurations**. For a mobile
robot that's `(x, y, θ)`; for a 7-DOF arm it's 7 joint angles. Obstacles in the
world become forbidden regions in C-space. Planning = finding a collision-free
*curve* from start to goal config. This reframing makes arms and cars the same
problem at different dimensions.

## 2. Global planning on a grid: Dijkstra & A*

Represent free space as an **occupancy grid** (0 = free, 1 = obstacle, from
perception/SLAM). Find the shortest path with graph search:

- **Dijkstra** — expands uniformly outward; finds the optimal path but explores a
  lot.
- **A\*** — Dijkstra + an **admissible heuristic** `h` (e.g. straight-line
  distance to goal) that focuses the search toward the goal. Optimal *and* fast.
  The heuristic must never overestimate, or you lose the optimality guarantee.

```python
from robotics.planning import astar
path = astar(occupancy_grid, start=(0, 0), goal=(9, 9), heuristic="euclidean")
```

Both are in [`robotics/planning.py`](../../robotics/planning.py) (A* with a
`"zero"` heuristic *is* Dijkstra). Variants you'll meet: **D\* / D\* Lite**
(replan efficiently when the map changes), **weighted A\*** (faster, slightly
suboptimal), **hybrid A\*** (respects car kinematics — used in parking).

## 3. Sampling-based planning: RRT (and friends)

A grid is hopeless in high dimensions (a 7-DOF arm grid is astronomically large).
**Sampling-based** planners instead randomly sample configurations and connect
collision-free ones:

- **RRT** (Rapidly-exploring Random Tree) — grow a tree from the start toward
  random samples; biased occasionally toward the goal. Fast, **probabilistically
  complete**, but paths are jagged and non-optimal.
- **RRT\*** — asymptotically **optimal** (rewires the tree); **PRM** —
  precompute a roadmap for repeated queries in a static map.

```python
from robotics.planning import RRT
rrt = RRT(bounds=((0,10),(0,10)), obstacles=[(5,5,1.5)], step=0.5)
path = rrt.plan(start=(1,1), goal=(9,9))
```

This is what **MoveIt** uses to plan arm motions (Part 9) and what plans drone
paths in 3D.

## 4. Local planning & obstacle avoidance (reactive)

The global path is computed on a static map, but the world moves (pedestrians!).
The **local planner** repeatedly adjusts the immediate motion at high rate:

- **Dynamic Window Approach (DWA)** — sample feasible `(v, ω)` commands, simulate
  short rollouts, score by progress + clearance + smoothness, pick the best.
- **Timed Elastic Band (TEB)** — deform the path to avoid obstacles while
  respecting dynamics.
- **Model Predictive Control (MPC)** — optimize a control sequence over a horizon
  subject to dynamics + constraints; the modern, powerful choice (also Part 9).
- **Potential fields** — attract to goal, repel from obstacles (simple, can get
  stuck in local minima).

Costmaps (from perception) inflate obstacles by the robot's radius so the planner
keeps a safe margin.

## 5. Motion planning for cars & drones (kinodynamic)

Real vehicles can't move sideways — planning must respect **kinematic
constraints** (a car's turning radius, **Dubins/Reeds-Shepp** paths) and
**dynamics** (momentum, acceleration limits). "Kinodynamic" planners plan in
state×time. For drones, paths are smooth **minimum-snap trajectories** in 3D that
respect thrust/attitude limits.

## 6. Decision-making — the layer above paths

Planning answers "how do I get there"; **decision-making** answers "what should I
do":
- **Finite State Machines / Behavior Trees** — structured, interpretable logic
  ("if pedestrian → stop"); the industry default for behavior orchestration.
- **Rule-based + cost-based** — hand-designed policies with safety guarantees.
- **Learned policies** — RL (Part 1) or imitation (Part 9) for hard-to-script
  behaviors; increasingly, **VLA models** (Part 10) fold perception→decision→
  action into one network.
- **Prediction** — anticipate other agents' futures (a pedestrian's path) before
  deciding — essential for driving (Part 11).

## 7. The full navigation stack (e.g., ROS 2 Nav2)
```
map (SLAM, Part 7) + pose ─► global planner (A*/Dijkstra) ─► global path
                                       │
sensors ─► costmap (perception) ─► local planner (DWA/MPC) ─► velocity cmd ─► control (Part 9)
                                       └────────── recovery behaviors (stuck? back up, spin) ──────────┘
```

## 🛠️ Project
Run [`planning_demo.py`](planning_demo.py): it builds a grid world with walls,
plans with **A\*** and **Dijkstra** (compare nodes expanded), then plans a
continuous path with **RRT** around circular obstacles — saving a visualization.
Then: add a moving obstacle and re-plan each step (a poor-man's local planner).

## ✅ Check your understanding
1. What is configuration space, and why plan in it?
2. A* vs Dijkstra — what does the heuristic buy you, and what must it satisfy?
3. Why use RRT instead of a grid for a 7-DOF arm?
4. What does a local planner do that the global planner can't?
5. Where does decision-making sit relative to planning, and what tools implement it?

## 📖 Go deeper
- LaValle, *Planning Algorithms* (free book) — the comprehensive reference.
- Lynch & Park, *Modern Robotics*, Ch. 10 (motion planning).
- ROS 2 **Nav2** docs; **MoveIt 2** tutorials; behavior-tree (BT.CPP) docs.

➡️ **Next:** [Part 9 — Manipulation & Teleoperation](../part9_manipulation_teleop/):
controlling arms, grasping, and collecting the demonstrations that train VLAs.
