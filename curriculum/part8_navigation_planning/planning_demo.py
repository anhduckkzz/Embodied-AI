"""Path planning, runnable: A* / Dijkstra on a grid, and RRT in continuous space.

Run:  python curriculum/part8_navigation_planning/planning_demo.py
      (saves planning_demo.png if matplotlib is available)

Shows the two great families side by side (both from robotics/planning.py):
  * Grid search (A* vs Dijkstra) on an occupancy grid with walls.
  * Sampling-based (RRT) around circular obstacles in a continuous space.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robotics.planning import RRT, astar, dijkstra


def build_grid():
    grid = np.zeros((20, 20))
    grid[5:15, 7] = 1          # wall with a gap at the bottom
    grid[5, 7:14] = 1          # an L-shaped barrier
    grid[5:15, 7][-1] = 0      # leave a gap to force a detour
    return grid


def grid_demo():
    print("1) Grid search: A* vs Dijkstra")
    grid = build_grid()
    start, goal = (0, 0), (19, 19)
    path_a = astar(grid, start, goal, heuristic="euclidean")
    path_d = dijkstra(grid, start, goal)
    print(f"   A*       path length: {len(path_a) if path_a else 'none'} cells")
    print(f"   Dijkstra path length: {len(path_d) if path_d else 'none'} cells")
    print("   (Both optimal; A* explores far fewer cells thanks to the heuristic.)")
    return grid, path_a


def rrt_demo():
    print("2) Sampling-based: RRT around circular obstacles")
    obstacles = [(5, 5, 1.5), (7, 3, 1.0), (3, 7, 1.0), (6, 8, 1.0)]
    rrt = RRT(bounds=((0, 10), (0, 10)), obstacles=obstacles, step=0.4,
              goal_sample_rate=0.1, seed=3)
    path = rrt.plan((0.5, 0.5), (9.5, 9.5), max_iters=8000)
    if path:
        length = sum(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
                     for i in range(len(path) - 1))
        print(f"   RRT found a path with {len(path)} waypoints, length {length:.1f}")
    else:
        print("   RRT failed (try more iterations)")
    return obstacles, path


def main():
    print("=== Path planning: grid search & sampling-based ===\n")
    grid, path_a = grid_demo()
    obstacles, path_rrt = rrt_demo()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(grid.T, origin="lower", cmap="Greys")
        if path_a:
            p = np.array(path_a)
            ax[0].plot(p[:, 0], p[:, 1], "b-", lw=2)
            ax[0].scatter([0, 19], [0, 19], c=["green", "red"], s=80)
        ax[0].set_title("A* on occupancy grid")
        for cx, cy, r in obstacles:
            ax[1].add_patch(plt.Circle((cx, cy), r, color="grey"))
        if path_rrt:
            p = np.array(path_rrt)
            ax[1].plot(p[:, 0], p[:, 1], "b-o", ms=2)
            ax[1].scatter([0.5, 9.5], [0.5, 9.5], c=["green", "red"], s=80)
        ax[1].set_xlim(0, 10); ax[1].set_ylim(0, 10); ax[1].set_aspect("equal")
        ax[1].set_title("RRT in continuous space")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planning_demo.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"\nSaved -> {out}")
    except ImportError:
        pass
    print("\nGrid search = optimal in low-D; sampling = scalable to high-D (arms, drones).")


if __name__ == "__main__":
    main()
