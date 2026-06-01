"""Path & motion planning: choosing *where* to move.

Given a map and a goal, find a collision-free path. Two complementary families:

* **Graph search** on a grid - **Dijkstra** (uniform cost) and **A-star** (adds an
  admissible heuristic for speed). Optimal, complete, great for 2D nav meshes.
* **Sampling-based** - **RRT** (Rapidly-exploring Random Tree) - scales to
  high-dimensional configuration spaces (robot arms, drones in 3D) where a grid
  is hopeless. Probabilistically complete, not optimal (see RRT* for that).

These are the algorithms inside ROS Nav2, MoveIt, and most autonomy stacks.
"""
from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

import numpy as np

Cell = Tuple[int, int]


# --------------------------------------------------------------------------
# Grid search: Dijkstra & A*
# --------------------------------------------------------------------------
def _neighbors(cell: Cell, grid: np.ndarray, diagonal: bool = True):
    r, c = cell
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in steps:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
            yield (nr, nc), np.hypot(dr, dc)


def astar(grid: np.ndarray, start: Cell, goal: Cell,
          heuristic: str = "euclidean", diagonal: bool = True) -> Optional[List[Cell]]:
    """A* shortest path on an occupancy grid (0 = free, 1 = obstacle).

    Set ``heuristic='zero'`` to recover Dijkstra (no heuristic). The heuristic
    must never *overestimate* the true cost (admissibility) to keep A* optimal.
    """
    def h(a: Cell) -> float:
        if heuristic == "zero":
            return 0.0
        dr, dc = abs(a[0] - goal[0]), abs(a[1] - goal[1])
        if heuristic == "manhattan":
            return dr + dc
        return np.hypot(dr, dc)  # euclidean

    open_heap = [(h(start), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()
    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct(came_from, current)
        if current in visited:
            continue
        visited.add(current)
        for nbr, cost in _neighbors(current, grid, diagonal):
            tentative = g + cost
            if tentative < g_score.get(nbr, np.inf):
                g_score[nbr] = tentative
                came_from[nbr] = current
                heapq.heappush(open_heap, (tentative + h(nbr), tentative, nbr))
    return None  # no path


def dijkstra(grid: np.ndarray, start: Cell, goal: Cell, diagonal: bool = True):
    """Dijkstra = A* with a zero heuristic (explores uniformly outward)."""
    return astar(grid, start, goal, heuristic="zero", diagonal=diagonal)


def _reconstruct(came_from: dict, current: Cell) -> List[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


# --------------------------------------------------------------------------
# Sampling-based: RRT
# --------------------------------------------------------------------------
class RRT:
    """Rapidly-exploring Random Tree in a 2D continuous space with circle obstacles.

    Grows a tree from the start by repeatedly sampling a random point, steering
    the nearest node toward it by ``step``, and keeping the new node if the edge
    is collision-free. Generalizes directly to N-D config spaces.
    """

    def __init__(self, bounds=((0, 10), (0, 10)), obstacles=None,
                 step: float = 0.5, goal_sample_rate: float = 0.05, seed: int = 0):
        self.bounds = bounds
        self.obstacles = obstacles or []   # list of (cx, cy, radius)
        self.step = step
        self.goal_sample_rate = goal_sample_rate
        self.rng = np.random.default_rng(seed)

    def _collision_free(self, p: np.ndarray) -> bool:
        return all(np.hypot(p[0] - cx, p[1] - cy) > r for cx, cy, r in self.obstacles)

    def _edge_free(self, a, b, n: int = 10) -> bool:
        return all(self._collision_free(a + (b - a) * t) for t in np.linspace(0, 1, n))

    def plan(self, start, goal, max_iters: int = 5000):
        """Return a list of waypoints from start to goal, or None on failure."""
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        nodes = [start]
        parents = {0: None}
        for _ in range(max_iters):
            if self.rng.random() < self.goal_sample_rate:
                sample = goal
            else:
                sample = np.array([self.rng.uniform(*self.bounds[0]),
                                   self.rng.uniform(*self.bounds[1])])
            dists = [np.linalg.norm(sample - n) for n in nodes]
            i_near = int(np.argmin(dists))
            near = nodes[i_near]
            direction = sample - near
            d = np.linalg.norm(direction)
            new = near + direction / d * min(self.step, d) if d > 1e-9 else near
            if not self._collision_free(new) or not self._edge_free(near, new):
                continue
            nodes.append(new)
            parents[len(nodes) - 1] = i_near
            if np.linalg.norm(new - goal) < self.step:
                return self._path(nodes, parents, len(nodes) - 1) + [goal]
        return None

    def _path(self, nodes, parents, idx):
        path = []
        while idx is not None:
            path.append(nodes[idx])
            idx = parents[idx]
        return path[::-1]
