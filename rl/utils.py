"""Small, dependency-light helpers shared across every algorithm.

Nothing here is RL-specific magic: seeding, exploration schedules, a running
mean/std for observation normalisation, a minimal metrics logger, and a
plotting helper. Keeping these in one place means each agent file stays focused
on the *algorithm* rather than boilerplate.
"""
from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:  # torch is optional for the pure-numpy tabular lessons
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch always present in practice
    _HAS_TORCH = False


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if available) PyTorch for reproducibility.

    ``deterministic=True`` also forces deterministic cuDNN kernels, which is
    slower but makes runs bit-for-bit comparable - handy when debugging.
    """
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "auto"):
    """Return the best available torch device.

    ``prefer`` may be "auto", "cpu", "cuda", or "mps" (Apple Silicon).
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for get_device().")
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LinearSchedule:
    """Linearly interpolate a value from ``start`` to ``end`` over ``duration`` steps.

    Used for epsilon-greedy exploration and learning-rate annealing. After
    ``duration`` steps the value is clamped at ``end``.
    """

    def __init__(self, start: float, end: float, duration: int):
        self.start = start
        self.end = end
        self.duration = max(1, duration)

    def value(self, step: int) -> float:
        frac = min(float(step) / self.duration, 1.0)
        return self.start + frac * (self.end - self.start)


class ExponentialSchedule:
    """Multiplicative decay, ``v <- max(end, v * decay)`` once per call.

    This mirrors the classic DQN epsilon schedule used in the original repo.
    """

    def __init__(self, start: float, end: float, decay: float):
        self.value = start
        self.end = end
        self.decay = decay

    def step(self) -> float:
        self.value = max(self.end, self.value * self.decay)
        return self.value


class RunningMeanStd:
    """Online (Welford-style) mean/variance for observation normalisation.

    Normalising observations is one of the highest-leverage tricks for getting
    policy-gradient methods (PPO, SAC) to train stably on continuous control.
    """

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


@dataclass
class Logger:
    """Append-only metrics logger: prints a tidy line and writes a CSV.

    Deliberately tiny so the curriculum has zero hidden dependencies. For
    serious experiments swap this for TensorBoard / Weights & Biases - the
    agents only call ``log()`` so the rest of the code is unaffected.
    """

    run_dir: str
    print_every: int = 1
    _rows: List[Dict] = field(default_factory=list)
    _t0: float = field(default_factory=time.time)
    _keys: Optional[List[str]] = None

    def __post_init__(self):
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, "metrics.csv")

    def log(self, step: int, metrics: Dict[str, float], force_print: bool = False) -> None:
        row = {"step": step, "elapsed_s": round(time.time() - self._t0, 1)}
        row.update({k: float(v) for k, v in metrics.items()})
        self._rows.append(row)
        if self._keys is None:
            self._keys = list(row.keys())
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._keys).writeheader()
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._keys).writerow(
                {k: row.get(k, "") for k in self._keys}
            )
        if force_print or len(self._rows) % self.print_every == 0:
            pretty = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items())
            print(f"[step {step:>8}] {pretty}")

    def rows(self) -> List[Dict]:
        return self._rows


def moving_average(values: List[float], window: int = 100) -> np.ndarray:
    """Trailing moving average used for the classic 'score over episodes' plot."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_scores(scores: List[float], window: int = 100, title: str = "Training",
                save_path: Optional[str] = None, show: bool = False):
    """Plot raw episode scores plus a moving average. Saves a PNG if requested."""
    import matplotlib

    if save_path and not show:
        matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(scores, alpha=0.35, label="episode score")
    if len(scores) >= 2:
        ma = moving_average(scores, window)
        ax.plot(range(len(scores) - len(ma), len(scores)), ma,
                color="crimson", lw=2, label=f"{window}-episode average")
    ax.set_xlabel("episode")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path
