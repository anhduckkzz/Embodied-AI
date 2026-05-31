"""Generate Colab-ready teaching notebooks for each curriculum module.

We build notebooks programmatically (raw nbformat-v4 JSON, no dependencies) so
they stay consistent and are trivial to regenerate when a lesson changes:

    python tools/build_notebooks.py

Each module gets a ``notebook.ipynb`` next to its README. Every notebook starts
with the same Colab setup cell (clone repo + install deps + pick GPU) so a
learner can "Open in Colab" and run top-to-bottom.
"""
from __future__ import annotations

import json
import os

REPO_URL = "https://github.com/anhduckkzz/lunarlander.git"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {},
            "source": _split("\n".join(lines))}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _split("\n".join(lines))}


def _split(text):
    # nbformat stores source as a list of lines each ending in "\n" (except last)
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


SETUP = code(
    "# === Colab setup: run me first ===",
    "import os, sys",
    "if not os.path.exists('rl'):",
    "    # On Colab, clone the repo so the `rl` package is importable.",
    f"    !git clone {REPO_URL} repo && (cp -r repo/* . 2>/dev/null || true)",
    "    !pip -q install 'gymnasium[box2d]>=0.29' torch numpy matplotlib imageio tqdm",
    "import torch",
    "print('Torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())",
    "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'",
)


def notebook(title, intro, cells):
    nb = {
        "cells": [md(f"# {title}", "", intro,
                     "", "> Tip: in Colab, set **Runtime → Change runtime type → GPU**."),
                  SETUP] + cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU", "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    return nb


# ---------------------------------------------------------------------------
# Per-module content
# ---------------------------------------------------------------------------
NOTEBOOKS = {}

NOTEBOOKS["01_tabular"] = notebook(
    "Module 01 — Tabular Q-learning & SARSA",
    "Watch a value function form from nothing but a NumPy table. "
    "Read [the lesson](README.md) alongside this notebook.",
    [
        md("## Q-learning on FrozenLake\n"
           "The whole algorithm is one update rule. Train, then inspect the learned table."),
        code("import gymnasium as gym",
             "from rl.agents.tabular import TabularAgent, train_tabular",
             "",
             "env = gym.make('FrozenLake-v1', is_slippery=False)",
             "agent = TabularAgent(env.observation_space.n, env.action_space.n,",
             "                     algo='q_learning', alpha=0.1, eps_decay=0.9995)",
             "scores = train_tabular(env, agent, n_episodes=8000, log_every=1000)"),
        md("### Inspect the learned value function"),
        code("import numpy as np",
             "V = agent.Q.max(axis=1).reshape(4, 4)   # FrozenLake is a 4x4 grid",
             "print('State values (max_a Q):')",
             "print(np.round(V, 2))"),
        md("## Now compare SARSA (on-policy) vs Q-learning on a slippery lake\n"
           "Turn on stochastic transitions and see how the safer, on-policy SARSA behaves."),
        code("import numpy as np",
             "from rl.agents.tabular import TabularAgent, train_tabular",
             "import gymnasium as gym",
             "for algo in ['q_learning', 'sarsa']:",
             "    env = gym.make('FrozenLake-v1', is_slippery=True)",
             "    ag = TabularAgent(env.observation_space.n, env.action_space.n, algo=algo, alpha=0.1)",
             "    sc = train_tabular(env, ag, n_episodes=20000, verbose=False)",
             "    print(f'{algo:10s} final avg success: {np.mean(sc[-1000:]):.3f}')"),
        md("**Exercise:** try `CliffWalking-v0`. Plot the path each agent prefers. "
           "Why does SARSA avoid the cliff edge?"),
    ],
)

NOTEBOOKS["02_value_based"] = notebook(
    "Module 02 — DQN on LunarLander",
    "Replace the table with a neural network and land the lunar module. "
    "See [the lesson](README.md).",
    [
        md("## Train DQN (Double DQN is on by default)"),
        code("from rl.envs import make_env, env_dims",
             "from rl.agents.dqn import DQN, DQNConfig",
             "from rl.train import train_offpolicy",
             "from rl.utils import ExponentialSchedule, Logger, plot_scores, set_seed",
             "",
             "set_seed(0)",
             "env = make_env('LunarLander-v3', seed=0)",
             "s_dim, a_dim, _ = env_dims(env)",
             "agent = DQN(s_dim, a_dim, DQNConfig(double=True), device=DEVICE)",
             "eps = ExponentialSchedule(1.0, 0.01, 0.9995)",
             "scores = train_offpolicy(agent, env, n_steps=200_000, eps_schedule=eps,",
             "                         logger=Logger('runs/dqn_nb'), solved_at=200)",
             "plot_scores(scores, title='DQN on LunarLander', show=True)"),
        md("## Save & watch it play (records a GIF you can display)"),
        code("agent.save('model/dqn_nb.pth')",
             "import imageio, numpy as np",
             "from rl.envs import make_env",
             "eval_env = make_env('LunarLander-v3', render_mode='rgb_array', seed=1)",
             "frames, state, done = [], eval_env.reset()[0], False",
             "while not done:",
             "    frames.append(eval_env.render())",
             "    state, r, term, trunc, _ = eval_env.step(agent.act(state, eps=0.0))",
             "    done = term or trunc",
             "imageio.mimsave('doc/dqn_nb.gif', frames, fps=30)",
             "print('saved doc/dqn_nb.gif —', len(frames), 'frames')"),
        md("**Ablation:** rerun with `DQNConfig(double=False)` and with `tau=1.0` "
           "(no target network). Compare the curves."),
    ],
)

NOTEBOOKS["03_policy_gradient"] = notebook(
    "Module 03 — REINFORCE",
    "Optimize the policy directly with Monte-Carlo policy gradients. "
    "See [the lesson](README.md).",
    [
        md("## REINFORCE with a learned baseline on CartPole"),
        code("from rl.envs import make_env, env_dims",
             "from rl.agents.reinforce import REINFORCE, ReinforceConfig",
             "from rl.train import train_episodic",
             "from rl.utils import Logger, plot_scores, set_seed",
             "",
             "set_seed(0)",
             "env = make_env('CartPole-v1', seed=0)",
             "s_dim, a_dim, _ = env_dims(env)",
             "agent = REINFORCE(s_dim, a_dim, discrete=True,",
             "                  cfg=ReinforceConfig(use_baseline=True), device=DEVICE)",
             "scores = train_episodic(agent, env, n_episodes=1000,",
             "                        logger=Logger('runs/reinforce_nb'), solved_at=195)",
             "plot_scores(scores, title='REINFORCE on CartPole', show=True)"),
        md("**Experiment:** set `use_baseline=False`. The learning curve should get "
           "noticeably noisier — that is the variance the baseline removes."),
    ],
)

NOTEBOOKS["04_actor_critic"] = notebook(
    "Module 04 — PPO (and A2C)",
    "The modern default, and the algorithm behind RLHF/VLA fine-tuning. "
    "See [the lesson](README.md).",
    [
        md("## Train PPO on LunarLander"),
        code("from rl.envs import make_env, env_dims",
             "from rl.agents.ppo import PPO, PPOConfig",
             "from rl.train import train_onpolicy",
             "from rl.utils import Logger, plot_scores, set_seed",
             "",
             "set_seed(0)",
             "env = make_env('LunarLander-v3', seed=0)",
             "s_dim, a_dim, _ = env_dims(env)",
             "agent = PPO(s_dim, a_dim, discrete=True,",
             "            cfg=PPOConfig(rollout_steps=2048, epochs=10), device=DEVICE)",
             "scores = train_onpolicy(agent, env, total_steps=400_000,",
             "                        logger=Logger('runs/ppo_nb'), solved_at=200)",
             "plot_scores(scores, title='PPO on LunarLander', show=True)",
             "agent.save('model/ppo_nb.pt')"),
        md("**Connect to VLA:** the clipped objective you just ran is, with a reward "
           "model swapped in for the env reward, exactly RLHF. Same code, different reward."),
    ],
)

NOTEBOOKS["05_continuous_control"] = notebook(
    "Module 05 — SAC on continuous control",
    "Output real-valued actions — the regime real robots live in. "
    "See [the lesson](README.md).",
    [
        md("## SAC on continuous LunarLander"),
        code("from rl.envs import make_env, env_dims",
             "from rl.agents.sac import SAC, SACConfig",
             "from rl.train import train_offpolicy",
             "from rl.utils import Logger, plot_scores, set_seed",
             "",
             "set_seed(0)",
             "env = make_env('LunarLander-v3', continuous=True, seed=0)",
             "s_dim, a_dim, _ = env_dims(env)",
             "scale = float(env.action_space.high[0])",
             "agent = SAC(s_dim, a_dim, action_scale=scale,",
             "            cfg=SACConfig(learning_starts=5000), device=DEVICE)",
             "scores = train_offpolicy(agent, env, n_steps=200_000, discrete=False,",
             "                         logger=Logger('runs/sac_nb'), solved_at=200)",
             "plot_scores(scores, title='SAC on LunarLanderContinuous', show=True)"),
        md("**Compare:** swap `SAC` for `TD3` and `DDPG`. Plot all three on the same "
           "axes to feel the sample-efficiency and stability differences."),
    ],
)

NOTEBOOKS["06_imitation_offline"] = notebook(
    "Module 06 — Behavioral Cloning",
    "Imitation learning — the dominant paradigm in modern robotics. "
    "See [the lesson](README.md).",
    [
        md("## Clone an expert PPO/DQN policy with supervised learning\n"
           "First train (or load) an expert, then clone it and compare returns."),
        code("from rl.envs import make_env, env_dims",
             "from rl.agents.dqn import DQN, DQNConfig",
             "from rl.agents.bc import BehavioralCloning, BCConfig, collect_demonstrations",
             "from rl.train import train_offpolicy",
             "from rl.utils import ExponentialSchedule, set_seed",
             "import numpy as np",
             "",
             "set_seed(0)",
             "env = make_env('LunarLander-v3', seed=0)",
             "s_dim, a_dim, _ = env_dims(env)",
             "",
             "# 1) Train a quick expert (use more steps for a stronger teacher)",
             "expert = DQN(s_dim, a_dim, DQNConfig(), device=DEVICE)",
             "train_offpolicy(expert, env, n_steps=120_000,",
             "                eps_schedule=ExponentialSchedule(1.0, 0.01, 0.999), solved_at=200)",
             "",
             "# 2) Collect demonstrations and clone them (pure supervised learning)",
             "states, actions = collect_demonstrations(env, expert, n_episodes=50)",
             "bc = BehavioralCloning(s_dim, a_dim, discrete=True, cfg=BCConfig(epochs=80), device=DEVICE)",
             "bc.fit(states, actions)"),
        md("### Evaluate the clone — and witness covariate shift"),
        code("import numpy as np",
             "def eval_agent(agent, n=20, greedy_dqn=False):",
             "    rets = []",
             "    for _ in range(n):",
             "        s, _ = env.reset(); done=False; R=0",
             "        while not done:",
             "            a = agent.act(s, eps=0.0) if greedy_dqn else agent.act(s)",
             "            s, r, t, tr, _ = env.step(a); R += r; done = t or tr",
             "        rets.append(R)",
             "    return np.mean(rets)",
             "print('expert return :', round(eval_agent(expert, greedy_dqn=True), 1))",
             "print('clone  return :', round(eval_agent(bc), 1))",
             "print('\\nIf the clone underperforms the expert, that gap is covariate shift —',",
             "      'states the expert never visited. Fixes: more/recovery data, DAgger, RL fine-tuning.')"),
    ],
)

NOTEBOOKS["07_vla_robotics"] = notebook(
    "Module 07 — VLA mini-demo",
    "An instruction-conditioned policy: the irreducible core of a "
    "Vision-Language-Action model. See [the lesson](README.md).",
    [
        md("## Run the tiny VLA: one policy, two instructions\n"
           "The same observation demands different actions depending on the language "
           "instruction — and a single conditioned policy learns both via behavioral cloning."),
        code("import sys; sys.path.insert(0, 'curriculum/07_vla_robotics')",
             "import vla_minidemo as demo",
             "import torch; torch.manual_seed(0)",
             "",
             "policy = demo.ConditionedPolicy()",
             "demo.train_bc(policy, epochs=60)",
             "acc = demo.evaluate(policy)",
             "print(f'\\nInstruction-following success: {acc:.1%}')",
             "demo.demonstrate_conditioning(policy)"),
        md("### What to change to make it a *real* VLA\n"
           "- Replace the `(x,y)` observation with **camera images** + a vision encoder (ViT/SigLIP).\n"
           "- Replace the instruction `Embedding` with a **text encoder** (a real tokenizer/LLM).\n"
           "- Replace the grid with a **robot env** and continuous actions (Module 05).\n"
           "- **Scale the demonstration data**, then **RL-fine-tune** with PPO (Module 04).\n"
           "\nThat path — BC at scale + RL refinement — is how RT-2, OpenVLA, Octo and π0 are built."),
    ],
)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for module, nb in NOTEBOOKS.items():
        path = os.path.join(here, "curriculum", module, "notebook.ipynb")
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
