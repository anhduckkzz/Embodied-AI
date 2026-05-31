"""Twin Delayed DDPG (TD3) - three targeted fixes that make DDPG reliable.

TD3 (Fujimoto et al., 2018) keeps DDPG's structure and adds:

1. **Twin critics** - learn two Q-functions, use the *minimum* for the target.
   Taking the min counteracts the systematic over-estimation that wrecks DDPG.
2. **Target policy smoothing** - add clipped noise to the target action so the
   critic cannot exploit sharp peaks (a form of regularisation).
3. **Delayed policy updates** - update the actor (and targets) less often than
   the critics, so the actor chases a more stable value estimate.

It inherits DDPG's config and only overrides the learning rule, which is the
cleanest way to *see* what the three tricks change.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from rl.buffers import ReplayBuffer
from rl.networks import ContinuousQNetwork, DeterministicActor


@dataclass
class TD3Config:
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 5e-3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden: tuple = (256, 256)
    exploration_noise: float = 0.1
    policy_noise: float = 0.2        # std of target-smoothing noise
    noise_clip: float = 0.5          # clip range for that noise
    policy_delay: int = 2            # actor update frequency
    learning_starts: int = 10_000


class TD3:
    def __init__(self, state_size: int, action_size: int, action_scale: float = 1.0,
                 cfg: TD3Config = None, device="cpu", seed: int = 0):
        self.cfg = cfg or TD3Config()
        self.device = device
        self.action_size = action_size
        self.action_scale = action_scale
        torch.manual_seed(seed)

        self.actor = DeterministicActor(state_size, action_size, self.cfg.hidden, action_scale).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        # Two independent critics.
        self.critic1 = ContinuousQNetwork(state_size, action_size, self.cfg.hidden).to(device)
        self.critic2 = ContinuousQNetwork(state_size, action_size, self.cfg.hidden).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.cfg.critic_lr)
        self.memory = ReplayBuffer(self.cfg.buffer_size, state_size, action_size, device, seed)
        self.rng = np.random.default_rng(seed)
        self.total_it = 0

    @torch.no_grad()
    def act(self, state, noise: bool = True):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(state_t).cpu().numpy().squeeze(0)
        if noise:
            action += self.rng.normal(0, self.cfg.exploration_noise * self.action_scale,
                                      size=self.action_size)
        return np.clip(action, -self.action_scale, self.action_scale)

    def step(self, *transition) -> None:
        self.memory.add(*transition)
        if len(self.memory) > max(self.cfg.batch_size, self.cfg.learning_starts):
            self._learn()

    def _learn(self) -> None:
        self.total_it += 1
        states, actions, rewards, next_states, dones = self.memory.sample(self.cfg.batch_size)
        rewards, dones = rewards.squeeze(-1), dones.squeeze(-1)

        with torch.no_grad():
            # (2) target policy smoothing
            noise = (torch.randn_like(actions) * self.cfg.policy_noise
                     ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_actions = (self.actor_target(next_states) + noise
                            ).clamp(-self.action_scale, self.action_scale)
            # (1) clipped double-Q: take the minimum of the two target critics
            q1 = self.critic1_target(next_states, next_actions)
            q2 = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1, q2)
            q_target = rewards + self.cfg.gamma * (1 - dones) * q_next

        critic_loss = (F.mse_loss(self.critic1(states, actions), q_target)
                       + F.mse_loss(self.critic2(states, actions), q_target))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # (3) delayed actor + target updates
        if self.total_it % self.cfg.policy_delay == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            for net, target in ((self.actor, self.actor_target),
                                (self.critic1, self.critic1_target),
                                (self.critic2, self.critic2_target)):
                self._soft_update(net, target)

    def _soft_update(self, net, target) -> None:
        with torch.no_grad():
            for tp, p in zip(target.parameters(), net.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(),
                    "critic1": self.critic1.state_dict(),
                    "critic2": self.critic2.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
