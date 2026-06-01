# Module 09 — Exploration & Multi-Armed Bandits

> Code: [`rl/agents/bandits.py`](../../../rl/agents/bandits.py).
> The **explore vs exploit** dilemma in its purest form — no states, no
> transitions, just "which arm pays best?" Every exploration trick in deep RL
> (ε-greedy in DQN, entropy in SAC, curiosity bonuses) descends from here.

## 1. The bandit problem

`k` slot-machine arms; arm `a` pays rewards with unknown mean `q*(a)`. You get
`T` pulls. **Maximize total reward** = quickly find and exploit the best arm
*while* spending enough pulls to be confident which it is. We measure failure as
**regret**: reward lost vs always pulling the true-best arm.

This isolates the core tension: **exploit** (pull your current best guess) vs
**explore** (try others to improve your guess). Too little exploration → stuck on
a suboptimal arm; too much → waste pulls.

## 2. Three canonical strategies (all runnable)

### ε-greedy
With prob ε pull a random arm, else the best estimate. Dead simple, works, but
explores *uniformly* forever (unless ε decays) and ignores *how uncertain* each
arm is. This is exactly DQN's exploration.

### UCB1 — optimism under uncertainty
Pull `argmax_a [ Q(a) + c·√(ln t / N(a)) ]`. The bonus is large for
rarely-pulled arms and shrinks with experience → **directed** exploration toward
genuinely uncertain arms. No randomness needed; logarithmic regret.

### Thompson Sampling — be Bayesian
Keep a posterior over each arm's value (Beta for Bernoulli), **sample** a value
from each, pull the argmax. Naturally explores in proportion to uncertainty;
often the best simple method in practice.

```python
from rl.agents.bandits import BernoulliBandit, UCB1, ThompsonSampling, run_bandit
bandit = BernoulliBandit([0.2, 0.5, 0.75, 0.3])
_, regret = run_bandit(bandit, ThompsonSampling(4), steps=3000)
print("avg regret:", regret[-1] / 3000)   # should be small
```

## 3. From bandits to full RL exploration

In a full MDP, exploration is *harder* because actions affect future *states*
(you must explore *trajectories*, and reward can be sparse/delayed). Key methods:
- **ε-greedy / Boltzmann** — the bandit ideas, applied per-state (DQN).
- **Entropy bonus** — reward policy randomness (A2C, SAC's max-entropy objective).
- **Optimism / count-based bonuses** — reward visiting novel states (UCB-style),
  e.g. pseudo-counts in large spaces.
- **Curiosity / intrinsic motivation** — reward *prediction error* of a learned
  model: **ICM**, **RND** (Random Network Distillation). Drives exploration when
  the extrinsic reward is sparse (Montezuma's Revenge).
- **Posterior sampling / bootstrapped DQN** — Thompson's idea for deep RL.

> Sparse-reward tasks (most real robotics!) live or die by exploration. A robot
> that only gets reward "when the cup is on the shelf" needs curiosity/shaping to
> ever stumble onto success.

## 4. Contextual bandits (the practical middle ground)
Bandits + a state/context but **no** long-term transitions: pick an arm given
features, get reward. This is recommendation systems, ad selection, and A/B
testing — and a clean stepping stone from bandits to RL.

## 🛠️ Project
1. Run all three strategies on the same `BernoulliBandit`; plot cumulative
   regret. Confirm UCB/Thompson beat ε-greedy, which beats random.
2. Sweep ε and UCB's `c`; see the explore/exploit trade-off as a U-shaped curve.
3. (Stretch) Add **RND** intrinsic reward to your Part-1 DQN on a sparse-reward
   task and watch exploration improve.

## ✅ Check your understanding
1. Define regret. Why is it the right success metric for a bandit?
2. How does UCB decide what to explore, without any randomness?
3. Why is Thompson sampling often better than ε-greedy?
4. Why is exploration harder in a full MDP than in a bandit?
5. What is intrinsic motivation, and when do you need it?

## 📖 Go deeper
- Sutton & Barto, Ch. 2 (bandits) — start here. Lattimore & Szepesvári,
  *Bandit Algorithms* (free, comprehensive).
- Pathak et al. (2017) ICM; Burda et al. (2018) RND (curiosity for deep RL).

➡️ **Next:** [Module 10 — Model-based RL](../10_model_based_rl/).
