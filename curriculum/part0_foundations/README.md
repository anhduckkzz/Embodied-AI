# Part 0 — Foundations: the math & ML you need (from zero)

> You do **not** need to be a mathematician. You need *working intuition* for a
> handful of ideas that recur everywhere in this curriculum. This part is a map +
> the minimum you must internalize, with the best free resources to go deeper.
> Skim it now, refer back as later parts use each tool.

If you remember one thing: **almost everything here is "turn a goal into a number,
then change parameters to improve that number."** That's optimization, and it's
the soul of both deep learning and control.

---

## 1. Linear algebra — the language of data and geometry

Robots and neural nets live in vectors and matrices.
- **Vector** — a list of numbers; a point/direction (a robot state, a pixel
  embedding). **Matrix** — a linear map; rotates/scales/projects vectors.
- **Matrix multiply** — composition of linear maps (this is *exactly* how we
  chained robot frames in Part 3, and how a neural net layer works).
- **Dot product** — similarity/projection; **norm** — length/distance.
- **Eigenvalues/vectors** — directions a matrix stretches; control stability
  (Part 14) and PCA hinge on them.
- **SVD** — the "swiss-army" decomposition; behind ICP (Part 6), least-squares,
  dimensionality reduction.
> Resource: 3Blue1Brown, *Essence of Linear Algebra* (watch all of it).

## 2. Calculus & gradients — how learning happens

- **Derivative/gradient** — the slope; which way to nudge a parameter to change
  the output. The **gradient** points uphill; we step *against* it to minimize.
- **Chain rule** — derivatives compose; **backpropagation** is the chain rule
  applied through a neural net.
- **Jacobian/Hessian** — matrices of first/second derivatives (Jacobian also
  appears in robot kinematics, Part 3 — same math, different use).
> Resource: 3Blue1Brown, *Essence of Calculus*; then the backprop chapter of any
> DL course.

## 3. Probability & statistics — reasoning under uncertainty

Every sensor is noisy, every outcome uncertain (Parts 5, 7; all of RL).
- **Random variable, distribution, expectation `E[·]`, variance.** RL maximizes
  *expected* return; the variance is why training is noisy.
- **Conditional probability & Bayes' rule** — update beliefs with evidence; the
  literal engine of state estimation/SLAM (Part 7).
- **Gaussian distribution** — the default model of noise; Kalman filters (Part 7)
  and Gaussian policies (Part 1) are built on it.
- **Sampling & Monte Carlo** — estimate expectations by averaging samples (REINFORCE,
  particle filters, MCTS).
> Resource: Sutton & Barto Ch. 2–3 use exactly this; *Probabilistic Robotics* Ch. 2.

## 4. Optimization — the unifying idea

- **Objective/loss function** — the number we want to minimize (error) or
  maximize (reward).
- **Gradient descent** — iteratively step parameters downhill: `θ ← θ − α∇L`.
  **SGD** uses a noisy gradient from a minibatch; **Adam** adapts the step size
  (the default optimizer you'll use everywhere).
- **Convex vs non-convex** — neural nets are non-convex (many local minima); we
  accept "good enough."
- **Constrained optimization** — optimize subject to limits (safe RL, MPC,
  trajectory optimization — Parts 14, 15).
> Resource: the optimization lectures of any ML course; the Adam paper's intuition.

## 5. Machine learning basics — the workflow

- **Supervised learning** — learn `x → y` from labeled examples (classification/
  regression). **Imitation learning** (Part 12) is supervised learning of actions.
- **Unsupervised / self-supervised** — find structure without labels (how vision
  foundation models, Part 16, are pretrained).
- **Train/validation/test split, overfitting, regularization, generalization** —
  the universal concerns. A policy that "overfits" the sim won't transfer (Part 4).
- **Loss, epochs, batches, learning rate** — the training vocabulary.

## 6. Neural networks & deep learning — the function approximators

- **Neuron/layer** — a linear map + a nonlinearity (ReLU). Stack them → an **MLP**
  that can approximate (almost) any function. (You used `rl/networks.py:mlp`.)
- **Backprop + SGD** — how the weights are learned (sections 2 & 4 combined).
- **CNNs** — weight-sharing for images (edges→textures→objects); the classic
  vision backbone (Part 6).
- **RNNs/LSTMs** — networks with memory for sequences (POMDPs, Part 1.13).
- **Transformers & attention** — the modern backbone for language *and* vision
  *and* action (Parts 15–16, VLA). Covered in depth in Part 16.
- **Embeddings** — learned vector representations of discrete things (words,
  actions, instructions — the VLA "language encoder").
> Resource: Andrej Karpathy, *Neural Networks: Zero to Hero* (build it from
> scratch — perfect for "from zero"); fast.ai; the *Dive into Deep Learning* book.

## 7. The tools you'll actually type

- **Python + NumPy** — arrays and math (this repo's `robotics/` is pure NumPy).
- **PyTorch** — tensors + autograd + GPU; defines/trains neural nets (this repo's
  `rl/`). Learn `nn.Module`, `optim`, `loss.backward()`, `.to('cuda')`.
- **Matplotlib** — plots (every demo here).
- Later: **Gymnasium** (RL envs), **OpenCV/Open3D** (vision/3D), **ROS 2** (robots),
  **Hugging Face** (foundation models/VLA).
> On your **RTX 4060**: `pip install torch` with CUDA; 8 GB VRAM trains small/
> medium nets and runs most demos. Use the **A100 (Colab)** for big models.

## 8. How much do you need *before* starting?

Enough to not be scared by the symbols. Concretely: comfort with **vectors/
matrices**, **gradients & gradient descent**, **probability & expectation**, and
**how a neural net is trained**. You can learn the rest *just in time* — each
later part introduces what it needs. **Start Part 1 once sections 1–6 feel
familiar**, and come back here whenever a symbol bites.

## ✅ Check your understanding
1. Why do we step *against* the gradient to minimize a loss?
2. What does backpropagation actually compute, and via which rule?
3. State Bayes' rule and where it appears in robotics.
4. What's the difference between supervised, self-supervised, and RL?
5. Why does a transformer's "attention" matter for VLA models?

## 📖 Go deeper (all free)
- 3Blue1Brown: Linear Algebra, Calculus, Neural Networks (visual intuition).
- Karpathy: *Neural Networks: Zero to Hero* (code it yourself).
- *Dive into Deep Learning* (d2l.ai); fast.ai Practical Deep Learning.
- Math-for-ML: *Mathematics for Machine Learning* (Deisenroth et al., free PDF).

➡️ **Next:** [Part 1 — Reinforcement Learning](../part1_reinforcement_learning/),
starting with [the foundations](../part1_reinforcement_learning/00_foundations/).
