# Module 16.5 - Diffusion Models and Action Heads

How a VLA actually outputs actions. After the VLM backbone (Module 16.4) produces
fused vision-language features, an action head turns those features into robot
commands. There are two dominant designs, and the second relies on diffusion or
flow models, which this module explains.

## 1. The two action-head designs

- Action tokenization (autoregressive): discretize each action dimension into
  bins and treat each bin as a token. The language model then predicts action
  tokens exactly as it predicts words. Simple, reuses the LLM decoder, used by
  RT-2, OpenVLA, and Pi0-FAST. Downsides: discretization error and one-token-at-
  a-time decoding.
- Continuous generative heads (diffusion / flow): directly generate a continuous
  action, usually a short chunk of several future steps at once. Smoother,
  higher frequency, and naturally handle multimodal behavior. Used by Diffusion
  Policy, Octo, and Pi0. This module focuses here.

## 2. Why a simple regression head is not enough

If you train a head to regress the single "correct" action with mean squared
error, it predicts the average action. When the data is multimodal (a human
demonstrator sometimes goes left and sometimes right around an obstacle, both
valid), the average is "go straight into the obstacle". This is the multimodality
problem of behavioral cloning (Part 12). Generative heads solve it by modeling the
whole distribution of good actions, not a single point.

## 3. Diffusion models in one page

A diffusion model learns to generate samples by reversing a gradual noising
process:

- Forward process: take a real data point (here, an action or action chunk) and
  add Gaussian noise over many steps until it is pure noise.
- Reverse process: train a network to predict and remove the noise, step by step.
  Starting from random noise, repeated denoising produces a clean sample from the
  data distribution.
- Conditioning: the denoiser is conditioned on the VLM features (the observation
  and instruction), so it generates actions appropriate to the current scene.

Because it samples from a distribution, a diffusion head can represent "left or
right" as two modes and commit to one, instead of averaging them.

## 4. Flow matching (the newer, faster variant)

Flow matching trains a network to predict a velocity field that transports noise
to data along straight-ish paths, often requiring far fewer sampling steps than
diffusion for similar quality. Pi0 uses flow matching to produce high-frequency
continuous actions. Conceptually it is the same goal as diffusion (turn noise into
a sample from the conditional action distribution) with a more efficient
formulation.

## 5. Action chunking

Both Diffusion Policy and ACT (Part 12) predict a short sequence of future actions
in one shot rather than a single step. This reduces compounding error (Part 12
covariate shift), produces smoother motion, and runs the heavy model less often.
Real-time chunking techniques stitch successive chunks together smoothly during
execution.

## 6. Where this sits

```
images + instruction -> VLM backbone (Module 16.4) -> features -> action head -> actions
                                                                   |          |
                                                       tokenized (RT-2)   diffusion/flow (Pi0, Octo)
```

Choosing the action head is one of the central design decisions in a VLA, trading
simplicity (tokenization) against smooth, multimodal, high-frequency control
(diffusion/flow).

## Project
1. Train a 1-D Diffusion Policy on a deliberately multimodal target distribution
   (samples drawn from two clusters). Show it recovers both modes, while an MSE
   regressor collapses to the mean. This is the clearest demonstration of why
   generative heads matter.
2. Read the Diffusion Policy and Pi0 papers and identify, for each, how the head
   is conditioned on observations and how many sampling steps it uses.

## Check your understanding
1. Why does an MSE-regression action head fail on multimodal demonstrations?
2. Describe the forward and reverse processes of a diffusion model.
3. What is the denoiser conditioned on inside a VLA action head?
4. How does flow matching differ in goal and efficiency from diffusion?
5. What are the benefits of predicting an action chunk instead of one step?

## Go deeper
- Ho et al. (2020), Denoising Diffusion Probabilistic Models.
- Chi et al. (2023), Diffusion Policy; Zhao et al. (2023), ACT.
- Lipman et al. (2022), Flow Matching; Physical Intelligence (2024), Pi0.

Next: [Module 16.6 - Efficient Fine-Tuning](../06_efficient_finetuning/).
