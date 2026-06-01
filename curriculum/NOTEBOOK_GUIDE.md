# Curriculum Notebook Guide

This curriculum keeps all 40 notebooks, but they should not all be studied in the
same way. Each notebook belongs to one of three learning types:

| Type | Purpose | How deep to go |
|---|---|---|
| **Type A — Core mechanism** | Understand the math/code mechanism that appears everywhere. | Slow and detailed. Re-implement, debug, and explain from memory. |
| **Type B — Framework/practice** | Learn how the same mechanism appears in standard libraries, APIs, and simulators. | Medium depth. Focus on inputs, outputs, configs, and what the framework hides. |
| **Type C — Real-system/application** | Combine many components into robotics, autonomous-driving, drone, humanoid, or VLA stacks. | Project depth. Focus on interfaces, failure modes, and system design. |

## Required notebook structure

Every notebook should help you study in this order:

1. **Why this matters** — the real problem this topic solves.
2. **Mental model** — the idea in plain English.
3. **Math mechanism** — the smallest useful formula, with symbols explained.
4. **From-scratch code** — readable code with minimal abstraction.
5. **Line-by-line explanation** — inputs, variables, update rule, outputs.
6. **Debug/visualize** — shapes, curves, maps, frames, paths, or attention weights.
7. **Framework version** — the practical library/API/simulator version.
8. **Scratch → framework mapping** — what maps to what, and what gets hidden.
9. **Real-system role** — where this lives in robots, cars, drones, or VLAs.
10. **Failure modes** — how it breaks and how to debug it.
11. **Exercises** — parameter changes, ablations, and small extensions.
12. **Mini-project** — a small artifact worth keeping.
13. **Next step** — the next notebook/tool/project to study.

## How to use the type labels

- If a notebook is **Type A**, do not rush to frameworks. First make sure you can
  write the key formula and reproduce the small example.
- If a notebook is **Type B**, start by asking: “Which from-scratch concept does
  this framework object replace?”
- If a notebook is **Type C**, draw the system diagram before coding. The goal is
  understanding module boundaries, not memorizing one API.

## Minimum mastery standard

For each notebook, you are done only when you can fill this checklist:

- [ ] I can explain why the topic matters for embodied AI.
- [ ] I can explain the mental model without equations.
- [ ] I can write the core formula or update rule.
- [ ] I can point to the line of code that implements the core mechanism.
- [ ] I can map the scratch implementation to at least one real framework/API.
- [ ] I can place the topic inside a robotics/driving/drone/VLA stack.
- [ ] I can name three failure modes and one debug action for each.
- [ ] I completed the mini-project or wrote my own equivalent.
