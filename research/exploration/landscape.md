# Research Landscape: Lenia + Automated Search for Artificial Life

## Working Problem Statement
*TBD — substrate is reproduced (Orbium glider works, see `proto/lenia-baseline/`). The research angle is still open. Most likely problem framing: "**Beat CLIP-as-judge** in ASAL-style Lenia search using a foundation-model-free lifelikeness metric, demonstrated by rediscovering Orbium-class creatures at lower compute and with less reward-hacking risk."* Will sharpen after the user picks an angle.

## What Is Known
- **Lenia substrate is well-defined and stable**: single-channel 2D continuous CA, bell-shaped kernel + polynomial growth, ~6 hyperparameters (R, T, μ, σ, b, kernel/growth families). Chan (2019) catalogued 400+ self-organized species in 18 families; Orbium *unicaudatus* is the canonical "glider".
- **Baseline reproduces Orbium cleanly**: 76.86 → 73.81 mass (~0.7 % oscillation, no drift after t ≈ 3), straight-line COM trajectory at 6.25 px / Lenia time-unit on a 128×128 torus. Implementation is 250 LOC of pure NumPy + matplotlib.
- **ASAL (Sakana AI, 2024)** is the SOTA search-over-Lenia method: CLIP embedding as judge, Sep-CMA-ES for continuous targets/open-endedness, custom GA for illumination. Supports 7 ALife substrates beyond Lenia.
- **Generalizations exist**: Glaberish (Davis 2022) decouples kernel from growth; "Existence of Life in Lenia" (Kojima 2022) formalizes soliton stability.

## Candidate Metrics
- **CLIP–text cosine similarity** (ASAL supervised) — given a text prompt, match the simulation's final-frame embedding.
- **CLIP temporal novelty** (ASAL open-endedness) — maximize variance of frame embeddings over the trajectory.
- **CLIP embedding coverage / illumination** (ASAL) — MAP-Elites-style diversity across embedding space.
- *Candidate alternatives (foundation-model-free):*
  - **Mass quasi-conservation residual** — `std(mass) / mean(mass)` over a sustained window; finite implies a settled creature.
  - **Net locomotion speed** — unwrapped COM displacement per Lenia time-unit, divided by R.
  - **Soliton persistence under perturbation** — fraction of perturbations from a target initial condition that still converge to a moving pattern (operationalizes Kojima 2022).
  - **Bilateral / radial symmetry score** — autocorrelation under reflection / rotation, normalized by image entropy.

## Candidate Systems
- *Default test matrix (placeholders, to be filled at `/init` time):*
  - `lenia-orbium-128` — single-channel R=13, T=10 grid; eval search must recover Orbium-like creatures from random initial condition + seed parameters.
  - `lenia-multichannel-128` — 3-channel R=18, T=10 grid (Chan 2020 family). Harder, more diverse creatures.
  - `lenia-glaberish-128` — Davis 2022 composition rules. Tests whether the search method survives substrate-extension.

## Candidate Baselines
- ASAL Sep-CMA-ES + CLIP-supervised on `lenia-orbium-128` — the obvious comparator.
- Random search (uniform over rule hyperparams + initial conditions) — establishes the floor.
- *To populate:* per-system baseline scores from the ASAL paper for the chosen metric. Will be required for `proposed_eval.yaml`.

## Open Questions
- What does "lifelike" or "interesting" mean as a *non-foundation-model* metric we can optimize against in Lenia rule/initial-condition space?
- Is the goal to *find* novel rules (search problem) or to *evolve* solitons toward target behaviors (optimization problem)? ASAL does both; we may need to pick one.
- Does ASAL's CLIP-as-judge get gamed by adversarial high-entropy textures with no biological structure? Worth a controlled diagnostic.
- Can a non-FM metric rediscover Orbium-class creatures starting from random Lenia parameters?

## Promising Directions
1. **Beat ASAL's CLIP-judge** with a foundation-model-free lifelikeness metric (mass-conservation + locomotion + persistence). Same Sep-CMA-ES, swap the evaluator. Win condition: comparable or better Orbium-class hit rate at lower compute, with a more interpretable / less gameable reward.
2. **Open-endedness benchmark with diagnostics** — construct adversarial Lenia simulations that produce high CLIP-novelty but no genuine soliton structure, and use them to probe whether ASAL's open-endedness metric is robust.
3. **Cheaper search** — gradient-through-simulation (Lenia is fully differentiable in JAX) + learned rule prior beats Sep-CMA-ES on FLOPs to a Orbium-class hit.

## Ruled Out
- *None yet — too early.*

## What Changed This Round
- Substrate is concrete: `proto/lenia-baseline/` reproduces Orbium and produces verifiable figures + animation.
- Initial-condition decoding bug surfaced and was diagnosed via `u`-distribution analysis: a hand-copied RLE string had 21 extra chars and 571 char diffs vs. ground truth, which dropped `u_mean` below the viable growth window. Lesson: bundle reference data (Bert Chan's `animals.json` entry for `O2u`) verbatim; never re-key by hand.
- Concept graph populated with 5 papers + 4 method/question/gap nodes across 3 clusters. `intent_confidence` 0.30 → 0.45 (substrate locked; angle still open).
- One foundation-model-free metric direction (mass + locomotion + persistence) is now the leading candidate for `/init`, pending user confirmation.
