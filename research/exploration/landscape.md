# Research Landscape: Lenia + Automated Search for Artificial Life

## Working Problem Statement
*Drafting: "**Beat CLIP-as-judge** in ASAL-style Lenia search using a foundation-model-free lifelikeness metric — a 5-dimensional vector `(mass_cv, locomotion_speed, footprint, symmetry, persistent)` already separates Chan's canonical translator / rotator / large-undulator behavior classes on a six-creature zoo (see `proto/lenia-zoo/`). The campaign question is whether the same vector (or a monotone combination of it) is sufficient to drive Sep-CMA-ES over Lenia rule space toward novel Orbium-class creatures."*  Two known weaknesses to address before freezing the eval at `/init`: (a) the bilateral-only symmetry metric undersells D4-symmetric creatures (Synorbium scored 0.31 despite being highly symmetric); (b) a static blob trivially satisfies every diagnostic — need a non-trivial "temporal complexity" leg.

## What Is Known
- **Lenia substrate is well-defined and stable**: single-channel 2D continuous CA, bell-shaped kernel + polynomial growth, ~6 hyperparameters (R, T, μ, σ, b, kernel/growth families). Chan (2019) catalogued 400+ self-organized species in 18 families; Orbium *unicaudatus* is the canonical "glider".
- **Baseline reproduces Orbium cleanly**: 76.86 → 73.81 mass (~0.7 % oscillation, no drift after t ≈ 3), straight-line COM trajectory at 6.25 px / Lenia time-unit on a 128×128 torus. Implementation is 250 LOC of pure NumPy + matplotlib. Shared simulator at `proto/lenia.py`.
- **Six-species zoo runs in 2 s total** on CPU and cleanly separates three behavior modes: translators (Orbium uni/bi/phantasma, Synorbium ~6 px/u), rotator (Gyrorbium ~0.5 px/u traces a closed COM loop), large-undulator (Vagorbium ~5.9 px/u with 614-px footprint vs 150–275 for others). See `proto/lenia-zoo/report.md`.
- **ASAL (Sakana AI, 2024)** is the SOTA search-over-Lenia method: CLIP embedding as judge, Sep-CMA-ES for continuous targets/open-endedness, custom GA for illumination. Supports 7 ALife substrates beyond Lenia.
- **Generalizations exist**: Glaberish (Davis 2022) decouples kernel from growth; "Existence of Life in Lenia" (Kojima 2022) formalizes soliton stability.

## Candidate Metrics
- **CLIP–text cosine similarity** (ASAL supervised) — given a text prompt, match the simulation's final-frame embedding.
- **CLIP temporal novelty** (ASAL open-endedness) — maximize variance of frame embeddings over the trajectory.
- **CLIP embedding coverage / illumination** (ASAL) — MAP-Elites-style diversity across embedding space.
- *Foundation-model-free vector — implemented in `proto/lenia.py` and validated on six creatures in `proto/lenia-zoo/`:*
  - **`mass_cv`** = `std(mass[3T:]) / mean(mass[3T:])` — coefficient of variation of total mass post-transient. Lower = better conserved. Zoo range: 0.002–0.025. ✓
  - **`locomotion_speed`** = unwrapped COM displacement post-transient, divided by Lenia time. Zoo range: 0.5 (rotator) – 6.4 (translator). Cleanly separates translators from rotators. ✓
  - **`footprint`** = mean `#{A > 0.1}` post-transient. Zoo range: 144–614. Separates large undulators from compact gliders. ✓
  - **`symmetry`** = max over {horizontal, vertical} reflection self-correlation (centered on COM). Zoo range: 0.31–0.71. **Caveat:** undersells D4 creatures (Synorbium 0.31 despite being highly symmetric under the full D4 group).
  - **`persistent`** = `mass[15:].min() > 0.5 · mean(mass[5:15])` — boolean. All six zoo creatures pass. Necessary but obviously not sufficient (a static blob also passes — see "reward-hacking surface" below).
- *Candidate sixth leg (not yet implemented):* **temporal complexity** = variance of `(area_t, sym_t)` post-transient — rules out static blobs that game the existing five.

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
- **Substrate is concrete**: `proto/lenia-baseline/` reproduces Orbium; `proto/lenia.py` extracted as the shared simulator (baseline numbers reproduce byte-for-byte after the refactor).
- **Six-creature zoo (`proto/lenia-zoo/`)**: all stable under the same simulator, five diagnostics cleanly separate translator / rotator / large-undulator modes. The foundation-model-free metric is no longer hypothetical — it works as a *classifier*. Whether it works as a *search objective* is the open question.
- **Initial-condition decoding bug**: a hand-copied RLE string had 21 extra chars and 571 char diffs vs. ground truth, which dropped `u_mean` below the viable growth window. Lesson: bundle reference data verbatim; never re-key by hand.
- **Symmetry metric weakness identified**: only checks H/V reflection, so D4-symmetric Synorbium scores 0.31. Need to extend to full D_n group before freezing eval — a 4-8 line fix.
- **Reward-hacking risk identified**: static blob trivially passes mass_cv ≈ 0, footprint > 0, symmetry → 1, persistent → True. Need a "temporal complexity" leg.
- `intent_confidence` 0.30 → 0.45 → 0.60 (substrate locked, metric prototyped on real data; specific gaps to close before `/init`).
