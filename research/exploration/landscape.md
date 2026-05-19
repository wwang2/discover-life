# Research Landscape: Lenia + Automated Search for Artificial Life

## Working Problem Statement
**Beat CLIP-as-judge** in ASAL-style Lenia search with a foundation-model-free five-dim lifelikeness vector `(mass_cv, locomotion_speed, footprint, dihedral_symmetry, temporal_complexity)` that (i) cleanly classifies Chan's canonical translator / rotator / large-undulator species, (ii) cannot be gamed by a static Gaussian blob, and (iii) is computable in <1 ms per frame on CPU. The campaign question: can the same vector (or a monotone combination of it) drive Sep-CMA-ES over Lenia rule space toward novel Orbium-class creatures at comparable hit rate to CLIP-supervised ASAL, at a fraction of the per-evaluation compute? Both pre-`/init` weaknesses identified in the first zoo round have been closed — see `proto/lenia-zoo/report.md`.

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
- *Foundation-model-free vector — implemented in `proto/lenia.py` and validated on a 7-species zoo (6 real Lenia creatures + a synthetic static-blob negative control) in `proto/lenia-zoo/`:*
  - **`mass_cv`** = `std(mass[3T:]) / mean(mass[3T:])`. Zoo: real 0.002–0.025, static 0.000. ✓
  - **`locomotion_speed`** = unwrapped COM displacement post-transient / Lenia time. Zoo: real 0.5 (rotator) – 6.4 (translator), static 0.0. ✓
  - **`footprint`** = mean `#{A > 0.1}` post-transient. Zoo: 144–614 (real), 277 (static). ✓
  - **`dihedral_symmetry`** (replaces the earlier bilateral-only metric) = max over {8 reflection-axis candidates ∪ rotation orders {2,3,4,6}} of self-correlation after COM-centering. Zoo: real 0.72–0.89 (Synorbium 0.78, up from 0.31), static 1.00. ✓
  - **`temporal_complexity`** = pixel-wise std of frames after centering each on its instantaneous COM. Captures *internal* dynamics only — translation alone contributes nothing. Zoo: real 0.0007–0.0105, **static 0.0000**. ✓ — load-bearing leg against trivial winners.
- *Gate metric:* **`persistent`** = `mass[15:].min() > 0.5 · mean(mass[5:15])`. Binary; all 7 zoo entries pass (necessary but not sufficient, hence the four real-valued legs above).

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
- **Both pre-`/init` metric gaps closed.** `dihedral_symmetry` extends bilateral to the full D_n group (Synorbium 0.31 → 0.78, all real creatures now cluster in 0.72–0.89). `temporal_complexity` (pixel-wise std of COM-centered frames) is the load-bearing leg against trivial winners: STATIC = 0.0000 vs ≥ 0.0007 for every real creature.
- **Negative control validated.** A static Gaussian blob passes mass_cv, footprint, symmetry, and persistent — the failure mode I'd been worried about — but is cleanly excluded by `temporal_complexity = 0`.
- **Eval-matrix sketch in `proto/lenia-zoo/report.md`** § "Defensible eval matrix sketch": four-leg "is lifelike" gate (`persistent ∧ mass_cv < 0.05 ∧ temporal_complexity > τ ∧ dihedral_symmetry > 0.6`), then a monotone scalarization of `speed`, `footprint`, `dihedral_symmetry` for ranking. Exact scalarization deferred to `/init` panels + eval-adversary.
- **Two adversarial concerns flagged for `/init` Phase 2.2**: slow-drifting smooth blob with `temporal_complexity` just above τ; two-creature collision that fuses into a rotating clump.
- `intent_confidence` 0.60 → 0.80 (substrate + diagnostic vector + negative control all working; the open work is `/init`-level matrix freezing, not exploration).
