# Research Landscape: Lenia + Automated Search for Artificial Life + Lenia as Computation

## Working Problem Statement
**Map and exploit the collision-outcome "instruction set" of Lenia solitons.** Pairs of Orbiums on intersecting trajectories produce *at least three distinct, deterministic outcome classes* depending on their relative incoming angle: pass-through (∼96 % mass conservation, both solitons survive), spawn-3 (two incoming, three outgoing), and annihilation. We will (a) characterize this instruction set across creature pairs (Orbium × Orbium, Gyrorbium × Orbium, Vagorbium × Orbium) and across the full {angle × phase × spatial offset} configuration space; (b) search for Lenia parameter regimes that *expand* the instruction set with new primitives (logic-gate-like, signal-carrier-like, memory-cell-like); (c) demonstrate at least one compositional circuit (e.g. a 2-input, 1-output collision gate fed by two wire-creatures). Empirical anchor: `proto/lenia-collisions/`.

**Why this is the right question:** the original "beat CLIP-as-judge with a non-FM metric" framing has weaker contribution-size — IMGEP (Reinke 2020) and Sensorimotor-Lenia (Hamon 2024) are already FM-free Lenia evaluators with better empirical traction than we'd reach. By contrast, *no paper in our sweep* attempts to systematically map collision-outcome classes in vanilla or Flow-Lenia. The 5-dim diagnostic vector becomes the substrate-gate ("is the candidate alive at all?"); the search objective becomes the count of *distinct stable collision-outcome classes* (or a downstream task that uses them).

## What Is Known

### Substrate (lenia-core)
- Continuous CA with FFT-toroidal convolution, polynomial kernel + growth (Chan 2019). Multichannel + higher-D extension (Chan 2020).
- **Soliton stability is formalized**: Kojima & Ikegami 2022 give existence-of-life conditions; Kojima 2023 shows asymptotic Lenia is a reaction-diffusion PDE; Kojima, Yevenko & Ikegami 2025 derive a *velocity-free glider equation* and use gradient descent on it to find new gliders (including faster ones than Chan's catalog).
- **Lenia ≡ a neural field equation.** Amari (1977) and Wilson-Cowan (1972) define continuous-population neural dynamics whose mathematical form (`∂_t u = −u + ∫ w(x−y) f(u(y)) dy + I`) is identical to asymptotic Lenia after relabeling. Solitons in Lenia are bump attractors in continuous-attractor neural networks. This is one of the most important findings of this sweep — it places Lenia at the intersection of three communities (ALife, computational neuroscience, deep learning).

### Lenia variants (lenia-variants)
- **Flow-Lenia (Plantec 2022, ALife 2023 Best Paper)** — adds mass conservation; rule parameters live *in the field itself* so multiple species with locally-coherent rules can coexist. Best current substrate for ecology / open-endedness studies. **We ported a minimal NumPy version in `proto/flow_lenia.py` and tested whether Pole 2's instruction set transfers: it does not.** Orbium parameters in Flow-Lenia produce stationary ring solitons that do not interact even at 30-px separation; mass is conserved to 4e-15. Flow-Lenia is a fundamentally different substrate (no dissipation gradient → no translation) and requires its own creature search before collision experiments are meaningful. See `proto/lenia-collisions/report_flow_lenia.md`.
- **Particle Lenia (Mordvintsev 2022)** — energy-based reformulation: each particle follows the negative gradient of a global energy. Differentiable end-to-end; gives a clean variational perspective.
- **Glaberish (Davis & Bongard 2022)** — decouples kernel from growth, opening composition rules outside Lenia's original space.

### Search & diversity (search)
- **ASAL (Kumar et al. 2024, Sakana)** — CLIP embedding as judge × Sep-CMA-ES / GA / brute force × 7 substrates.
- **IMGEP for Lenia (Reinke, Etcheverry, Oudeyer 2020, ICLR)** — curiosity-driven goal exploration on Lenia, **FM-free, predates ASAL by 4 years**, and was specifically built to overcome random-sampling failure in high-D non-linear parameter spaces.
- **Sensorimotor agency in CA (Hamon et al. 2024, Science Advances)** — discovers Lenia creatures that *navigate obstacles robustly*; operationalizes "agency" as a perturbation-response diversity signal. Direct precedent for using behavioral metrics (not CLIP) as the evaluator.
- **Curiosity-driven AI Scientist on Flow-Lenia (Schulze et al. 2025)** — IMGEP + evolutionary activity / compression-complexity / multi-scale entropy.

### Trainable / Neural CA (neural-ca) — this is the cluster that changed our priors most
- **Growing NCA (Mordvintsev et al. 2020, Distill)** — end-to-end differentiable CA grows / regenerates target images from a seed. Conv + ReLU pointwise rule. The canonical "learnable Lenia".
- **Self-classifying NCA (Randazzo et al. 2020, Distill)** — NCA classifies MNIST via local message-passing only, no global pooling. 22 k params.
- **Universal NCA (Béna, Faldor, Goodman, Cully 2025, GECCO)** — **trains NCA via gradient descent to perform matrix multiplication, matrix transpose, AND emulate a NN solving MNIST — all within the CA state.** Single most-relevant precedent we found for "Lenia as computation".
- **NCA for ARC-AGI (Xu & Miikkulainen 2025, ALife)** — NCAs attacking explicit symbolic reasoning. The community is actively probing reasoning capacity.

### Reservoir computing + edge-of-chaos (edge-of-chaos)
- CAs at the edge of chaos exhibit highest computational capacity (Yilmaz 2014, Carroll 2020). "Where in Lenia parameter space does life live?" is the right physics framing for our diagnostic vector — it's an empirical edge-of-chaos manifold.

## Candidate Metrics
- *Foundation-model-free vector (validated in `proto/lenia-zoo/`):* `mass_cv`, `locomotion_speed`, `footprint`, `dihedral_symmetry`, `temporal_complexity`, with `persistent` as a binary gate. Both metric gaps closed (Synorbium D4 lift, static-blob negative-control failure).
- **Hamon-style perturbation-response agency** (from sensorimotor Lenia) — could augment the vector with a 6th leg measuring robustness to random perturbations.
- **CLIP-novelty (ASAL)** as a baseline to beat — must remain in the eval matrix so the contribution is comparable.
- **Computational correctness** (Pole 2): task-specific scalar (signal arrival, gate truth-table fidelity, NN-emulation accuracy à la Béna 2025).

## Candidate Systems (eval matrix)

| system | what it stresses |
|---|---|
| `lenia-orbium-128` | single-channel, R=13, T=10. Baseline + 5 zoo species + STATIC negative control. Already implemented. |
| `flow-lenia-128`   | mass-conservative substrate. Tests whether method generalizes beyond vanilla Lenia. |
| `lenia-task-wire`  | (Pole 2 only) — initial condition has signal-source at one corner; metric is "does a glider carry it to the other corner within N steps". |
| `lenia-task-gate`  | (Pole 2 only) — two glider streams at known phases; metric is logical-correctness of the outgoing glider stream. |

## Candidate Baselines (must beat for any contribution)
- ASAL Sep-CMA-ES + CLIP-supervised on `lenia-orbium-128`. Cited per-system score required.
- Reinke-2020 IMGEP on `lenia-orbium-128`. Same.
- Random search over rule hyperparams + initial conditions. The floor.

## Open Questions
- (Pole 2) **Can soliton collisions be made deterministic and function-shaped?** Vanilla Lenia gliders typically merge or annihilate. Does training K, G (NCA) give us this for free? Does it require Flow-Lenia's mass conservation? Does Glaberish's decoupled kernel/growth help?
- (Pole 2) **What is the information capacity of a single Lenia glider?** Bits per glider. Channel capacity. Hard upper bounds.
- (Both poles) **Does the 5-dim diagnostic vector form a meaningful manifold over Lenia parameter space?** I.e., is there a continuous "lifelike region" with clear boundaries, or is it shattered? This is the empirical edge-of-chaos question for our substrate.
- (Pole 1) **How does our vector compare against Hamon's perturbation-response signal at recovering known Lenia species under matched compute?** Direct head-to-head.

## Promising Directions (sharpened by sweep + collision prototype)

1. **(Pole 2 — committed)** *Map Lenia's collision instruction set, then search for new primitives.* Three stages: (a) full {angle × phase × spatial offset} sweep at 5°-angular resolution for Orbium × Orbium to map the deterministic boundaries between outcome classes; (b) repeat for Gyrorbium, Synorbium, Vagorbium pairs and cross-pairs; (c) search Lenia parameter space (Sep-CMA-ES or curiosity-IMGEP) for *unobserved-in-Chan's-catalog* collision-outcome classes. The 5-dim vector remains as the substrate-gate.
2. **(Pole 1.5 — fall-back, second priority)** *Diagnostic vector as auxiliary loss for trained NCA*: take Béna-2025 setup, add the 5-leg gate as an aux loss; see if the trained CA solves the task while *also* looking like a Lenia creature. Tests "lifelikeness as an inductive bias". Kept as a follow-on if Pole 2 stalls.
3. **(Pole 1 — de-prioritized)** *Cheap, principled, non-FM evaluator head-to-head*: covered by IMGEP and Sensorimotor-Lenia already; contribution would be tooling, not science.

## Ruled Out / De-prioritized
- ❌ *Pure ASAL re-implementation with a different evaluator.* Schulze 2025 and Hamon 2024 are stronger non-CLIP baselines than we'd be able to beat without a sharper angle. The 5-dim vector alone is a tooling contribution, not a paper.
- ❌ *Pure "find more solitons" without a new objective.* Chan's catalog is exhaustive; new families need either new substrate (Glaberish, Flow-Lenia) or a new functional criterion (computation, agency).

## What Changed This Round
- **Mental model upgrade**: Lenia ≠ just an ALife toy. Trainable Lenia (NCA) is *literally a recurrent ConvNet*, and the 2025 Béna paper shows it can be trained to emulate a NN solving MNIST inside the CA state. Asymptotic Lenia is structurally identical to Amari/Wilson-Cowan neural fields. **Lenia sits at the intersection of ALife, computational neuroscience, and deep learning** — picking a problem at that intersection is the highest-leverage move.
- **Pole 2 (computation) committed** after two empirical rounds in `proto/lenia-collisions/`. Round-1 found 3 outcome classes in vanilla Lenia at 45° resolution. Round-2 (placement fix + impact-parameter sweep + Synorbium cross-test + delta-based classifier) sharpened to **a robust, creature-portable 3-class instruction set**: at 135°, spawn+1 is stable across a 30-pixel impact-parameter window and transitions sharply to merge−1 at b ≥ +20. Synorbium reproduces passthrough and spawn+1 with slightly different boundaries.
- **Flow-Lenia substrate detour resolved cleanly**: a minimal NumPy port (`proto/flow_lenia.py`) shows mass is conserved to machine precision but Orbium parameters produce stationary ring solitons that don't interact at any tested separation. Mass conservation eliminates the dissipation gradient that drives translation — *vanilla Lenia gliders are dissipative structures in the Prigogine sense*. Flow-Lenia is a stretch goal, not a parallel system, for the campaign.
- **Two non-FM evaluators predate ours** (IMGEP 2020, Sensorimotor-Lenia 2024). The 5-dim vector is *not* the first non-FM evaluator on Lenia; that's a story-correction.
- 13 new references added; 8 clusters in the concept graph.
- `intent_confidence` 0.75 → 0.85 → **0.90** (Pole 2 empirically verified at two creature pairs with measurable regime boundaries; Flow-Lenia detour confirmed primary substrate. Past the threshold for writing `proposed_eval.yaml`).
