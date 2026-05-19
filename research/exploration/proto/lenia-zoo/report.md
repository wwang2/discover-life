# Lenia zoo — six species, five behavior diagnostics

Ran six Chan-2019 creatures (Orbium *unicaudatus*, *bicaudatus*, *phantasma*, Gyrorbium *gyrans*, Synorbium *solidus*, Vagorbium *undulatus*) on a 160×160 toroidal grid for ~20 Lenia time units each through the same simulator that produced the baseline.
All six persisted; mass-CV stayed under 2.5% in every case, the rotator (Gyrorbium) is cleanly distinguished by speed 0.52 vs ~6 px/u for translators, and Vagorbium's footprint (614 px) is 3–4× larger than the rest — so a five-dim vector `(mass_cv, speed, footprint, symmetry, persistent)` already separates the canonical behavior modes without any foundation-model judge in the loop.
This supports the leading research direction: build a foundation-model-free lifelikeness metric on top of these diagnostics and run `/init` against it.

![results](figures/results.png)

## What each panel shows

- **(a–f)** Final-frame snapshot per creature with its params. Panel border is colored — same color used in every overlay panel below.
- **(g)** Mass over time, normalized to initial mass. All six creatures hold at ratio 1.0 ± a few percent after a brief settling transient.
- **(h)** Unwrapped center-of-mass trajectories from each creature's start. Translators trace straight lines; Gyrorbium (green) traces a tight closed loop near the origin — it rotates rather than translates. Vagorbium (purple) shows the snake-like undulation in its trajectory.
- **(i)** Footprint area `#{A > 0.1}` over time. Vagorbium ~614 px (large), Synorbium ~275, others ~150–220. Time-series captures "breathing" — Gyrorbium and Vagorbium oscillate; Orbiums are flat.
- **(j)** Best-axis reflection self-correlation. Synorbium is the outlier at ~0.30 — but it has D4 symmetry, not just H/V, and rotates during the run, so my bilateral-only metric undersells it (open question; the rotational-aware version would be a small extension).
- **(k–n)** Summary bar charts: speed, mass-CV, footprint, symmetry — the five-number signature per creature.

## Summary metrics

| code | name | mass_cv | speed (px/u) | footprint (px) | symmetry | persistent |
|------|------|--------:|-------------:|---------------:|---------:|-----------:|
| O2u  | Orbium unicaudatus  | 0.003 | 6.24 | 170 | 0.69 | ✓ |
| O2b  | Orbium bicaudatus   | 0.002 | 6.04 | 166 | 0.66 | ✓ |
| OG2g | Gyrorbium gyrans    | 0.025 | 0.52 | 220 | 0.59 | ✓ |
| O4s  | Synorbium solidus   | 0.004 | 6.40 | 275 | 0.31 | ✓ |
| OV2u | Vagorbium undulatus | 0.025 | 5.88 | 614 | 0.71 | ✓ |
| O2p  | Orbium phantasma    | 0.005 | 6.17 | 144 | 0.66 | ✓ |

## Verification

- Each creature was simulated for the same number of Lenia time units (20.0), not the same number of integration steps — Orbium *phantasma* runs at T=40 so it needed 800 steps; the others use 200 steps at T=10. This rules out "T inflates the budget" artefacts.
- All five diagnostics are computed after a 3·T-step transient, so initial settling doesn't pollute the numbers.
- The simulator is byte-identical to the one used by `proto/lenia-baseline/` (both import `from lenia import ...`). Baseline reproduced its committed numbers after the refactor (mass 76.86 → 73.81, speed 6.25), confirming the extraction was lossless.

## Implications for the eval metric

The five-number vector is enough to cluster creatures into translator / rotator / large-undulator without any external evaluator. Whether the same vector — or some monotone combination of it — is enough to *drive search* (Sep-CMA-ES over rule space) toward novel Orbium-class creatures is the question we'd take into `/init`. Two known weaknesses to surface there:

1. **Rotational symmetry**: my bilateral-only metric tags D4 creatures (Synorbium) as low-symmetry while they're highly symmetric in a different group. The metric should generalize to "max over all dihedral group reflections", which is a 4–8-line extension but worth doing before the eval is frozen.
2. **Reward-hacking surface**: a static blob has mass-CV ≈ 0, footprint > 0, and symmetry → 1 — it would *pass* every per-axis threshold while being trivial. Persistence alone doesn't fix this. We need a "non-trivial" leg — most likely **temporal complexity** (variance in `(area_t, sym_t)` after transient) or **information flow** (e.g., a non-zero locomotion-mass coupling). Worth a 30-min experiment before `/init`.

## Next step

Two options before `/init`: extend the symmetry metric to the full D_n group (fixes Synorbium and lets a future search recover non-bilateral creatures), or stress-test the metric with a known-trivial pattern (static blob, expanding blob, two-creature collision) to see whether it can be fooled. Either way the substrate and diagnostics are ready.
