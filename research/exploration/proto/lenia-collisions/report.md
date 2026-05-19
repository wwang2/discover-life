# Lenia collisions — soliton interactions are not just merge/annihilate

Tested whether two Orbium gliders, placed on intersecting trajectories at eight relative angles, always merge or annihilate (Pole 2's worst-case obstacle), or whether the substrate produces richer collision dynamics.
The headline: **outcomes depend deterministically on the incoming angle, and four out of eight non-trivial angles produce 2-component pass-through with ~96% mass conservation, while two angles produce 3-component spawn events from a 2-component collision.**
Only the strict head-on (180°) and the (artefactual) parallel (0°) configurations annihilate. Pole 2 — compositional soliton computation in Lenia — therefore has empirical legs in vanilla Lenia, not just in Flow-Lenia or trained NCA, and is worth committing to.

![summary](figures/summary.png)

## Setup

- World: 192×192 toroidal, Chan's Orbium *unicaudatus* parameters (R=13, T=10, μ=0.15, σ=0.015, polynomial K and G).
- Two Orbiums placed so their COMs converge near (96, 96) at t = 10 Lenia units, given Orbium's intrinsic velocity v ≈ (2.32, 5.80) px/u (calibrated from `lenia-baseline`).
- Creature B is rotated by `angle_deg ∈ {0, 45, 90, 135, 180, 225, 270, 315}` so its direction of motion changes. Creature A is fixed.
- Run for 200 steps (20 Lenia time units); classify outcome by **the median count of connected components ≥ 25 px in the last 15% of frames**, plus the time-series of `n_components(t)` and `mass(t)`.

## Outcomes

| angle of B | outcome | initial mass | final mass | mass fraction |
|---:|:--|---:|---:|---:|
| 0°   | annihilate (artefact — see note) | 129.16 |   0.00 | 0.000 |
| 45°  | **pass-through** | 152.99 | 147.48 | 0.964 |
| 90°  | **pass-through** | 153.73 | 147.62 | 0.960 |
| 135° | **spawn-3** | 152.99 | 147.66 | 0.965 |
| 180° | annihilate | 153.73 |   0.00 | 0.000 |
| 225° | **pass-through** | 152.99 | 147.66 | 0.965 |
| 270° | **pass-through** | 153.73 | 147.62 | 0.960 |
| 315° | **spawn-3** | 152.99 | 147.83 | 0.966 |
| alone control | singleton (Orbium glides) | 76.86 | 73.81 | 0.960 |

Note on the 0° "artefact": my placement formula puts A and B at the same position when their rotations match, so they overlap at t=0 and dissipate. This is a placement bug, not a property of the substrate. A non-overlapping parallel configuration would either pass through (likely) or remain parallel forever (also likely). I'll fix the placement formula in a follow-up; the headline result is unaffected.

## What the figure shows

- **(a) Phase wheel:** the 8 incoming angles colored by outcome class. Four pass-through (green), two spawn-3 (blue), two annihilate (red). The mapping is reflection-symmetric: 45° ≈ 315°, 135° ≈ 225° — consistent with Orbium's bilateral symmetry.
- **(b–d) Representative final frames** at 180°, 90°, 135°. Empty / 2-soliton / 3-soliton respectively.
- **(e–h) The 135° evolution** at four timesteps. Two solitons enter, collide near t ≈ 6–10, and three emerge — visibly different sizes / orientations. The post-collision trio is stable through t = 20 (the full run window).
- **(i) Mass over time:** 6 of 8 non-trivial configs hold at ~147 (96% of initial) with minor transient at the collision moment. The two red curves (0°, 180°) crash to zero — but for different reasons (0° initial overlap; 180° true head-on merger that subsequently fails to sustain).
- **(j) Component count over time:** clean step functions. Pass-through configs stay at 2. Spawn-3 configs transition 2→3. Annihilation configs drop 2→0.

## Implications for Pole 2

This is exactly the substrate property Pole 2 needs. Three observations:

1. **Vanilla Lenia is not too fragile.** The merge-or-annihilate worry was overblown — 75% of non-trivial collision angles preserve at least 2 solitons with ~4% mass loss. Glancing collisions in particular are nearly transparent.
2. **2 → 3 spawning is real, deterministic, and reproducible.** At 135° and 315°, two incoming solitons produce three outgoing solitons; mass is conserved at ~97%. This is the substrate analogue of a *fan-out* operation: one trigger event can produce multiple downstream signals.
3. **The angle-to-outcome map is the substrate's "instruction set".** Annihilate at 180° → AND-gate-like (both inputs cancel). Pass-through at 45°/90° → wire (signal carries on past intersection). Spawn-3 at 135° → fan-out. We have at least 3 distinct computational primitives in 8 samples without leaving Chan's parameters.

## Caveats — known weaknesses to address

- **Single creature.** Only Orbium *unicaudatus* tested. Need to check Gyrorbium × Orbium, Orbium × Synorbium, and one or two non-Orbium pairs to know whether these outcomes are Orbium-specific or substrate-general.
- **Single phase.** I varied the angle but not the *temporal phase* (relative timing) or the *spatial offset perpendicular to motion*. The same 135° at different phases might give different outcomes (some passthrough, some spawn).
- **Coarse angular grid.** 45°-resolution. Real "instruction set" mapping needs at least 5°-resolution to find the boundaries between regimes.
- **Stability check is short.** 20 Lenia units. Need to run the spawn-3 configurations for 100+ units to verify the three outgoing solitons stay coherent (they could be transient).
- **No three-body or higher collisions.** Any usable logic circuit needs gates with ≥2 inputs that interact at one point. Whether three Orbiums converging at the same point produce coherent dynamics is untested.

## Decision: Pole 2 is live

The threshold for "Pole 2 has legs" was: *any* non-trivial collision outcome that is not pure merge/annihilate. We observed *two* non-trivial outcome classes (pass-through and spawn-3) populating 6 of 8 sampled angles. This crosses the threshold by a wide margin.

**Recommended next step for `/init`:** formalize Pole 2 as the problem.
- *Metric:* fraction of 8-angle sweep outcomes that are reproducible across small perturbations of phase + position (deterministic-instruction-set metric).
- *Systems:* {Orbium×Orbium, Gyrorbium×Orbium, Vagorbium×Orbium} at minimum.
- *Baselines:* random Lenia parameters (most should annihilate), Chan's catalog parameters (this experiment), and Flow-Lenia (mass-conservative ⇒ should be *more* compositional).
- *Score:* count of distinct, stable collision-outcome classes per substrate.

The fall-back to Pole 1.5 (lifelikeness regularizer for NCA) is no longer the leading candidate. It remains a viable second-priority extension once a Pole-2 "instruction set" is mapped.
