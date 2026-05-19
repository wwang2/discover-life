"""Resolution-stability check — does the 3-class collision instruction set survive
under finer / coarser temporal and spatial resolution?

Motivated by Davis 2024 ("Non-Platonic Autopoiesis of a Cellular Automaton Glider
in Asymptotic Lenia") which shows that asymptotic-Lenia gliders are resolution-
dependent. The Pole 2 eval frozen at /init must measure outcome stability under
resolution perturbations.

Test plan:
  (1) Temporal scaling — fix R=13, world=192, vary T ∈ {5, 10, 20}.
      Run for the same 20 Lenia time units; classifier compares outcomes per angle.

  (2) Spatial scaling — fix T=10, vary (R, world) ∈ {(13, 192), (26, 384)}.
      Upscale the Orbium pattern with scipy.ndimage.zoom to match. (Half-scale
      omitted — Orbium at R=6 is below its size of stability.)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import zoom

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent))
from lenia import decode_rle_2d, place_pattern, simulate_field  # noqa: E402
from experiment import count_components, classify  # noqa: E402

ORBIUM = json.loads((HERE.parent / "lenia-baseline" / "orbium.json").read_text())
ORBIUM_PATTERN = decode_rle_2d(ORBIUM["cells"])
V_ORBIUM = np.array([2.32, 5.80])  # px / Lenia unit, calibrated at R=13, T=10

ANGLES = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
LENIA_TIME = 20.0
T_COLLIDE = 10.0
PERP_OFFSET = 8.0


def make_initial_state(angle_deg: float, *, R: int, world: int, scale: float = 1.0):
    """Build a 2-Orbium initial state at the given scale.

    `scale` rescales the pattern and the meeting point; R and world are derived
    externally."""
    pattern = ORBIUM_PATTERN
    if scale != 1.0:
        pattern = zoom(pattern, scale, order=1)
        pattern = np.clip(pattern, 0, 1)

    v = V_ORBIUM * scale
    center = (world // 2, world // 2)

    A_start = (center[0] - int(v[1] * T_COLLIDE / 2),
               center[1] - int(v[0] * T_COLLIDE / 2))

    th = np.deg2rad(angle_deg)
    R_mat = np.array([[np.cos(th), -np.sin(th)],
                       [np.sin(th),  np.cos(th)]])
    vB = R_mat @ v
    perp_A = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    B_start = (int(center[0] - vB[1] * T_COLLIDE / 2 + PERP_OFFSET * scale * perp_A[1]),
               int(center[1] - vB[0] * T_COLLIDE / 2 + PERP_OFFSET * scale * perp_A[0]))

    A0 = np.zeros((world, world))
    A0 = place_pattern(A0, pattern, A_start, angle_deg=0.0)
    A0 = place_pattern(A0, pattern, B_start, angle_deg=angle_deg)
    return A0


def run_sweep(*, T: int, R: int, world: int, scale: float, label: str) -> dict:
    """Run an 8-angle sweep at the given resolution; return per-angle outcomes."""
    steps = int(round(LENIA_TIME * T))
    params = {"R": R, "T": T, "m": 0.15, "s": 0.015}
    out_per_angle = {}
    t0 = time.time()
    for angle in ANGLES:
        A0 = make_initial_state(angle, R=R, world=world, scale=scale)
        sim = simulate_field(A0, params, steps=steps, keep_every=max(1, steps // 100))
        nc_t = np.array([count_components(f)[0] for f in sim.frames])
        outcome = classify(nc_t, sim.mass)
        out_per_angle[angle] = {
            "outcome": outcome,
            "initial_mass": float(sim.mass[0]),
            "final_mass": float(sim.mass[-1]),
            "nc_initial": int(nc_t[0]),
            "nc_final": int(round(np.median(nc_t[-len(nc_t) // 7:]))),
        }
    elapsed = time.time() - t0
    return {"label": label, "T": T, "R": R, "world": world, "scale": scale,
            "elapsed": elapsed, "per_angle": out_per_angle}


# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------

configs = [
    {"label": "T=5  R=13 W=192", "T": 5,  "R": 13, "world": 192, "scale": 1.0},
    {"label": "T=10 R=13 W=192 (default)", "T": 10, "R": 13, "world": 192, "scale": 1.0},
    {"label": "T=20 R=13 W=192", "T": 20, "R": 13, "world": 192, "scale": 1.0},
    {"label": "T=10 R=26 W=384 (2x spatial)", "T": 10, "R": 26, "world": 384, "scale": 2.0},
]

print(f"[resolution-check] running {len(configs)} sweeps × {len(ANGLES)} angles")
all_results = []
for cfg in configs:
    print(f"  {cfg['label']}:")
    res = run_sweep(**cfg)
    all_results.append(res)
    for angle, r in res["per_angle"].items():
        print(f"    {angle:5.0f}°  {r['outcome']:12s}  "
              f"nc {r['nc_initial']}→{r['nc_final']}  "
              f"mass {r['initial_mass']:6.2f}→{r['final_mass']:6.2f}")
    print(f"    ({res['elapsed']:.1f}s)")

# ---------------------------------------------------------------------------
# Comparison table — does outcome match across resolutions?
# ---------------------------------------------------------------------------

print("\n[resolution-check] cross-resolution outcome table:")
header = f"  {'angle':>5s}  " + "  ".join(f"{r['label'][:18]:>18s}" for r in all_results)
print(header)
for angle in ANGLES:
    row = f"  {angle:5.0f}°  " + "  ".join(f"{r['per_angle'][angle]['outcome']:>18s}" for r in all_results)
    print(row)

# Compute agreement matrix
n = len(all_results)
agreement = np.zeros((n, n), dtype=int)
for i in range(n):
    for j in range(n):
        match = sum(1 for a in ANGLES
                    if all_results[i]["per_angle"][a]["outcome"]
                    == all_results[j]["per_angle"][a]["outcome"])
        agreement[i, j] = match

print(f"\n  agreement matrix (out of {len(ANGLES)}):")
print(f"  {'':>22s}  " + "  ".join(f"{r['label'][:18]:>18s}" for r in all_results))
for i, r in enumerate(all_results):
    print(f"  {r['label'][:22]:>22s}  " + "  ".join(f"{agreement[i, j]:>18d}" for j in range(n)))

# Save summary
summary = {
    "configs": [
        {"label": r["label"], "T": r["T"], "R": r["R"], "world": r["world"],
         "scale": r["scale"], "elapsed": r["elapsed"],
         "per_angle": {str(a): v for a, v in r["per_angle"].items()}}
        for r in all_results
    ],
    "agreement_matrix": agreement.tolist(),
    "n_angles": len(ANGLES),
}
(HERE / "resolution_outcomes.json").write_text(json.dumps(summary, indent=2))

# ---------------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------------

OUTCOME_COLOR = {
    "annihilate": "#C44E52",
    "passthrough": "#55A868",
    "spawn+1":    "#4C72B0",
    "spawn+2":    "#3A5D8F",
    "merge-1":    "#DD8452",
}


fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
y_labels = [r["label"] for r in all_results]
x_labels = [f"{a:.0f}°" for a in ANGLES]
for j, angle in enumerate(ANGLES):
    for i, r in enumerate(all_results):
        out = r["per_angle"][angle]["outcome"]
        c = OUTCOME_COLOR.get(out, "#666666")
        ax.add_patch(plt.Rectangle((j - 0.45, n - 1 - i - 0.45), 0.9, 0.9,
                                     facecolor=c, edgecolor="#222222", linewidth=0.6))
        # Short label inside
        short = out.replace("passthrough", "pass").replace("annihilate", "anni") \
                   .replace("spawn+", "+").replace("merge-", "-m")
        ax.text(j, n - 1 - i, short, ha="center", va="center",
                fontsize=9, color="white" if out != "passthrough" else "#102018",
                fontweight="bold")
ax.set_xlim(-0.5, len(ANGLES) - 0.5)
ax.set_ylim(-0.5, n - 0.5)
ax.set_xticks(range(len(ANGLES)))
ax.set_xticklabels(x_labels)
ax.set_yticks(range(n))
ax.set_yticklabels(y_labels[::-1])
ax.set_xlabel("incoming angle of creature B")
ax.set_title("Collision-outcome class — by angle × resolution\n"
             "rows of identical color = resolution-stable substrate")
ax.grid(False)
ax.set_aspect("equal")
fig.savefig(FIG_DIR / "resolution_outcomes.png", dpi=140, bbox_inches="tight")
plt.close(fig)

print(f"\n[resolution-check] wrote {FIG_DIR / 'resolution_outcomes.png'} and resolution_outcomes.json")
