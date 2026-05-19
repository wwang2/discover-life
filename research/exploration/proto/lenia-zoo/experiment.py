"""Lenia zoo — run six creatures and compute behavior diagnostics.

Substrate is shared with `proto/lenia-baseline/` via `proto/lenia.py`. Each
creature is decoded from Chan's `Chakazul/Lenia/Python/animals.json` (bundled
in `creatures.json`), run on a 160×160 toroidal grid for ~20 Lenia time units,
and characterized by five diagnostics:

  - mass_cv          coefficient of variation of total mass (post-transient)
  - speed            unwrapped center-of-mass displacement per Lenia time-unit
  - footprint        mean count of cells with A > 0.1
  - symmetry         max over {horizontal, vertical} of reflection self-corr
  - persistent       did mass stay above 50% of initial-mass-after-transient

These five together form a candidate foundation-model-free "lifelikeness"
vector — the diagnostic version of what a non-CLIP evaluator could
optimize against in /init.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE.parent))
from lenia import (  # noqa: E402
    run_simulation,
    mass_cv,
    locomotion_speed,
    footprint_area,
    dihedral_symmetry,
    temporal_complexity,
    persistent,
    synthetic_static_blob_sim,
)

CREATURES = json.loads((HERE / "creatures.json").read_text())
# Stable display order, picked to span behavior modes (translate / rotate /
# 4-fold-symmetric / undulate / two-tailed / slow-dynamics). The "STATIC"
# entry is a synthetic negative control — a Gaussian blob that never changes,
# used to verify the temporal-complexity leg of the diagnostic vector.
ORDER = ["O2u", "O2b", "OG2g", "O4s", "OV2u", "O2p", "STATIC"]


# ---------------------------------------------------------------------------
# Style.
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 150,
})

# One color per creature, reused across all overlay panels.
PALETTE = {
    "O2u":    "#4C72B0",  # blue
    "O2b":    "#DD8452",  # orange
    "OG2g":   "#55A868",  # green
    "O4s":    "#C44E52",  # red
    "OV2u":   "#8172B3",  # purple
    "O2p":    "#937860",  # brown
    "STATIC": "#BBBBBB",  # gray — negative control
}


# ---------------------------------------------------------------------------
# Run all creatures.
# ---------------------------------------------------------------------------

WORLD = 160
LENIA_TIME = 20.0  # how many Lenia time units to simulate per creature


def run_zoo() -> dict:
    results: dict = {}
    for code in ORDER:
        if code == "STATIC":
            name = "static blob (negative control)"
            T = 10
            params = {"R": 0, "T": T, "m": 0.0, "s": 0.0}
            sim = synthetic_static_blob_sim(target_mass=75.0, sigma=6.0, world_size=WORLD, steps=200, T=T)
            steps = 200
            keep_every = 1
            elapsed = 0.0
            t0 = time.time()
        else:
            cre = CREATURES[code]
            name = cre["name"]
            params = cre["params"]
            T = params["T"]
            steps = int(round(LENIA_TIME * T))
            keep_every = max(1, steps // 200)
            t0 = time.time()
            sim = run_simulation(cre, steps=steps, world_size=WORLD, keep_every=keep_every)
            elapsed = time.time() - t0

        # Per-frame diagnostics
        area_t = np.array([(f > 0.1).sum() for f in sim.frames])
        sym_t = np.array([dihedral_symmetry(f) for f in sim.frames])

        skip_t = int(3 * T)
        results[code] = {
            "name": name,
            "params": params,
            "sim": sim,
            "area_t": area_t,
            "sym_t": sym_t,
            "metrics": {
                "mass_cv":     mass_cv(sim.mass, skip=skip_t),
                "speed":       locomotion_speed(sim.com_y, sim.com_x, T, skip=skip_t),
                "footprint":   float(area_t[skip_t:].mean()) if len(area_t) > skip_t else float(area_t.mean()),
                "symmetry":    float(sym_t[skip_t:].mean()) if len(sym_t) > skip_t else float(sym_t.mean()),
                "temporal":    temporal_complexity(sim.frames, sim.com_y, sim.com_x, WORLD, skip=skip_t),
                "persistent":  persistent(sim.mass),
                "final_mass_frac": float(sim.mass[-1] / sim.mass[0]) if sim.mass[0] > 0 else 0.0,
            },
        }
        m = results[code]["metrics"]
        print(f"  {code:6s} {name[:32]:32s} steps={steps:4d} keep={keep_every} elapsed={elapsed:.2f}s  "
              f"sym={m['symmetry']:.3f} temp={m['temporal']:.4f}")
    return results


# ---------------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------------

def make_zoo_figure(results: dict, path: Path) -> None:
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = gridspec.GridSpec(
        4, 7, figure=fig,
        height_ratios=[1.1, 1.0, 1.0, 0.9],
    )

    # Row 0 — final-frame snapshots, one per creature
    for i, code in enumerate(ORDER):
        ax = fig.add_subplot(gs[0, i])
        sim = results[code]["sim"]
        ax.imshow(sim.frames[-1], cmap="magma", vmin=0, vmax=1)
        params = results[code]["params"]
        ax.set_title(
            f"({chr(97+i)}) {results[code]['name']}\n"
            f"R={params['R']}  T={params['T']}  μ={params['m']}  σ={params['s']}",
            fontsize=10,
        )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        # Color-tag the panel border to match the overlay color
        for spine in ax.spines.values():
            spine.set_color(PALETTE[code])
            spine.set_linewidth(2.0)

    # Row 1 left half (cols 0–3): mass(t) overlay
    ax_m = fig.add_subplot(gs[1, 0:4])
    for code in ORDER:
        sim = results[code]["sim"]
        t = np.arange(len(sim.mass)) / sim.params["T"]
        m_norm = sim.mass / sim.mass[0]  # normalize to initial=1 for cross-creature comparison
        ax_m.plot(t, m_norm, color=PALETTE[code], lw=1.5, label=code, alpha=0.85)
    ax_m.set_xlabel("time (Lenia units)")
    ax_m.set_ylabel("mass / initial mass")
    ax_m.set_title("(g) mass conservation (normalized)")
    ax_m.axhline(1.0, color="#888888", lw=0.6, linestyle="--")
    ax_m.legend(loc="upper right", ncol=3, fontsize=9)
    ax_m.set_ylim(0.0, 1.4)

    # Row 1 right half (cols 4–6): COM trajectories
    ax_c = fig.add_subplot(gs[1, 4:7])
    for code in ORDER:
        sim = results[code]["sim"]
        # Re-zero so all trajectories start at (0,0) — easier to compare
        dx = sim.com_x - sim.com_x[0]
        dy = sim.com_y - sim.com_y[0]
        ax_c.plot(dx, dy, color=PALETTE[code], lw=1.5, label=code, alpha=0.85)
        ax_c.scatter([dx[-1]], [dy[-1]], color=PALETTE[code], s=30, zorder=3)
    ax_c.set_xlabel("Δ x  (px from start)")
    ax_c.set_ylabel("Δ y  (px from start)")
    ax_c.set_title("(h) unwrapped COM trajectory")
    ax_c.legend(loc="best", ncol=3, fontsize=9)
    ax_c.axhline(0, color="#888888", lw=0.4, linestyle=":")
    ax_c.axvline(0, color="#888888", lw=0.4, linestyle=":")
    ax_c.set_aspect("equal", adjustable="datalim")

    # Row 2 left (cols 0–3): footprint area(t)
    ax_a = fig.add_subplot(gs[2, 0:4])
    for code in ORDER:
        sim = results[code]["sim"]
        t = np.linspace(0, len(sim.mass) / sim.params["T"], len(results[code]["area_t"]))
        ax_a.plot(t, results[code]["area_t"], color=PALETTE[code], lw=1.5, label=code, alpha=0.85)
    ax_a.set_xlabel("time (Lenia units)")
    ax_a.set_ylabel(r"footprint area  $\#\{A > 0.1\}$  (px)")
    ax_a.set_title("(i) footprint area (size & breathing)")
    ax_a.legend(loc="best", ncol=3, fontsize=9)

    # Row 2 right (cols 4–6): dihedral symmetry(t)
    ax_s = fig.add_subplot(gs[2, 4:7])
    for code in ORDER:
        sim = results[code]["sim"]
        t = np.linspace(0, len(sim.mass) / sim.params["T"], len(results[code]["sym_t"]))
        ax_s.plot(t, results[code]["sym_t"], color=PALETTE[code], lw=1.5, label=code, alpha=0.85)
    ax_s.set_xlabel("time (Lenia units)")
    ax_s.set_ylabel("dihedral self-corr.")
    ax_s.set_title("(j) dihedral symmetry (max over reflection axes + rotation orders)")
    ax_s.legend(loc="best", ncol=4, fontsize=9)
    ax_s.set_ylim(0.0, 1.0)

    # Row 3 — five summary bar charts: speed, mass-CV, footprint, symmetry, temporal
    metric_panels = [
        ("speed",     "(k) speed (px / u)"),
        ("mass_cv",   "(l) mass-CV   ← lower = conserved"),
        ("footprint", "(m) footprint (mean px)"),
        ("symmetry",  "(n) dihedral symmetry (mean)"),
        ("temporal",  "(o) temporal complexity"),
    ]
    sub_gs = gs[3, :].subgridspec(1, len(metric_panels), wspace=0.4)
    for i, (key, title) in enumerate(metric_panels):
        ax_b = fig.add_subplot(sub_gs[0, i])
        vals = [results[code]["metrics"][key] for code in ORDER]
        bars = ax_b.bar(
            range(len(ORDER)),
            vals,
            color=[PALETTE[c] for c in ORDER],
            edgecolor="#222222",
            linewidth=0.6,
        )
        ax_b.set_xticks(range(len(ORDER)))
        ax_b.set_xticklabels(ORDER, fontsize=8, rotation=0)
        ax_b.set_title(title, fontsize=10)
        # Number on top of each bar
        for bar, v in zip(bars, vals):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02 if bar.get_height() >= 0 else bar.get_height() - 0.02,
                f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}",
                ha="center", va="bottom", fontsize=8,
            )
        if vals and max(vals) > 0:
            ax_b.set_ylim(0, max(vals) * 1.18)

    fig.suptitle(
        "Lenia zoo — six species × five behavior diagnostics",
        fontsize=14, fontweight="medium",
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[lenia-zoo] running {len(ORDER)} creatures on {WORLD}x{WORLD} grid, "
          f"~{LENIA_TIME:.0f} Lenia time units each...")
    results = run_zoo()

    print("\n[lenia-zoo] summary metrics:")
    print(f"  {'code':6s} {'name':32s}  {'mass_cv':>8s}  {'speed':>7s}  {'footprint':>10s}  {'symm':>6s}  {'temp':>8s}  persistent")
    for code in ORDER:
        m = results[code]["metrics"]
        print(f"  {code:6s} {results[code]['name'][:32]:32s}  "
              f"{m['mass_cv']:8.4f}  {m['speed']:7.3f}  {m['footprint']:10.1f}  "
              f"{m['symmetry']:6.3f}  {m['temporal']:8.4f}  {str(m['persistent']):>10s}")

    # Persist a slim summary for the report / concept graph
    summary = {
        code: {
            "name": results[code]["name"],
            "params": results[code]["params"],
            "metrics": results[code]["metrics"],
        }
        for code in ORDER
    }
    (HERE / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[lenia-zoo] wrote metrics.json")

    print("[lenia-zoo] rendering comparison figure ...")
    make_zoo_figure(results, FIG_DIR / "results.png")
    print("[lenia-zoo] done.")
