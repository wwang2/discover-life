"""Lenia collisions — do Orbium gliders ever produce deterministic, function-shaped
collision outcomes?

For 8 incoming angles, place a pair of Orbiums on intersecting trajectories and
classify the outcome by connected-component count + total mass:
  * 0 components → annihilation
  * 1 component  → merge
  * 2 components → pass-through  (the interesting case for Pole 2)
  * ≥3 components → spawn / fragmentation

If we ever observe pass-through or spawn, the substrate has at least *some*
non-trivial collision dynamics and Pole 2 (compositional soliton computation)
remains live. If every collision merges or annihilates, the substrate is too
fragile and we should pivot to Flow-Lenia or trained NCA.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import label as cc_label

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE.parent))
from lenia import decode_rle_2d, place_pattern, simulate_field  # noqa: E402

# Load Orbium from the baseline directory's bundled JSON.
ORBIUM_PATH = HERE.parent / "lenia-baseline" / "orbium.json"
ORBIUM = json.loads(ORBIUM_PATH.read_text())
ORBIUM_PARAMS = {
    "R": ORBIUM["params"]["R"],
    "T": ORBIUM["params"]["T"],
    "m": ORBIUM["params"]["m"],
    "s": ORBIUM["params"]["s"],
}
ORBIUM_PATTERN = decode_rle_2d(ORBIUM["cells"])


# ---------------------------------------------------------------------------
# Geometry — Orbium's intrinsic velocity, calibrated from the baseline run.
# ---------------------------------------------------------------------------
# Baseline result: net displacement (dx=+46.35, dy=+116.09) over 20 Lenia time
# units, so v⃗ ≈ (2.32, 5.80) px/u in (x, y), pointing down-right at ~68° below
# horizontal.
V_ORBIUM = np.array([2.32, 5.80])  # (vx, vy) in px / Lenia unit


# ---------------------------------------------------------------------------
# Run all configurations.
# ---------------------------------------------------------------------------

WORLD = 192
LENIA_TIME = 20.0
STEPS = int(round(LENIA_TIME * ORBIUM_PARAMS["T"]))

# Place A at (yA, xA) — no rotation, will move down-right.
# Place B at (yB, xB) with rotation θ. By rotating Orbium by θ we change *its*
# direction of motion — at θ=180° B moves up-left (head-on with A); at θ=90° B
# moves perpendicular to A; at θ=0° both move parallel; etc.
#
# Position B so that at t ≈ 10 Lenia units both creatures' COMs are near the
# world center (96, 96). This guarantees the head-on case actually collides
# and gives the others a chance.

T_COLLIDE = 10.0
A_START = (96 - int(V_ORBIUM[1] * T_COLLIDE / 2), 96 - int(V_ORBIUM[0] * T_COLLIDE / 2))


def b_start(angle_deg: float) -> tuple[int, int]:
    """Where to place creature B so that, with its intrinsic velocity rotated
    by angle_deg, its COM also lands near (96, 96) at t = T_COLLIDE."""
    th = np.deg2rad(angle_deg)
    R_mat = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]])
    vB = R_mat @ V_ORBIUM
    yB = int(96 - vB[1] * T_COLLIDE / 2)
    xB = int(96 - vB[0] * T_COLLIDE / 2)
    return yB, xB


CONFIGS = [
    {"name": "alone",     "angle": None, "skip_B": True},          # control: no creature B
    {"name": "0°",        "angle": 0.0},                            # parallel, same direction
    {"name": "45°",       "angle": 45.0},
    {"name": "90°",       "angle": 90.0},                           # perpendicular T-bone
    {"name": "135°",      "angle": 135.0},
    {"name": "180°",      "angle": 180.0},                          # head-on
    {"name": "225°",      "angle": 225.0},
    {"name": "270°",      "angle": 270.0},
    {"name": "315°",      "angle": 315.0},
]


def count_components(A: np.ndarray, threshold: float = 0.1, min_size: int = 25) -> tuple[int, list[int]]:
    """Connected-component count + per-component mass-pixel-count.
    Filters out components below `min_size` pixels (debris)."""
    labels, n_total = cc_label(A > threshold)
    sizes = []
    for k in range(1, n_total + 1):
        s = int((labels == k).sum())
        if s >= min_size:
            sizes.append(s)
    return len(sizes), sorted(sizes, reverse=True)


def classify(n_components_t: np.ndarray, mass_t: np.ndarray) -> str:
    """Classify outcome from time-series. Looks at *late* state (last 15%)."""
    tail_len = max(1, len(n_components_t) // 7)
    nc = int(round(np.median(n_components_t[-tail_len:])))
    if nc == 0:
        return "annihilate"
    if nc == 1:
        # Single component at end. Was there ever a 2nd? If yes → merge, else → singleton
        return "merge" if n_components_t.max() >= 2 else "singleton"
    if nc == 2:
        return "passthrough"
    return f"spawn-{nc}"


def run_config(config: dict) -> dict:
    A0 = np.zeros((WORLD, WORLD))
    A0 = place_pattern(A0, ORBIUM_PATTERN, A_START, angle_deg=0.0)
    if not config.get("skip_B"):
        B_pos = b_start(config["angle"])
        A0 = place_pattern(A0, ORBIUM_PATTERN, B_pos, angle_deg=config["angle"])

    t0 = time.time()
    sim = simulate_field(A0, ORBIUM_PARAMS, steps=STEPS, keep_every=1)
    elapsed = time.time() - t0

    nc_t = np.array([count_components(f)[0] for f in sim.frames])
    outcome = classify(nc_t, sim.mass)
    return {
        "name": config["name"],
        "angle": config["angle"],
        "sim": sim,
        "nc_t": nc_t,
        "elapsed": elapsed,
        "outcome": outcome,
        "final_mass": float(sim.mass[-1]),
        "initial_mass": float(sim.mass[0]),
    }


# ---------------------------------------------------------------------------
# Plotting.
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

OUTCOME_COLOR = {
    "annihilate": "#C44E52",  # red
    "merge":      "#DD8452",  # orange
    "singleton":  "#888888",  # gray (no collision)
    "passthrough": "#55A868", # green
    "spawn-3":    "#4C72B0",
    "spawn-4":    "#4C72B0",
}


def make_figure(results: list[dict], path: Path) -> None:
    n = len(results)
    fig = plt.figure(figsize=(20, 13), constrained_layout=True)
    gs = gridspec.GridSpec(4, n, figure=fig, height_ratios=[1.0, 1.0, 1.0, 1.0])

    for i, r in enumerate(results):
        sim = r["sim"]
        frames = sim.frames
        n_steps = len(frames) - 1
        snap_idx = [0, n_steps // 3, 2 * n_steps // 3, n_steps]

        for row, idx in enumerate(snap_idx):
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(frames[idx], cmap="magma", vmin=0, vmax=1)
            color = OUTCOME_COLOR.get(r["outcome"], "#000000")
            if row == 0:
                ax.set_title(f"{r['name']}\n{r['outcome']}",
                             fontsize=10, color=color, fontweight="bold")
            else:
                ax.set_title(f"$t = {idx / ORBIUM_PARAMS['T']:.1f}$", fontsize=9)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(1.4)

    fig.suptitle(
        "Orbium × Orbium collisions — eight incoming angles + lone-creature control",
        fontsize=14, fontweight="medium",
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_diagnostics(results: list[dict], path: Path) -> None:
    """Time-series diagnostics: mass(t) and #components(t) overlay for all configs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for r in results:
        if r["name"] == "alone":
            continue  # control
        sim = r["sim"]
        t = np.arange(len(sim.mass)) / ORBIUM_PARAMS["T"]
        color = OUTCOME_COLOR.get(r["outcome"], "#000000")
        axes[0].plot(t, sim.mass, color=color, lw=1.4, alpha=0.8, label=r["name"])
        axes[1].plot(t, r["nc_t"], color=color, lw=1.4, alpha=0.8, label=r["name"])

    axes[0].set_xlabel("time (Lenia units)")
    axes[0].set_ylabel("total mass")
    axes[0].set_title("(a) mass over time — collisions cost mass differently")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)

    axes[1].set_xlabel("time (Lenia units)")
    axes[1].set_ylabel("# connected components (≥ 25 px)")
    axes[1].set_title("(b) component count over time")
    axes[1].set_yticks([0, 1, 2, 3, 4])
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_summary(results: list[dict], path: Path) -> None:
    """Single-page summary: phase-wheel + 4 representative snapshots + 2 time series."""
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.1, 0.9])

    # ---- Row 0: phase wheel (col 0) + 3 representative final-frame snapshots ----
    ax_w = fig.add_subplot(gs[0, 0], projection="polar")
    by_angle = {r["angle"]: r for r in results if r["angle"] is not None}
    angles_deg = sorted(by_angle.keys())
    for a in angles_deg:
        r = by_angle[a]
        color = OUTCOME_COLOR.get(r["outcome"], "#000000")
        th = np.deg2rad(a)
        ax_w.bar(th, 1.0, width=np.deg2rad(40), bottom=0.0,
                 color=color, edgecolor="#222222", linewidth=0.8, alpha=0.85)
        ax_w.text(th, 1.18, f"{a:.0f}°", ha="center", va="center",
                  fontsize=9, fontweight="bold")
        ax_w.text(th, 0.55, r["outcome"][:8], ha="center", va="center",
                  fontsize=7, color="white" if r["outcome"] != "passthrough" else "#222222")
    ax_w.set_ylim(0, 1.5)
    ax_w.set_yticks([])
    ax_w.set_xticks([])
    ax_w.grid(False)
    ax_w.set_title("(a) outcome by incoming angle of B", fontsize=11, pad=15)

    # 3 representative collisions: head-on (annihilate), 90° (passthrough), 135° (spawn-3)
    picks = [("180°", "head-on → annihilate"),
             ("90°",  "T-bone → pass-through"),
             ("135°", "→ spawn-3")]
    by_name = {r["name"]: r for r in results}
    for j, (name, caption) in enumerate(picks, start=1):
        r = by_name[name]
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(r["sim"].frames[-1], cmap="magma", vmin=0, vmax=1)
        color = OUTCOME_COLOR.get(r["outcome"], "#000000")
        ax.set_title(f"({chr(97+j)}) {caption}", fontsize=10, color=color, fontweight="bold")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2.0)

    # ---- Row 1: 4-frame strip from the 135° (spawn-3) collision ----
    spawn = by_name["135°"]
    snap_idx = [0, len(spawn["sim"].frames) // 4,
                len(spawn["sim"].frames) // 2,
                3 * len(spawn["sim"].frames) // 4,
                len(spawn["sim"].frames) - 1]
    # 5 strict subdivisions doesn't fit 4 cols; use 4 evenly-spaced
    snap_idx = [int(round(k)) for k in np.linspace(0, len(spawn["sim"].frames) - 1, 4)]
    for j, idx in enumerate(snap_idx):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(spawn["sim"].frames[idx], cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"({chr(101+j)}) 135°, $t={idx / ORBIUM_PARAMS['T']:.1f}$  →  3 outgoing solitons",
                     fontsize=9)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(OUTCOME_COLOR["spawn-3"])
            spine.set_linewidth(1.6)

    # ---- Row 2: mass(t) + nc(t) ----
    ax_m = fig.add_subplot(gs[2, 0:2])
    ax_n = fig.add_subplot(gs[2, 2:4])
    for r in results:
        if r["name"] == "alone":
            continue
        sim = r["sim"]
        t = np.arange(len(sim.mass)) / ORBIUM_PARAMS["T"]
        color = OUTCOME_COLOR.get(r["outcome"], "#000000")
        ax_m.plot(t, sim.mass, color=color, lw=1.3, alpha=0.8, label=r["name"])
        ax_n.plot(t, r["nc_t"], color=color, lw=1.3, alpha=0.8, label=r["name"])
    ax_m.set_xlabel("time (Lenia units)")
    ax_m.set_ylabel("total mass")
    ax_m.set_title("(i) mass over time")
    ax_m.legend(loc="lower right", ncol=4, fontsize=8)
    ax_n.set_xlabel("time (Lenia units)")
    ax_n.set_ylabel("# connected components (≥ 25 px)")
    ax_n.set_title("(j) component count over time")
    ax_n.set_yticks([0, 1, 2, 3, 4])
    ax_n.legend(loc="lower right", ncol=4, fontsize=8)

    fig.suptitle("Orbium × Orbium collisions — outcomes depend deterministically on incoming angle",
                 fontsize=14, fontweight="medium")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[lenia-collisions] running {len(CONFIGS)} configs, "
          f"world={WORLD}×{WORLD}, steps={STEPS}")
    results = []
    for cfg in CONFIGS:
        r = run_config(cfg)
        results.append(r)
        nm = r["name"] if r["name"] != "alone" else "alone (control)"
        print(f"  {nm:6s}  outcome={r['outcome']:12s}  "
              f"mass {r['initial_mass']:6.2f} → {r['final_mass']:6.2f}  "
              f"({r['elapsed']:.2f}s)")

    print("\n[lenia-collisions] rendering figures ...")
    make_figure(results, FIG_DIR / "results.png")
    make_diagnostics(results, FIG_DIR / "diagnostics.png")
    make_summary(results, FIG_DIR / "summary.png")

    # Save summary
    summary = {
        r["name"]: {
            "angle": r["angle"],
            "outcome": r["outcome"],
            "initial_mass": r["initial_mass"],
            "final_mass": r["final_mass"],
            "mass_fraction": r["final_mass"] / r["initial_mass"] if r["initial_mass"] > 0 else 0.0,
        }
        for r in results
    }
    (HERE / "outcomes.json").write_text(json.dumps(summary, indent=2))
    print("[lenia-collisions] done.")
