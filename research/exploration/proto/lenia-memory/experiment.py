"""Lenia as continuous-attractor memory — basin-of-attraction experiment.

Hypothesis (Hopfield-like, content-addressable memory): if Lenia's zoo creatures are
attractors of the dynamics, then a noisy initial condition derived from creature C
should relax back to (a centered, rephased copy of) C, provided the noise stays
inside C's basin of attraction. Failure modes: the noisy IC either (a) dissolves to
mass = 0, (b) wanders into a different creature's basin, or (c) settles into an
amorphous metastable state that doesn't match any zoo creature.

Three experiments:
  (1) Basin of attraction per creature — additive Gaussian noise at 5 levels × 5 seeds
  (2) Pattern completion — mask top half / bottom half / random 50% of the pattern
  (3) Cross-substrate stability — place each non-Orbium creature in Orbium's params

Recovery metric: centered cross-correlation between the final frame and the
clean-run final frame (both centered on their respective COMs).
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
from lenia import decode_rle_2d, run_simulation, _com_periodic  # noqa: E402

ZOO = json.loads((HERE.parent / "lenia-zoo" / "creatures.json").read_text())
ORDER = ["O2u", "O2b", "OG2g", "O4s", "OV2u", "O2p"]

WORLD = 160
LENIA_TIME = 20.0   # how many Lenia time units to run
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]   # additive Gaussian std
N_SEEDS = 5


# ---------------------------------------------------------------------------
# Utilities.
# ---------------------------------------------------------------------------

def _center_on_com(A: np.ndarray) -> np.ndarray:
    """Roll A so its COM lands at the grid center. Periodic-COM trick."""
    m = A.sum()
    if m <= 1e-9:
        return A
    H, W = A.shape
    ys, xs = np.indices((H, W))
    cy, cx = _com_periodic(A, ys, xs, W)
    return np.roll(A, (H // 2 - int(round(cy)), W // 2 - int(round(cx))), axis=(0, 1))


def recovery_score(A_final: np.ndarray, A_canonical: np.ndarray) -> float:
    """Centered normalized cross-correlation between two final frames."""
    if A_final.sum() < 1e-3:
        return 0.0
    Ac = _center_on_com(A_final)
    Cc = _center_on_com(A_canonical)
    num = (Ac * Cc).sum()
    den = np.sqrt((Ac * Ac).sum() * (Cc * Cc).sum())
    return float(num / den) if den > 0 else 0.0


def make_creature_dict(code: str) -> dict:
    """Stitch the zoo entry into the shape expected by run_simulation."""
    return ZOO[code]


# ---------------------------------------------------------------------------
# Experiment 1 — Basin of attraction per creature.
# ---------------------------------------------------------------------------

def add_gaussian_noise(A: np.ndarray, sigma: float, rng: np.random.Generator,
                        only_where_pattern: bool = True) -> np.ndarray:
    """Add Gaussian noise to non-zero region (so we don't seed new clusters far away)."""
    noisy = A + rng.normal(0, sigma, A.shape)
    if only_where_pattern:
        # Mask to cells within distance 1.5*R from the pattern's bounding box
        # (cheap proxy — just add noise everywhere A > 0.02 or its dilation)
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(A > 0.02, iterations=8)
        noisy = np.where(mask, noisy, A)
    return np.clip(noisy, 0, 1)


def basin_experiment(code: str) -> dict:
    """For one creature, run clean baseline + noisy reruns at multiple noise levels."""
    creature = make_creature_dict(code)
    # 1. Clean baseline
    clean = run_simulation(creature, steps=int(LENIA_TIME * creature["params"]["T"]),
                            world_size=WORLD, keep_every=1)
    canonical_final = clean.frames[-1]
    canonical_initial = clean.frames[0]
    canonical_mass = clean.mass[-1]

    # 2. Sweep noise × seeds
    rows = []
    for noise_sigma in NOISE_LEVELS:
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed=hash((code, noise_sigma, seed)) & 0xFFFFFFFF)
            A0 = add_gaussian_noise(canonical_initial, noise_sigma, rng)
            # Reuse run_simulation by hot-substituting the initial state.
            # Easiest: construct a fake "creature" dict whose cells produce A0 — but
            # that requires re-encoding the array as RLE. Cleaner: build directly via
            # simulate_field.
            from lenia import simulate_field
            params = {"R": creature["params"]["R"], "T": creature["params"]["T"],
                       "m": creature["params"]["m"], "s": creature["params"]["s"]}
            sim = simulate_field(A0, params,
                                  steps=int(LENIA_TIME * params["T"]),
                                  keep_every=max(1, int(LENIA_TIME * params["T"]) // 100))
            score = recovery_score(sim.frames[-1], canonical_final)
            mass_frac = sim.mass[-1] / canonical_mass if canonical_mass > 0 else 0.0
            rows.append({"noise": noise_sigma, "seed": seed,
                          "recovery": score, "mass_frac": float(mass_frac),
                          "sim": sim if seed == 0 else None})  # keep first seed's frames
    return {"code": code, "name": creature["name"],
            "canonical_clean": clean, "canonical_final": canonical_final,
            "canonical_initial": canonical_initial, "rows": rows}


def summarize_basins(results: dict) -> dict:
    """Per-(code, noise) mean recovery."""
    summary = {}
    for code, res in results.items():
        by_noise = {}
        for r in res["rows"]:
            by_noise.setdefault(r["noise"], []).append(r["recovery"])
        summary[code] = {nl: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                          for nl, v in by_noise.items()}
    return summary


# ---------------------------------------------------------------------------
# Experiment 2 — Pattern completion (mask half).
# ---------------------------------------------------------------------------

def pattern_completion_experiment(code: str = "O2u") -> dict:
    """Mask portions of the canonical Orbium and see if it completes."""
    from lenia import simulate_field
    creature = make_creature_dict(code)
    clean = run_simulation(creature, steps=int(LENIA_TIME * creature["params"]["T"]),
                            world_size=WORLD, keep_every=1)
    canonical_final = clean.frames[-1]
    A_seed = clean.frames[0]
    H, W = A_seed.shape

    masks = {
        "clean": A_seed,
        "top_half_zeroed": A_seed * (np.indices((H, W))[0] >= H // 2)[None, :, :][0],
        "bottom_half_zeroed": A_seed * (np.indices((H, W))[0] < H // 2)[None, :, :][0],
        "random_50pct_zeroed": A_seed * (np.random.default_rng(seed=42).random((H, W)) > 0.5),
    }
    out = {}
    for name, A0 in masks.items():
        params = {"R": creature["params"]["R"], "T": creature["params"]["T"],
                   "m": creature["params"]["m"], "s": creature["params"]["s"]}
        sim = simulate_field(A0, params,
                              steps=int(LENIA_TIME * params["T"]), keep_every=1)
        out[name] = {"sim": sim,
                      "recovery": recovery_score(sim.frames[-1], canonical_final),
                      "mass_frac": float(sim.mass[-1] / clean.mass[-1])
                                    if clean.mass[-1] > 0 else 0.0}
    return out, canonical_final


# ---------------------------------------------------------------------------
# Experiment 3 — Cross-substrate stability.
# Place each non-Orbium creature in Orbium's substrate params; does it survive?
# ---------------------------------------------------------------------------

def cross_substrate_experiment() -> dict:
    from lenia import simulate_field, decode_rle_2d
    orb_params = {"R": ZOO["O2u"]["params"]["R"], "T": ZOO["O2u"]["params"]["T"],
                   "m": ZOO["O2u"]["params"]["m"], "s": ZOO["O2u"]["params"]["s"]}
    out = {}
    for code in ORDER:
        pattern = decode_rle_2d(ZOO[code]["cells"])
        A0 = np.zeros((WORLD, WORLD))
        h, w = pattern.shape
        cy, cx = WORLD // 2 - h // 2, WORLD // 2 - w // 2
        A0[cy:cy+h, cx:cx+w] = pattern
        sim = simulate_field(A0, orb_params, steps=int(LENIA_TIME * orb_params["T"]),
                              keep_every=max(1, int(LENIA_TIME * orb_params["T"]) // 50))
        # Survival score: did mass stay >50% of initial?
        survived = (sim.mass[-1] / sim.mass[0] > 0.5) if sim.mass[0] > 0 else False
        out[code] = {"sim": sim, "survived": survived,
                      "initial_mass": float(sim.mass[0]),
                      "final_mass": float(sim.mass[-1])}
    return out


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 150,
})
PALETTE = {"O2u": "#4C72B0", "O2b": "#DD8452", "OG2g": "#55A868",
           "O4s": "#C44E52", "OV2u": "#8172B3", "O2p": "#937860"}


def make_basin_figure(results: dict, summary: dict, path: Path) -> None:
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.9])

    # (a) Basin-of-attraction curves — recovery vs noise, per creature
    ax = fig.add_subplot(gs[0, 0:2])
    for code in ORDER:
        if code not in summary: continue
        levels = sorted(summary[code].keys())
        means = [summary[code][nl]["mean"] for nl in levels]
        stds = [summary[code][nl]["std"] for nl in levels]
        ax.errorbar(levels, means, yerr=stds, color=PALETTE[code],
                     lw=1.6, capsize=3, marker="o", label=code, alpha=0.9)
    ax.set_xlabel("noise standard deviation (additive Gaussian)")
    ax.set_ylabel("centered recovery correlation")
    ax.set_title("(a) basin of attraction per creature  (mean ± std over 5 seeds)")
    ax.legend(loc="best", ncol=3)
    ax.axhline(0.95, color="#888888", lw=0.5, linestyle="--")
    ax.axhline(0.50, color="#888888", lw=0.5, linestyle=":")

    # (b) Canonical creature thumbnails (final frames)
    sub_b = gs[0, 2].subgridspec(2, 3)
    for i, code in enumerate(ORDER):
        if code not in results: continue
        ax_t = fig.add_subplot(sub_b[i // 3, i % 3])
        ax_t.imshow(results[code]["canonical_final"], cmap="magma", vmin=0, vmax=1)
        ax_t.set_title(f"{code}", fontsize=9, color=PALETTE[code], fontweight="bold")
        ax_t.grid(False); ax_t.set_xticks([]); ax_t.set_yticks([])

    # (c–h) For each creature, final frame at noise=0.10, 0.30
    for col, code in enumerate(ORDER):
        if code not in results: continue
        # Pick the first-seed sim at noise=0.10 and noise=0.30
        rows = results[code]["rows"]
        n10 = next((r for r in rows if r["noise"] == 0.10 and r["seed"] == 0), None)
        n30 = next((r for r in rows if r["noise"] == 0.30 and r["seed"] == 0), None)
        if n10 is None or n10["sim"] is None: continue
        sub = gs[1, col % 3].subgridspec(2, 2) if col // 3 == 0 else gs[1, col % 3].subgridspec(2, 2)
        # Use the second row of the main grid in 2x3 layout
        idx_in_row = col % 3
        sub = gs[1, idx_in_row].subgridspec(1, 2)
        ax1 = fig.add_subplot(sub[0, 0])
        ax2 = fig.add_subplot(sub[0, 1])
        ax1.imshow(n10["sim"].frames[-1], cmap="magma", vmin=0, vmax=1)
        ax1.set_title(f"{code} σ=0.10\n  r={n10['recovery']:.2f}", fontsize=8,
                       color=PALETTE[code])
        ax1.grid(False); ax1.set_xticks([]); ax1.set_yticks([])
        ax2.imshow(n30["sim"].frames[-1], cmap="magma", vmin=0, vmax=1)
        ax2.set_title(f"σ=0.30  r={n30['recovery']:.2f}", fontsize=8, color=PALETTE[code])
        ax2.grid(False); ax2.set_xticks([]); ax2.set_yticks([])

    fig.suptitle("Lenia as continuous-attractor memory — basin of attraction",
                 fontsize=13, fontweight="medium")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_completion_figure(completion_out: dict, canonical_final: np.ndarray,
                             path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for j, (name, r) in enumerate(completion_out.items()):
        sim = r["sim"]
        # Top row: initial (masked) state
        axes[0, j].imshow(sim.frames[0], cmap="magma", vmin=0, vmax=1)
        axes[0, j].set_title(f"{name}\nt=0", fontsize=10)
        axes[0, j].grid(False); axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        # Bottom row: final
        axes[1, j].imshow(sim.frames[-1], cmap="magma", vmin=0, vmax=1)
        axes[1, j].set_title(f"recovery={r['recovery']:.2f}, mass={r['mass_frac']:.2f}",
                              fontsize=10)
        axes[1, j].grid(False); axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
    fig.suptitle("Pattern completion: half-zeroed Orbium → does it regrow?",
                 fontsize=13, fontweight="medium")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_cross_substrate_figure(cross_out: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(ORDER), figsize=(3 + 2 * len(ORDER), 3.5),
                               constrained_layout=True)
    for j, code in enumerate(ORDER):
        if code not in cross_out: continue
        ax = axes[j] if len(ORDER) > 1 else axes
        ax.imshow(cross_out[code]["sim"].frames[-1], cmap="magma", vmin=0, vmax=1)
        survived = cross_out[code]["survived"]
        color = "#55A868" if survived else "#C44E52"
        ax.set_title(
            f"{code} in Orbium params\n"
            f"{'✓' if survived else '✗'} mass {cross_out[code]['initial_mass']:.0f}→"
            f"{cross_out[code]['final_mass']:.0f}",
            fontsize=10, color=color, fontweight="bold")
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color); spine.set_linewidth(2.0)
    fig.suptitle("Cross-substrate stability — zoo creatures in Orbium's params (R=13, μ=0.15, σ=0.015)",
                 fontsize=12, fontweight="medium")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # (1) Basin of attraction per creature
    print(f"[lenia-memory] basin-of-attraction: {len(ORDER)} creatures × "
          f"{len(NOISE_LEVELS)} noise levels × {N_SEEDS} seeds = "
          f"{len(ORDER) * len(NOISE_LEVELS) * N_SEEDS} runs")
    t0 = time.time()
    results = {}
    for code in ORDER:
        t1 = time.time()
        results[code] = basin_experiment(code)
        print(f"  {code:5s} {ZOO[code]['name']:30s} elapsed={time.time() - t1:.1f}s")
    print(f"  total: {time.time() - t0:.1f}s")

    summary = summarize_basins(results)
    print("\n  basin summary (mean recovery):")
    header = "  " + " " * 6 + "  ".join(f"σ={nl:.2f}" for nl in NOISE_LEVELS)
    print(header)
    for code in ORDER:
        if code not in summary: continue
        row = f"  {code:5s} " + "  ".join(f"{summary[code][nl]['mean']:>6.3f}"
                                            for nl in NOISE_LEVELS)
        print(row)

    print("\n[lenia-memory] rendering basin figure ...")
    make_basin_figure(results, summary, FIG_DIR / "basin.png")

    # (2) Pattern completion
    print("\n[lenia-memory] pattern completion (Orbium)...")
    completion_out, canonical_final = pattern_completion_experiment("O2u")
    for name, r in completion_out.items():
        print(f"  {name:24s} recovery={r['recovery']:.3f}  mass={r['mass_frac']:.2f}")
    make_completion_figure(completion_out, canonical_final, FIG_DIR / "completion.png")

    # (3) Cross-substrate stability
    print("\n[lenia-memory] cross-substrate stability...")
    cross_out = cross_substrate_experiment()
    for code in ORDER:
        if code not in cross_out: continue
        print(f"  {code:5s} {'survived' if cross_out[code]['survived'] else 'died':>9s}  "
              f"mass {cross_out[code]['initial_mass']:6.2f}→{cross_out[code]['final_mass']:6.2f}")
    make_cross_substrate_figure(cross_out, FIG_DIR / "cross_substrate.png")

    # Persist summary
    summary_out = {
        "basin_summary": {code: {f"sigma_{nl}": v for nl, v in s.items()}
                            for code, s in summary.items()},
        "pattern_completion": {n: {"recovery": r["recovery"], "mass_frac": r["mass_frac"]}
                                 for n, r in completion_out.items()},
        "cross_substrate": {code: {"survived": bool(cross_out[code]["survived"]),
                                     "initial_mass": cross_out[code]["initial_mass"],
                                     "final_mass": cross_out[code]["final_mass"]}
                              for code in ORDER if code in cross_out},
    }
    (HERE / "summary.json").write_text(json.dumps(summary_out, indent=2))
    print("\n[lenia-memory] wrote summary.json")
    print("[lenia-memory] done.")
