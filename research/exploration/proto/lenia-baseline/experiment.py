"""Lenia baseline — reproduce the canonical Orbium glider (Chan 2019).

Uses the shared core simulator at `proto/lenia.py`. Single-channel 2D Lenia
with polynomial kernel + growth (Chan kn=gn=1), Orbium *unicaudatus*
parameters from his Chakazul/Lenia animals.json.

Reference: Chan, B. W.-C. "Lenia: Biology of Artificial Life" (Complex
Systems 28(3), 2019; arXiv:1812.05433).

Outputs:
  figures/results.png  — multi-panel: kernel, growth, mass/COM, snapshots
  figures/behavior.gif — animated glider
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import gridspec

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE.parent))
from lenia import run_simulation, growth  # noqa: E402

ORBIUM = json.loads((HERE / "orbium.json").read_text())


# ---------------------------------------------------------------------------
# Figures.
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
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

COLORS = {
    "primary":   "#4C72B0",
    "secondary": "#DD8452",
    "tertiary":  "#55A868",
    "accent":    "#C44E52",
    "muted":     "#888888",
}


def make_results_png(out, path: Path) -> None:
    """6-panel figure: kernel, growth, mass, COM trajectory, four timestep snapshots."""
    frames = out.frames
    K = out.kernel
    p = out.params
    n_steps = len(frames) - 1
    snap_idx = [0, n_steps // 3, 2 * n_steps // 3, n_steps]

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.9])

    # (a) Kernel
    ax_k = fig.add_subplot(gs[0, 0])
    crop = 2 * p["R"] + 4
    cy = K.shape[0] // 2
    Kc = K[cy - crop : cy + crop, cy - crop : cy + crop]
    ax_k.imshow(Kc, cmap="viridis")
    ax_k.set_title("(a) kernel $K$")
    ax_k.grid(False)
    ax_k.set_xticks([])
    ax_k.set_yticks([])

    # (b) Growth
    ax_g = fig.add_subplot(gs[0, 1])
    u = np.linspace(0, 0.4, 400)
    g = growth(u, p["m"], p["s"])
    ax_g.plot(u, g, color=COLORS["primary"], lw=2)
    ax_g.axhline(0, color=COLORS["muted"], lw=0.6, linestyle="--")
    ax_g.axvline(p["m"], color=COLORS["accent"], lw=0.8, linestyle=":",
                 label=fr"$\mu={p['m']}$")
    ax_g.set_xlabel("$u$ (kernel-weighted neighborhood)")
    ax_g.set_ylabel("growth $G(u)$")
    ax_g.set_title("(b) growth function")
    ax_g.legend(loc="lower left")
    ax_g.set_xlim(0, 0.35)

    # (c) Mass
    ax_m = fig.add_subplot(gs[0, 2])
    t = np.arange(len(out.mass)) / p["T"]
    ax_m.plot(t, out.mass, color=COLORS["primary"], lw=2)
    ax_m.set_xlabel("time (Lenia units)")
    ax_m.set_ylabel(r"total mass $\sum A$")
    ax_m.set_title("(c) mass conservation")

    # (d) COM trajectory
    ax_c = fig.add_subplot(gs[0, 3])
    ax_c.plot(out.com_x, out.com_y, color=COLORS["secondary"], lw=2)
    ax_c.scatter([out.com_x[0]], [out.com_y[0]], color=COLORS["primary"],
                 s=40, zorder=3, label="start")
    ax_c.scatter([out.com_x[-1]], [out.com_y[-1]], color=COLORS["accent"],
                 s=40, zorder=3, label="end")
    ax_c.set_xlabel("center-of-mass $x$ (px)")
    ax_c.set_ylabel("center-of-mass $y$ (px)")
    ax_c.set_title("(d) glider trajectory")
    ax_c.legend(loc="best", fontsize=9)
    ax_c.set_aspect("equal", adjustable="datalim")

    # (e–h) Four snapshots
    for i, idx in enumerate(snap_idx):
        ax = fig.add_subplot(gs[1:, i])
        ax.imshow(frames[idx], cmap="magma", vmin=0, vmax=1)
        label = "efgh"[i]
        ax.set_title(f"({label}) $t = {idx / p['T']:.1f}$")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Lenia baseline — Orbium glider (Chan 2019 params)",
                 fontsize=14, fontweight="medium")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_behavior_gif(out, path: Path, fps: int = 20) -> None:
    frames = out.frames
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(frames[0], cmap="magma", vmin=0, vmax=1, animated=True)
    ax.set_title("Orbium — Lenia baseline")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        im.set_array(frames[i])
        return [im]

    stride = max(1, len(frames) // 100)
    indices = list(range(0, len(frames), stride))
    anim = FuncAnimation(fig, update, frames=indices, interval=1000 / fps, blit=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("[lenia-baseline] running Orbium for 200 steps on 128x128 grid...")
    t0 = time.time()
    out = run_simulation(ORBIUM, steps=200, world_size=128)
    print(f"[lenia-baseline] simulation done in {time.time() - t0:.2f}s")

    n = len(out.frames) - 1
    print(f"[lenia-baseline] initial mass = {out.mass[0]:.4f}")
    print(f"[lenia-baseline] final mass   = {out.mass[-1]:.4f}  "
          f"(Δ = {out.mass[-1] - out.mass[0]:+.2e})")
    dx = out.com_x[-1] - out.com_x[0]
    dy = out.com_y[-1] - out.com_y[0]
    speed = np.sqrt(dx ** 2 + dy ** 2) / (n / out.params["T"])
    print(f"[lenia-baseline] glider net displacement: ({dx:+.2f}, {dy:+.2f}) px")
    print(f"[lenia-baseline] mean speed: {speed:.3f} px / Lenia time-unit")

    print("[lenia-baseline] saving results.png ...")
    make_results_png(out, FIG_DIR / "results.png")
    print("[lenia-baseline] saving behavior.gif ...")
    make_behavior_gif(out, FIG_DIR / "behavior.gif")
    print("[lenia-baseline] done.")
