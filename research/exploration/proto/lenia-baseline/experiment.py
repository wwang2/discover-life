"""Lenia baseline — reproduce the canonical Orbium glider (Chan 2019).

A minimal, self-contained NumPy implementation of single-channel 2D Lenia.
Decodes Bert Chan's RLE-encoded Orbium initial condition (from his
Chakazul/Lenia animals.json) and runs it on a 128x128 toroidal grid with
FFT-based kernel convolution.

Reference: Chan, B. W.-C. "Lenia: Biology of Artificial Life" (Complex
Systems 28(3), 2019; arXiv:1812.05433).

This is an explore-phase prototype, not a campaign deliverable. Outputs:
  figures/results.png  — multi-panel: kernel, growth, mass/COM, snapshots
  figures/behavior.gif — animated glider
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import gridspec

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Load Orbium unicaudatus directly from Chan's animals.json (bundled).
_ORBIUM = json.loads((HERE / "orbium.json").read_text())
ORBIUM_RLE = _ORBIUM["cells"]
ORBIUM_PARAMS = {
    "R": _ORBIUM["params"]["R"],
    "T": _ORBIUM["params"]["T"],
    "m": _ORBIUM["params"]["m"],
    "s": _ORBIUM["params"]["s"],
    "b": [1.0],  # Chan's "b": "1" → single ring
}


# ---------------------------------------------------------------------------
# RLE decoder (1:1 port of Bert Chan's `rle2cells` for 2D, no multi-channel).
# ---------------------------------------------------------------------------

def _ch2val(c: str) -> int:
    if c in ".b":
        return 0
    if c == "o":
        return 255
    if len(c) == 1:
        return ord(c) - ord("A") + 1
    return (ord(c[0]) - ord("p")) * 24 + (ord(c[1]) - ord("A") + 25)


def decode_rle_2d(rle: str) -> np.ndarray:
    """Decode Chan's 2D-RLE encoding into a [0, 1] float grid."""
    rows: list[list[float]] = []
    current: list[float] = []
    last = ""
    count = ""
    rle = rle.rstrip("!")
    for ch in rle:
        if ch.isdigit():
            count += ch
        elif ch in "pqrstuvwxy@":
            last = ch
        elif ch == "$":
            rows.append(current)
            current = []
            last, count = "", ""
        else:
            val = _ch2val(last + ch) / 255.0
            n = int(count) if count else 1
            current.extend([val] * n)
            last, count = "", ""
    if current:
        rows.append(current)
    max_w = max(len(r) for r in rows)
    arr = np.zeros((len(rows), max_w), dtype=np.float64)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    return arr


# ---------------------------------------------------------------------------
# Lenia kernel + growth function.
# ---------------------------------------------------------------------------

def kernel_core(r: np.ndarray) -> np.ndarray:
    """Polynomial 'quad4' kernel core (Chan 2019, kn=1): (4*r*(1-r))^4 for 0<r<1, else 0."""
    return ((r > 0) & (r < 1)) * (4 * r * (1 - r)) ** 4


def make_kernel(R: int, b: list[float], size: int) -> np.ndarray:
    """Annular polynomial kernel K, normalized to sum 1, placed on a `size`x`size` grid."""
    y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
    r = np.sqrt(x * x + y * y) / R
    Bn = len(b)
    Br = r * Bn
    band = np.minimum(Br.astype(int), Bn - 1)
    inner_r = Br - band  # equivalent to (r*Bn) % 1 for r < 1
    K = (r < 1) * np.array(b)[band] * kernel_core(inner_r)
    K /= K.sum()
    return K


def growth(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Polynomial 'quad4' growth (Chan 2019, gn=1): 2 * max(0, 1 - (u-m)^2/(9s^2))^4 - 1."""
    return 2 * np.maximum(0, 1 - (u - mu) ** 2 / (9 * sigma ** 2)) ** 4 - 1


def fft_convolve(A: np.ndarray, K_fft: np.ndarray) -> np.ndarray:
    """Toroidal convolution via FFT; K_fft is the precomputed conjugate-shifted kernel FFT."""
    return np.real(np.fft.ifft2(np.fft.fft2(A) * K_fft))


# ---------------------------------------------------------------------------
# Run the simulation.
# ---------------------------------------------------------------------------

def run_simulation(steps: int = 200, world_size: int = 128) -> dict:
    """Run Orbium for `steps` Lenia timesteps. Returns frames, mass, center-of-mass."""
    p = ORBIUM_PARAMS
    R, T, mu, sigma = p["R"], p["T"], p["m"], p["s"]
    dt = 1.0 / T

    K = make_kernel(R, p["b"], world_size)
    # Precompute conjugated/shifted FFT so convolution is a centered op.
    K_fft = np.fft.fft2(np.fft.ifftshift(K))

    orbium = decode_rle_2d(ORBIUM_RLE)
    A = np.zeros((world_size, world_size))
    h, w = orbium.shape
    cy, cx = world_size // 2 - h // 2, world_size // 2 - w // 2
    A[cy : cy + h, cx : cx + w] = orbium

    frames = [A.copy()]
    masses = [A.sum()]
    ys, xs = np.indices(A.shape)

    def _com_periodic(A: np.ndarray) -> tuple[float, float]:
        """Center-of-mass on a torus via the circular-mean trick."""
        m = A.sum()
        if m <= 1e-9:
            return float("nan"), float("nan")
        theta_y = 2 * np.pi * ys / world_size
        theta_x = 2 * np.pi * xs / world_size
        ay = float(np.arctan2((A * np.sin(theta_y)).sum(), (A * np.cos(theta_y)).sum()))
        ax = float(np.arctan2((A * np.sin(theta_x)).sum(), (A * np.cos(theta_x)).sum()))
        return (ay % (2 * np.pi)) / (2 * np.pi) * world_size, (ax % (2 * np.pi)) / (2 * np.pi) * world_size

    coms_y, coms_x = [], []
    cy0, cx0 = _com_periodic(A)
    coms_y.append(cy0)
    coms_x.append(cx0)

    for _ in range(steps):
        U = fft_convolve(A, K_fft)
        A = np.clip(A + dt * growth(U, mu, sigma), 0.0, 1.0)
        frames.append(A.copy())
        masses.append(A.sum())
        cy, cx = _com_periodic(A)
        coms_y.append(cy)
        coms_x.append(cx)

    # Unwrap the trajectory: detect cross-boundary jumps and add ±world_size to undo.
    def _unwrap(c):
        c = np.asarray(c, dtype=float)
        d = np.diff(c)
        d[d > world_size / 2] -= world_size
        d[d < -world_size / 2] += world_size
        return np.concatenate([[c[0]], c[0] + np.cumsum(d)])

    coms_y = _unwrap(coms_y)
    coms_x = _unwrap(coms_x)

    return {
        "frames": np.array(frames),
        "mass": np.array(masses),
        "com_y": np.array(coms_y),
        "com_x": np.array(coms_x),
        "kernel": K,
        "params": p,
        "world_size": world_size,
    }


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


def make_results_png(out: dict, path: Path) -> None:
    """6-panel figure: kernel, growth, 4 timestep snapshots, mass + COM trajectory."""
    frames = out["frames"]
    K = out["kernel"]
    p = out["params"]
    n_steps = len(frames) - 1
    snap_idx = [0, n_steps // 3, 2 * n_steps // 3, n_steps]

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = gridspec.GridSpec(3, 4, figure=fig,
                           height_ratios=[1, 1, 0.9])

    # (a) Kernel K
    ax_k = fig.add_subplot(gs[0, 0])
    crop = 2 * p["R"] + 4
    cy = K.shape[0] // 2
    Kc = K[cy - crop : cy + crop, cy - crop : cy + crop]
    ax_k.imshow(Kc, cmap="viridis")
    ax_k.set_title("(a) kernel $K$")
    ax_k.grid(False)
    ax_k.set_xticks([])
    ax_k.set_yticks([])

    # (b) Growth function G(u)
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

    # (c) Mass over time
    ax_m = fig.add_subplot(gs[0, 2])
    t = np.arange(len(out["mass"])) / p["T"]
    ax_m.plot(t, out["mass"], color=COLORS["primary"], lw=2)
    ax_m.set_xlabel("time (Lenia units)")
    ax_m.set_ylabel(r"total mass $\sum A$")
    ax_m.set_title("(c) mass conservation")

    # (d) Center-of-mass trajectory
    ax_c = fig.add_subplot(gs[0, 3])
    ax_c.plot(out["com_x"], out["com_y"], color=COLORS["secondary"], lw=2)
    ax_c.scatter([out["com_x"][0]], [out["com_y"][0]], color=COLORS["primary"],
                 s=40, zorder=3, label="start")
    ax_c.scatter([out["com_x"][-1]], [out["com_y"][-1]], color=COLORS["accent"],
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
        ax.set_title(f"({label}) $t = {idx/p['T']:.1f}$")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Lenia baseline — Orbium glider (Chan 2019 params)",
                 fontsize=14, fontweight="medium")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_behavior_gif(out: dict, path: Path, fps: int = 20) -> None:
    frames = out["frames"]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(frames[0], cmap="magma", vmin=0, vmax=1, animated=True)
    ax.set_title("Orbium — Lenia baseline")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        im.set_array(frames[i])
        return [im]

    # Subsample to keep gif small (~1.5MB target)
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
    out = run_simulation(steps=200, world_size=128)
    dt = time.time() - t0
    print(f"[lenia-baseline] simulation done in {dt:.2f}s")

    n = len(out["frames"]) - 1
    print(f"[lenia-baseline] initial mass = {out['mass'][0]:.4f}")
    print(f"[lenia-baseline] final mass   = {out['mass'][-1]:.4f}  "
          f"(Δ = {out['mass'][-1] - out['mass'][0]:+.2e})")
    dx = out["com_x"][-1] - out["com_x"][0]
    dy = out["com_y"][-1] - out["com_y"][0]
    speed = np.sqrt(dx ** 2 + dy ** 2) / (n / out["params"]["T"])
    print(f"[lenia-baseline] glider net displacement: ({dx:+.2f}, {dy:+.2f}) px")
    print(f"[lenia-baseline] mean speed: {speed:.3f} px / Lenia time-unit")

    print("[lenia-baseline] saving results.png ...")
    make_results_png(out, FIG_DIR / "results.png")
    print("[lenia-baseline] saving behavior.gif ...")
    make_behavior_gif(out, FIG_DIR / "behavior.gif")
    print("[lenia-baseline] done.")
