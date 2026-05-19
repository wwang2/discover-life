"""Minimal NumPy port of Flow-Lenia (Plantec et al. 2022, ALife 2023 Best Paper).

A 1:1 port of the single-channel ('C=1', no parameter embedding) version of the
JAX reference at github.com/erwanplantec/FlowLenia — enough to test whether the
mass-conservative advection update strengthens or breaks the soliton-collision
instruction set we found in vanilla Lenia (`proto/lenia-collisions/`).

Mathematical core (per `flowlenia/flowlenia.py:__call__` and
`flowlenia/reintegration_tracking.py`):

    U  = K * A                               # standard Lenia kernel pass
    G  = h · (2·bell(U; μ, σ) − 1)            # Lenia growth potential
    F  = ∇G · (1 − α) − ∇A · α                # force/velocity field
         where α = clip(A², 0, 1)
    μ(x)   = x + dt · F(x)                    # advection target for cell x
    A'(x)  = Σ_{(dx,dy) ∈ [-dd, dd]²} roll(A, dx, dy) · overlap( pos(x), roll(μ, dx, dy) )

The `overlap` is the area of intersection between a unit square at `pos(x)` and a
unit square at the rolled target — a partition-of-unity that makes the total-mass
sum invariant under the update (mass-conservative advection by construction).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import sobel


# ---------------------------------------------------------------------------
# Kernel + growth — same forms as vanilla Lenia. Reuse our existing functions.
# ---------------------------------------------------------------------------

def _bell(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def _polynomial_kernel_core(r: np.ndarray) -> np.ndarray:
    return ((r > 0) & (r < 1)) * (4 * r * (1 - r)) ** 4


def make_kernel(R: int, size: int, kn: int = 1) -> np.ndarray:
    """Bell-annular kernel (single ring, b=[1]). kn=1: polynomial; kn=3: Gaussian."""
    y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
    r = np.sqrt(x * x + y * y) / R
    if kn == 1:
        K = (r < 1) * _polynomial_kernel_core(r)
    elif kn == 3:
        K = (r < 1) * _bell(r, 0.5, 0.15)
    else:
        raise ValueError(f"kn must be 1 or 3, got {kn}")
    K /= K.sum()
    return K


def growth_polynomial(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 2 * np.maximum(0, 1 - (u - mu) ** 2 / (9 * sigma ** 2)) ** 4 - 1


def growth_gaussian(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 2 * _bell(u, mu, sigma) - 1


# ---------------------------------------------------------------------------
# Reintegration tracking (the heart of Flow-Lenia).
# ---------------------------------------------------------------------------

def reintegrate(A: np.ndarray, F: np.ndarray, *, dt: float, dd: int, sigma: float) -> np.ndarray:
    """Mass-conservative advection of A according to force field F.

    F has shape (H, W, 2) — F[..., 0] is the y-component, F[..., 1] is x-component.
    dd is the neighborhood half-width (typically 5). sigma is the spread of the source
    distribution (typically 0.65). Torus boundary.
    """
    H, W = A.shape
    Y, X = np.indices((H, W))
    pos = np.stack([Y, X], axis=-1) + 0.5  # cell-center coords, (H, W, 2)

    # Cap the flow magnitude so cells don't try to advect farther than dd
    ma = dd - sigma
    F_clipped = np.clip(dt * F, -ma, ma)
    mu = pos + F_clipped  # target positions, (H, W, 2)

    out = np.zeros_like(A)
    for dx in range(-dd, dd + 1):
        for dy in range(-dd, dd + 1):
            Ar = np.roll(A, (dy, dx), axis=(0, 1))
            mur = np.roll(mu, (dy, dx), axis=(0, 1))
            # Toroidal distance (Plantec uses min over 3×3 lattice shifts)
            dy_abs = np.minimum.reduce([
                np.abs(pos[..., 0] - (mur[..., 0] + di)) for di in (-H, 0, H)
            ])
            dx_abs = np.minimum.reduce([
                np.abs(pos[..., 1] - (mur[..., 1] + dj)) for dj in (-W, 0, W)
            ])
            sz_y = 0.5 - dy_abs + sigma
            sz_x = 0.5 - dx_abs + sigma
            cap = min(1.0, 2.0 * sigma)
            area = (np.clip(sz_y, 0, cap) * np.clip(sz_x, 0, cap)) / (4.0 * sigma ** 2)
            out += Ar * area
    return out


# ---------------------------------------------------------------------------
# Flow-Lenia config + step.
# ---------------------------------------------------------------------------

@dataclass
class FlowLeniaConfig:
    R: int = 13            # kernel radius
    m: float = 0.15         # growth mu
    s: float = 0.015        # growth sigma
    h: float = 1.0          # growth scale
    dt: float = 0.2         # advection timestep
    dd: int = 5             # reintegration neighborhood half-width
    sigma: float = 0.65     # source-distribution spread
    growth_fn: str = "polynomial"  # 'polynomial' (Chan gn=1) or 'gaussian'


def flow_lenia_step(A: np.ndarray, K: np.ndarray, cfg: FlowLeniaConfig) -> np.ndarray:
    """One Flow-Lenia step."""
    # 1. Lenia kernel pass — kernel-weighted neighborhood
    K_fft = np.fft.fft2(np.fft.ifftshift(K))
    U = np.real(np.fft.ifft2(np.fft.fft2(A) * K_fft))

    # 2. Growth potential
    if cfg.growth_fn == "polynomial":
        G = cfg.h * growth_polynomial(U, cfg.m, cfg.s)
    else:
        G = cfg.h * growth_gaussian(U, cfg.m, cfg.s)

    # 3. Gradients via Sobel (axis order: y, x)
    grad_G_y = sobel(G, axis=0, mode="wrap") / 8.0
    grad_G_x = sobel(G, axis=1, mode="wrap") / 8.0
    grad_A_y = sobel(A, axis=0, mode="wrap") / 8.0
    grad_A_x = sobel(A, axis=1, mode="wrap") / 8.0

    # 4. Blend coefficient (Plantec eq.) — α ∈ [0, 1]
    alpha = np.clip(A ** 2, 0.0, 1.0)

    # 5. Force field F = ∇G·(1−α) − ∇A·α
    F = np.stack([
        grad_G_y * (1 - alpha) - grad_A_y * alpha,
        grad_G_x * (1 - alpha) - grad_A_x * alpha,
    ], axis=-1)

    # 6. Reintegration-tracking advection
    return reintegrate(A, F, dt=cfg.dt, dd=cfg.dd, sigma=cfg.sigma)


def run_flow_lenia(A0: np.ndarray, cfg: FlowLeniaConfig, *, steps: int = 100,
                    keep_every: int = 1) -> dict:
    """Run for `steps` steps. Returns dict with frames + mass + COM(y,x) trajectory."""
    K = make_kernel(cfg.R, A0.shape[0])
    A = A0.copy()
    frames = [A.copy()]
    masses = [A.sum()]
    H, W = A.shape
    ys, xs = np.indices((H, W))

    def _com():
        m = A.sum()
        if m <= 1e-9:
            return float("nan"), float("nan")
        ty = 2 * np.pi * ys / H
        tx = 2 * np.pi * xs / W
        ay = np.arctan2((A * np.sin(ty)).sum(), (A * np.cos(ty)).sum())
        ax = np.arctan2((A * np.sin(tx)).sum(), (A * np.cos(tx)).sum())
        return (ay % (2 * np.pi)) / (2 * np.pi) * H, (ax % (2 * np.pi)) / (2 * np.pi) * W

    cy, cx = _com()
    coms_y = [cy]
    coms_x = [cx]
    for step in range(steps):
        A = flow_lenia_step(A, K, cfg)
        if (step + 1) % keep_every == 0 or step == steps - 1:
            frames.append(A.copy())
        masses.append(A.sum())
        cy, cx = _com()
        coms_y.append(cy)
        coms_x.append(cx)

    return {
        "frames": np.array(frames),
        "mass": np.array(masses),
        "com_y": np.array(coms_y),
        "com_x": np.array(coms_x),
        "kernel": K,
        "cfg": cfg,
    }
