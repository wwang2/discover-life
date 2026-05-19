"""Lenia core simulator (single-channel 2D, polynomial kernel + growth).

Shared by proto/lenia-baseline/ and proto/lenia-zoo/. Reproduces Chan (2019)
with kn=1 (polynomial 'quad4' kernel core (4r(1-r))^4) and gn=1 (polynomial
growth 2*max(0,1-(u-m)^2/(9s^2))^4 - 1). Other Chan kernel/growth families
(kn,gn ∈ {2,3,4}) are not implemented here — every Lenia creature in the
zoo uses kn=gn=1.

Each proto puts this directory on sys.path and imports `from lenia import ...`.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# RLE decoder — 1:1 port of Bert Chan's `rle2cells` for 2D (no multichannel).
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
# Kernel + growth (Chan kn=1, gn=1 — polynomial).
# ---------------------------------------------------------------------------

def kernel_core(r: np.ndarray) -> np.ndarray:
    """(4*r*(1-r))^4 for 0<r<1, else 0."""
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
    """2 * max(0, 1 - (u-m)^2/(9s^2))^4 - 1 — Chan gn=1."""
    return 2 * np.maximum(0, 1 - (u - mu) ** 2 / (9 * sigma ** 2)) ** 4 - 1


# ---------------------------------------------------------------------------
# Simulator.
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    frames: np.ndarray         # (T+1, H, W)
    mass: np.ndarray            # (T+1,)
    com_y: np.ndarray           # (T+1,), unwrapped on the torus
    com_x: np.ndarray
    kernel: np.ndarray          # (H, W)
    params: dict
    world_size: int


def _com_periodic(A: np.ndarray, ys: np.ndarray, xs: np.ndarray, W: int) -> tuple[float, float]:
    """Center-of-mass on a torus via the circular-mean trick."""
    m = A.sum()
    if m <= 1e-9:
        return float("nan"), float("nan")
    theta_y = 2 * np.pi * ys / W
    theta_x = 2 * np.pi * xs / W
    ay = float(np.arctan2((A * np.sin(theta_y)).sum(), (A * np.cos(theta_y)).sum()))
    ax = float(np.arctan2((A * np.sin(theta_x)).sum(), (A * np.cos(theta_x)).sum()))
    return (ay % (2 * np.pi)) / (2 * np.pi) * W, (ax % (2 * np.pi)) / (2 * np.pi) * W


def _unwrap_periodic(c: np.ndarray, W: int) -> np.ndarray:
    """Undo torus wraparound jumps in a 1-D trajectory."""
    c = np.asarray(c, dtype=float)
    d = np.diff(c)
    d[d > W / 2] -= W
    d[d < -W / 2] += W
    return np.concatenate([[c[0]], c[0] + np.cumsum(d)])


def run_simulation(
    creature: dict,
    *,
    steps: int = 200,
    world_size: int = 128,
    keep_every: int = 1,
) -> SimResult:
    """Run a creature (dict with 'params' and 'cells') for `steps` Lenia timesteps.

    `keep_every`: store every Nth frame (default every frame). Useful when running
    long simulations and animation memory matters.
    """
    p = creature["params"]
    R, T, mu, sigma = p["R"], p["T"], p["m"], p["s"]
    b = [1.0]  # Chan's "b": "1" → single ring
    dt = 1.0 / T

    K = make_kernel(R, b, world_size)
    K_fft = np.fft.fft2(np.fft.ifftshift(K))

    pattern = decode_rle_2d(creature["cells"])
    A = np.zeros((world_size, world_size))
    h, w = pattern.shape
    cy, cx = world_size // 2 - h // 2, world_size // 2 - w // 2
    A[cy : cy + h, cx : cx + w] = pattern

    ys, xs = np.indices(A.shape)
    frames = [A.copy()]
    masses = [A.sum()]
    cy0, cx0 = _com_periodic(A, ys, xs, world_size)
    coms_y = [cy0]
    coms_x = [cx0]

    for step in range(steps):
        U = np.real(np.fft.ifft2(np.fft.fft2(A) * K_fft))
        A = np.clip(A + dt * growth(U, mu, sigma), 0.0, 1.0)
        if (step + 1) % keep_every == 0 or step == steps - 1:
            frames.append(A.copy())
        masses.append(A.sum())
        cy, cx = _com_periodic(A, ys, xs, world_size)
        coms_y.append(cy)
        coms_x.append(cx)

    return SimResult(
        frames=np.array(frames),
        mass=np.array(masses),
        com_y=_unwrap_periodic(coms_y, world_size),
        com_x=_unwrap_periodic(coms_x, world_size),
        kernel=K,
        params=p,
        world_size=world_size,
    )


# ---------------------------------------------------------------------------
# Diagnostics — extracted so both proto/ scripts share them.
# ---------------------------------------------------------------------------

def mass_cv(mass: np.ndarray, skip: int = 30) -> float:
    """Coefficient of variation of mass after a transient. Lower = better conserved."""
    tail = mass[skip:]
    return float(tail.std() / tail.mean()) if tail.mean() > 1e-9 else float("nan")


def locomotion_speed(com_y: np.ndarray, com_x: np.ndarray, T: int, skip: int = 30) -> float:
    """Net displacement / Lenia time-unit, computed on the unwrapped trajectory after transient."""
    tail_t = (len(com_y) - skip - 1) / T
    if tail_t <= 0:
        return 0.0
    dy = com_y[-1] - com_y[skip]
    dx = com_x[-1] - com_x[skip]
    return float(np.sqrt(dy * dy + dx * dx) / tail_t)


def footprint_area(A: np.ndarray, threshold: float = 0.1) -> float:
    return float((A > threshold).sum())


def bilateral_symmetry(A: np.ndarray) -> float:
    """Best-axis bilateral symmetry score: correlation with mirrored self after centering.

    Only checks horizontal / vertical reflections — kept for back-compat. Use
    `dihedral_symmetry` for the rotation-aware version.
    """
    m = A.sum()
    if m <= 1e-9:
        return 0.0
    ys, xs = np.indices(A.shape)
    cy = int(round((A * ys).sum() / m))
    cx = int(round((A * xs).sum() / m))
    H, W = A.shape
    Ac = np.roll(A, (H // 2 - cy, W // 2 - cx), axis=(0, 1))
    flips = [np.fliplr(Ac), np.flipud(Ac)]
    scores = []
    for f in flips:
        num = (Ac * f).sum()
        den = np.sqrt((Ac * Ac).sum() * (f * f).sum())
        scores.append(float(num / den) if den > 0 else 0.0)
    return max(scores)


def _center_on_com(A: np.ndarray) -> np.ndarray:
    m = A.sum()
    if m <= 1e-9:
        return A
    H, W = A.shape
    ys, xs = np.indices((H, W))
    cy = int(round((A * ys).sum() / m))
    cx = int(round((A * xs).sum() / m))
    return np.roll(A, (H // 2 - cy, W // 2 - cx), axis=(0, 1))


def dihedral_symmetry(
    A: np.ndarray,
    *,
    reflection_angles: tuple[float, ...] = (0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0),
    rotation_orders: tuple[int, ...] = (2, 3, 4, 6),
) -> float:
    """Best-axis dihedral symmetry score — rotation-aware extension of bilateral.

    After centering on the COM, returns the max over:
      - reflection self-corr at `reflection_angles` candidate axes
      - rotation self-corr at `rotation_orders` candidate orders

    A creature with D_k symmetry (e.g. Synorbium D4) scores ~1 because rotation
    by 360/k aligns it with itself, even if no axis-aligned bilateral flip does.
    """
    from scipy.ndimage import rotate as _rotate

    m = A.sum()
    if m <= 1e-9:
        return 0.0
    Ac = _center_on_com(A)
    Ac_norm = (Ac * Ac).sum()
    if Ac_norm <= 1e-12:
        return 0.0

    best = 0.0
    for theta in reflection_angles:
        Ar = _rotate(Ac, theta, reshape=False, order=1, mode="constant", cval=0.0)
        flipped = np.fliplr(Ar)
        num = (Ar * flipped).sum()
        den = np.sqrt((Ar * Ar).sum() * (flipped * flipped).sum())
        score = float(num / den) if den > 0 else 0.0
        if score > best:
            best = score

    for n in rotation_orders:
        Ar = _rotate(Ac, 360.0 / n, reshape=False, order=1, mode="constant", cval=0.0)
        num = (Ac * Ar).sum()
        den = np.sqrt(Ac_norm * (Ar * Ar).sum())
        score = float(num / den) if den > 0 else 0.0
        if score > best:
            best = score

    return best


def temporal_complexity(
    frames: np.ndarray, com_y: np.ndarray, com_x: np.ndarray, world_size: int, *, skip: int = 30
) -> float:
    """Pixel-wise std of frames after centering each on its instantaneous COM.

    Captures *internal* dynamics only — pure translation contributes nothing
    because the creature is registered to the centroid before differencing.
    A static blob → 0. A breathing / rotating / undulating creature → > 0.
    """
    if len(frames) <= skip + 1:
        return 0.0
    H = W = world_size
    centered = []
    n_frames = len(frames)
    n_traj = len(com_y)
    # frames may be subsampled (keep_every > 1); map frame index → trajectory index.
    for i, f in enumerate(frames):
        traj_idx = min(int(round(i * (n_traj - 1) / max(1, n_frames - 1))), n_traj - 1)
        cy = com_y[traj_idx] % H
        cx = com_x[traj_idx] % W
        if np.isnan(cy) or np.isnan(cx):
            continue
        shifted = np.roll(f, (H // 2 - int(round(cy)), W // 2 - int(round(cx))), axis=(0, 1))
        centered.append(shifted)
    centered = np.array(centered[max(0, skip // max(1, n_traj // n_frames)):])
    if len(centered) < 2:
        return 0.0
    return float(centered.std(axis=0).mean())


def persistent(mass: np.ndarray, fraction: float = 0.5) -> bool:
    """True iff mass never falls below `fraction` × initial-mass-after-transient."""
    if len(mass) < 10:
        return False
    initial = mass[5:15].mean()
    return bool(mass[15:].min() > fraction * initial)


# ---------------------------------------------------------------------------
# Synthetic creature — static Gaussian blob, for negative-control diagnostics.
# ---------------------------------------------------------------------------

def synthetic_static_blob_sim(
    target_mass: float = 75.0,
    sigma: float = 6.0,
    world_size: int = 160,
    steps: int = 200,
    T: int = 10,
) -> SimResult:
    """Build a fake SimResult: identical Gaussian-blob frames, COM fixed at center.

    Negative control. Mass quasi-conserved (exactly, since the blob never changes).
    Footprint, symmetry are constant. temporal_complexity → 0. Speed → 0.
    """
    H = W = world_size
    ys, xs = np.indices((H, W))
    blob = np.exp(-((xs - W / 2) ** 2 + (ys - H / 2) ** 2) / (2 * sigma ** 2))
    blob = blob * (target_mass / blob.sum())
    blob = np.clip(blob, 0.0, 1.0)

    frames = np.tile(blob[None, :, :], (steps + 1, 1, 1))
    mass = np.full(steps + 1, blob.sum())
    com = np.full(steps + 1, world_size / 2)
    return SimResult(
        frames=frames,
        mass=mass,
        com_y=com,
        com_x=com,
        kernel=np.zeros((world_size, world_size)),
        params={"R": 0, "T": T, "m": 0.0, "s": 0.0},
        world_size=world_size,
    )
