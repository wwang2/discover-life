"""Quick smoke test: does our minimal NumPy Flow-Lenia conserve mass and produce
non-trivial dynamics on (a) a Gaussian blob and (b) the canonical Orbium seed?
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
sys.path.insert(0, str(HERE.parent))
from flow_lenia import FlowLeniaConfig, run_flow_lenia  # noqa: E402
from lenia import decode_rle_2d, place_pattern  # noqa: E402

ORBIUM = json.loads((HERE.parent / "lenia-baseline" / "orbium.json").read_text())

W = 128
cfg = FlowLeniaConfig(R=13, m=0.15, s=0.015, h=1.0, dt=0.2, dd=5, sigma=0.65)

# --- (a) Gaussian blob ---
ys, xs = np.indices((W, W))
A0_blob = 0.7 * np.exp(-((xs - W / 2) ** 2 + (ys - W / 2) ** 2) / (2 * 8.0 ** 2))

# --- (b) Orbium seed ---
A0_orb = np.zeros((W, W))
A0_orb = place_pattern(A0_orb, decode_rle_2d(ORBIUM["cells"]), (W // 2, W // 2), angle_deg=0.0)

cases = [("blob", A0_blob), ("orbium", A0_orb)]
print(f"[smoke] Flow-Lenia config: {cfg}")
for name, A0 in cases:
    t0 = time.time()
    out = run_flow_lenia(A0, cfg, steps=80)
    elapsed = time.time() - t0
    m0 = out["mass"][0]
    m1 = out["mass"][-1]
    print(f"  {name:8s} steps=80 elapsed={elapsed:.1f}s  "
          f"mass {m0:.3f} → {m1:.3f}  (Δ={m1-m0:+.2e}, rel={abs(m1-m0)/m0:.2e})")
    # 1 figure per case
    fig = plt.figure(figsize=(11, 3), constrained_layout=True)
    gs = gridspec.GridSpec(1, 5, figure=fig)
    snap_idx = [0, 20, 40, 60, 80]
    for j, idx in enumerate(snap_idx):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(out["frames"][idx], cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"$t={idx}$  m={out['mass'][idx]:.1f}", fontsize=9)
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Flow-Lenia smoke: {name}", fontsize=11, fontweight="medium")
    out_path = HERE / "figures" / f"flow_smoke_{name}.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")
