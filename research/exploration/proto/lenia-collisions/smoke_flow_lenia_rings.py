"""Two-ring Flow-Lenia test — do stationary rings interact at varying separations?

Since Orbium-parameter Flow-Lenia produces stationary ring solitons (not gliders),
the analogous 'collision' test in Flow-Lenia is to place two rings at different
center-to-center separations and see whether they (a) coexist independently,
(b) repel, (c) attract & merge, (d) attract & spawn extra rings."""
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
sys.path.insert(0, str(HERE.parent))
from flow_lenia import FlowLeniaConfig, run_flow_lenia  # noqa: E402
from lenia import decode_rle_2d, place_pattern  # noqa: E402

ORBIUM = json.loads((HERE.parent / "lenia-baseline" / "orbium.json").read_text())
ORBIUM_PATTERN = decode_rle_2d(ORBIUM["cells"])

W = 192
cfg = FlowLeniaConfig(R=13, m=0.15, s=0.015, h=1.0, dt=0.2, dd=5, sigma=0.65)
STEPS = 100

# Vary separation along x
separations_px = [30, 50, 70, 90]
print(f"[rings] Flow-Lenia config: {cfg}")
results = []
for sep in separations_px:
    A0 = np.zeros((W, W))
    A0 = place_pattern(A0, ORBIUM_PATTERN, (W // 2, W // 2 - sep // 2), angle_deg=0.0)
    A0 = place_pattern(A0, ORBIUM_PATTERN, (W // 2, W // 2 + sep // 2), angle_deg=0.0)
    t0 = time.time()
    out = run_flow_lenia(A0, cfg, steps=STEPS)
    elapsed = time.time() - t0
    # Classify: final connected components > threshold 0.05
    final = out["frames"][-1]
    labels, n_total = cc_label(final > 0.05)
    sizes = sorted([(labels == k).sum() for k in range(1, n_total + 1)
                    if (labels == k).sum() >= 25], reverse=True)
    m0, mf = out["mass"][0], out["mass"][-1]
    results.append({"sep": sep, "out": out, "nc_final": len(sizes), "sizes": sizes,
                     "mass_drift": abs(mf - m0) / m0 if m0 > 0 else 0.0})
    print(f"  sep={sep:3d}px  steps={STEPS} elapsed={elapsed:.1f}s  "
          f"mass {m0:.2f} → {mf:.2f} (Δrel={abs(mf-m0)/m0:.2e})  "
          f"final-rings={len(sizes)}  sizes={sizes}")

# Render figure
n = len(results)
fig = plt.figure(figsize=(3 + 2.4 * n, 7), constrained_layout=True)
gs = gridspec.GridSpec(2, n, figure=fig, height_ratios=[1.0, 1.0])
snap_idx_map = {30: [0, 25, 50, 75, 100],
                 50: [0, 25, 50, 75, 100],
                 70: [0, 25, 50, 75, 100],
                 90: [0, 25, 50, 75, 100]}

for i, r in enumerate(results):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(r["out"]["frames"][0], cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"sep={r['sep']}px, t=0", fontsize=10)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, i])
    ax.imshow(r["out"]["frames"][-1], cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"t={STEPS}\nfinal n_rings={r['nc_final']}", fontsize=10)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])

fig.suptitle(f"Flow-Lenia: two Orbium-seeded rings at varying separation",
             fontsize=12, fontweight="medium")
fig.savefig(HERE / "figures" / "flow_rings.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  → {HERE / 'figures' / 'flow_rings.png'}")
