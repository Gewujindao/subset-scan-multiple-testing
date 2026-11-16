"""Summary figure for the README: type-I inflation under ignored weak correlation,
and type-I calibration of the whitened full-subset test.

Run: python make_figures.py   (writes summary.png next to this script)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY = "#1f4e79"
GREY = "#9aa5b1"
INK = "#222222"
MUTED = "#5f6b76"
AXIS = "#444444"
GRID = "#e8ebee"

FS_TITLE, FS_MAIN, FS_SMALL = 13, 10, 8.5

# Panel a: equicorrelation rho = 0.01, alpha = 1e-3, critical values from the independence model.
# Only K = 300 is present in code/w2/res/.
K = 300  # code/w2/res/o_eq.txt, line 1
INFL_FULL = 23.58  # code/w2/res/o_eq.txt, line 1 ("alpha=0.001 rho=0.01: full 23.58x")
INFL_FAST = 1.67  # code/w2/res/o_eq.txt, line 1 ("alpha=0.001 rho=0.01: ... fast 1.67x")

# Panel b: observed / nominal type-I ratio after whitening, random estimation error only (b = 0).
# (config, full_W at 1e-3, full_W at 1e-4, Wsplit at 1e-3, Wsplit at 1e-4)
WHITENED = [
    ("K=30 A M=1e4", 1.01, 0.95, 0.99, 0.98),     # code/w3/res/summary.txt, lines 6 and 8
    ("K=30 A M=4e4", 1.00, 0.92, 1.01, 1.00),     # code/w3/res/summary.txt, lines 13 and 15
    ("K=30 B M=1e4", 1.02, 0.96, 1.01, 1.12),     # code/w3/res/summary.txt, lines 27 and 29
    ("K=30 B M=4e4", 0.98, 1.06, 0.96, 1.13),     # code/w3/res/summary.txt, lines 34 and 36
    ("K=30 I M=1e4", 1.01, 1.04, 0.99, 1.10),     # code/w3/res/summary.txt, lines 48 and 50
    ("K=116 A M=1e4", 0.97, 1.12, 0.99, 0.95),    # code/w3/res/summary.txt, lines 55 and 57
    ("K=116 A M=4e4", 0.99, 1.10, 0.99, 0.90),    # code/w3/res/summary.txt, lines 62 and 64
    ("K=116 B M=1e4", 0.99, 1.08, 1.01, 0.98),    # code/w3/res/summary.txt, lines 76 and 78
    ("K=116 B M=4e4", 1.02, 1.01, 1.03, 0.98),    # code/w3/res/summary.txt, lines 83 and 85
    ("K=300 A M=1e4", 1.05, 0.96, 1.05, 1.06),    # code/w3/res/summary.txt, lines 97 and 99
]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FS_SMALL,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "axes.linewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

fig, (axa, axb) = plt.subplots(1, 2, figsize=(8, 4.5), dpi=200,
                               gridspec_kw={"width_ratios": [1, 1.55]})
fig.subplots_adjust(left=0.09, right=0.985, top=0.75, bottom=0.17, wspace=0.28)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, which="major")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=FS_SMALL, colors=AXIS)
    ax.tick_params(axis="x", length=0, pad=5, labelsize=FS_SMALL, labelcolor=INK)


# Panel a: inflation factor at K = 300 (log scale, bars start at 1x = nominal)
xs = [0, 1]
vals = [INFL_FULL, INFL_FAST]
axa.bar(xs, [v - 1 for v in vals], bottom=1, width=0.55, color=GREY, edgecolor="none", zorder=2)
for x, v in zip(xs, vals):
    axa.text(x, v * 1.12, f"{v:.2f}×", ha="center", va="bottom", fontsize=FS_MAIN, color=INK)
axa.set_yscale("log")
axa.set_ylim(0.7, 100)
axa.set_yticks([1, 10, 100])
axa.set_yticklabels(["1×", "10×", "100×"])
axa.minorticks_off()
axa.axhline(1, color=AXIS, linewidth=0.7, zorder=1)
axa.set_xticks(xs)
axa.set_xticklabels(["Full search", "Pre-screened"])
axa.set_xlim(-0.6, 1.6)
axa.set_ylabel("Type-I error / nominal", fontsize=FS_MAIN, color=INK)
axa.set_title(f"Ignored correlation, K = {K:,}", loc="left", fontsize=FS_MAIN, color=INK, pad=8)
axa.set_xlabel("\u03c1 = 0.01, \u03b1 = 0.001, independence-model critical values",
               fontsize=FS_SMALL, color=MUTED, labelpad=8)
style(axa)

# Panel b: whitened test, observed / nominal ratio
cols = [("Full search\n\u03b1 = 0.001", 1), ("Full search\n\u03b1 = 0.0001", 2),
        ("W-split\n\u03b1 = 0.001", 3), ("W-split\n\u03b1 = 0.0001", 4)]
offsets = np.linspace(-0.22, 0.22, len(WHITENED))
for ci, (_, idx) in enumerate(cols):
    ys = [row[idx] for row in WHITENED]
    axb.scatter(ci + offsets, ys, s=22, color=NAVY, zorder=3, linewidths=0)
axb.axhline(1.0, color=AXIS, linewidth=0.7, zorder=2)
all_vals = [v for row in WHITENED for v in row[1:]]
lo, hi = min(all_vals), max(all_vals)
lo_i = next((ci, j) for ci, (_, idx) in enumerate(cols) for j, row in enumerate(WHITENED) if row[idx] == lo)
hi_i = next((ci, j) for ci, (_, idx) in enumerate(cols) for j, row in enumerate(WHITENED) if row[idx] == hi)
axb.annotate(f"min {lo:.2f}", (lo_i[0] + offsets[lo_i[1]], lo), xytext=(-7, 0),
             textcoords="offset points", ha="right", va="center", fontsize=FS_SMALL, color=INK)
axb.annotate(f"max {hi:.2f}", (hi_i[0] + offsets[hi_i[1]], hi), xytext=(-7, 0),
             textcoords="offset points", ha="right", va="center", fontsize=FS_SMALL, color=INK)
axb.set_xticks(range(len(cols)))
axb.set_xticklabels([c[0] for c in cols])
axb.set_xlim(-0.5, 3.5)
axb.set_ylim(0.8, 1.2)
axb.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
axb.set_yticklabels([f"{t:.2f}" for t in [0.8, 0.9, 1.0, 1.1, 1.2]])
axb.set_title("Whitened test, estimated correlation", loc="left", fontsize=FS_MAIN, color=INK, pad=8)
axb.set_xlabel("10 configurations per column", fontsize=FS_SMALL, color=MUTED, labelpad=8)
style(axb)

# Panel letters, aligned on the same baseline as the panel titles
for ax, letter in ((axa, "a"), (axb, "b")):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-43.5, 8),
                textcoords="offset points", ha="left", va="baseline",
                fontsize=FS_TITLE, fontweight="bold", color=INK)

fig.align_xlabels([axa, axb])

fig.text(0.015, 0.955, "Whitening brings all-subset type-I error back to nominal",
         ha="left", va="top", fontsize=FS_TITLE, fontweight="bold", color=INK)
fig.text(0.015, 0.89, "Observed / nominal type-I error in null simulations; whitened runs cover "
         "K = 30, 116, 300 and structures A, B, I", ha="left", va="top", fontsize=FS_MAIN, color=MUTED)

fig.savefig(Path(__file__).with_name("summary.png"), dpi=200, facecolor="white")
