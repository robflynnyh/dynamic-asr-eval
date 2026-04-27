"""Render the half_concat_eval sweep as a bar chart of relative WER change.

Each setting in this folder is one (adapt_overlap, lr) configuration at
epoch 1. The script plots one bar per setting showing the relative change
in held-out WER vs the unadapted baseline:
  delta% = (adapted.wer - baseline.wer) / baseline.wer * 100

Run: `python plot_half_concat_bars.py [--out half_concat_bars.pdf]`
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggregate import aggregate


PRETTY = {
    "earnings22-test-half-concat-epoch-1":              "ao=14336\nlr=9e-5",
    "earnings22-test-half-concat-epoch-1-ao0-lr9e6":    "ao=0\nlr=9e-6",
    "earnings22-test-half-concat-epoch-1-ao14336-lr9e6": "ao=14336\nlr=9e-6",
    "earnings22-test-half-concat-epoch-1-ao14336-lr1e6": "ao=14336\nlr=1e-6",
}
ORDER = [
    "earnings22-test-half-concat-epoch-1",
    "earnings22-test-half-concat-epoch-1-ao0-lr9e6",
    "earnings22-test-half-concat-epoch-1-ao14336-lr9e6",
    "earnings22-test-half-concat-epoch-1-ao14336-lr1e6",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="half_concat_bars.pdf")
    args = parser.parse_args()

    here = Path(__file__).parent
    results = aggregate(here)

    keys = [k for k in ORDER if k in results]
    labels = [PRETTY.get(k, k) for k in keys]
    rel_change = np.array([
        (results[k]["adapted"]["wer"] - results[k]["baseline"]["wer"])
        / results[k]["baseline"]["wer"] * 100.0
        for k in keys
    ])

    x = np.arange(len(keys))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in rel_change]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(x, rel_change, 0.6, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, rel_change):
        va = "bottom" if v >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.2f}%",
                ha="center", va=va, fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("WER change vs unadapted (%)")
    ymax = max(abs(rel_change.min()), abs(rel_change.max())) * 1.4 + 0.5
    ax.set_ylim(-ymax, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Half-concat adapt-only sweep (earnings22 test, epoch 1)", fontsize=10)

    out_path = (here / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
