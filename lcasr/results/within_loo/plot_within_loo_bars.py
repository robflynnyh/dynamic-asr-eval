"""Render the within_loo sweep as a bar chart.

One bar per setting (epoch 1 and epoch 5), showing the relative WER change
of the LOO stitched decoding vs the unadapted windowed baseline:
  delta% = (loo.wer - baseline.wer) / baseline.wer * 100

Run: `python plot_within_loo_bars.py [--out within_loo_bars.pdf]`
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggregate import aggregate


EPOCH_RE = re.compile(r"epoch-(\d+)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="within_loo_bars.pdf")
    args = parser.parse_args()

    here = Path(__file__).parent
    results = aggregate(here)

    parsed = []
    for k, v in results.items():
        m = EPOCH_RE.search(k)
        if not m:
            continue
        parsed.append((int(m.group(1)), k, v))
    parsed.sort()

    labels = [f"epoch {e}" for e, _, _ in parsed]
    rel_change = np.array([
        (v["loo"]["wer"] - v["baseline"]["wer"]) / v["baseline"]["wer"] * 100.0
        for _, _, v in parsed
    ])

    x = np.arange(len(parsed))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in rel_change]

    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    bars = ax.bar(x, rel_change, 0.5, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, rel_change):
        va = "bottom" if v >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.2f}%",
                ha="center", va=va, fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WER change vs unadapted (%)")
    ymax = max(abs(rel_change.min()), abs(rel_change.max())) * 1.4 + 0.5
    ax.set_ylim(-ymax, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Within-recording LOO (earnings22 test)", fontsize=10)

    out_path = (here / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
