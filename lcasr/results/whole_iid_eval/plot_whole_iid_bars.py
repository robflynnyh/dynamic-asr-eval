"""Render the whole_iid_eval sweep as a bar chart of WER improvement.

Bar height: relative WER improvement vs the unadapted baseline:
  improvement% = (baseline.wer - adapted.wer) / baseline.wer * 100

Bar labels show the adapted WER as an absolute percentage.

Run: `python plot_whole_iid_bars.py [--dataset earnings22] [--lrs 9e-6 9e-5]`
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggregate import aggregate


LR_PRETTY = {"9em5": "9e-5", "9em6": "9e-6"}
LR_TAG = {v: k for k, v in LR_PRETTY.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="earnings22", choices=["earnings22", "tedlium", "chime6", "rev16"])
    parser.add_argument("--lrs", nargs="+", default=["9e-6", "9e-5"],
                        help="Learning rates to plot (e.g. 9e-6 9e-5).")
    parser.add_argument("--out", type=str, default="whole_iid_bars.pdf")
    args = parser.parse_args()

    here = Path(__file__).parent
    results = aggregate(here)
    setting_re = re.compile(rf"{re.escape(args.dataset)}-test-whole-iid-epoch-(\d+)-lr-(\d+e[mp]?\d+)")

    parsed = {}
    for k, v in results.items():
        m = setting_re.match(k)
        if not m:
            continue
        parsed[(int(m.group(1)), m.group(2))] = v

    selected_tags = []
    for lr in args.lrs:
        tag = LR_TAG.get(lr)
        if tag is None:
            raise SystemExit(f"Unknown lr {lr!r}; known: {list(LR_TAG)}")
        selected_tags.append(tag)

    epochs = sorted({e for e, lr in parsed if lr in selected_tags})
    if not epochs:
        raise SystemExit(f"No results found for dataset={args.dataset}, lrs={args.lrs}")

    x = np.arange(len(epochs))
    n = len(selected_tags)
    width = 0.7 / n
    offsets = (np.arange(n) - (n - 1) / 2) * width
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for i, lr_tag in enumerate(selected_tags):
        improvement = []
        for e in epochs:
            r = parsed.get((e, lr_tag))
            if r is None:
                improvement.append(np.nan)
                continue
            improvement.append((r["baseline"]["wer"] - r["adapted"]["wer"])
                               / r["baseline"]["wer"] * 100.0)
        improvement = np.array(improvement)
        bars = ax.bar(x + offsets[i], improvement, width,
                      label=f"lr={LR_PRETTY[lr_tag]}",
                      color=colors[i % len(colors)],
                      edgecolor="black", linewidth=0.5)
        for e, b, v in zip(epochs, bars, improvement):
            if np.isnan(v):
                continue
            r = parsed[(e, lr_tag)]
            text_y = min(v + 0.5, 24.2) if v >= 0 else max(v - 0.5, -4.7)
            ax.text(
                b.get_x() + b.get_width() / 2,
                text_y,
                f"{r['adapted']['wer'] * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"epoch {e}" for e in epochs])
    ax.set_ylabel("WER improvement vs unadapted (%)")
    if n > 1:
        ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, which="both")
    ax.set_axisbelow(True)
    ax.set_ylim(-5, 25)

    out_path = (here / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
