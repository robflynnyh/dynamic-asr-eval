"""Render the whole_concat_eval sweep as a bar chart of WER improvement.

X-axis: number of adaptation epochs over the full concatenated test spec.
Bar height: relative WER improvement vs the unadapted baseline (positive
means lower WER after adaptation):
  improvement% = (baseline.wer - adapted.wer) / baseline.wer * 100

Bar labels show the adapted WER as an absolute percentage.

By default only the lr=9e-6 sweep is shown. Pass `--lrs 9e-6 9e-5` to
overlay multiple learning rates as grouped bars.

Run: `python plot_whole_concat_bars.py [--lrs 9e-6] [--out whole_concat_bars.pdf]`
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggregate import aggregate


SETTING_RE = re.compile(r"earnings22-test-whole-concat-epoch-(\d+)-lr-(\d+e[mp]?\d+)")
LR_PRETTY = {"9em5": "9e-5", "9em6": "9e-6"}
LR_TAG = {v: k for k, v in LR_PRETTY.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrs", nargs="+", default=["9e-6", "9e-5"],
                        help="Learning rates to plot (e.g. 9e-6 9e-5).")
    parser.add_argument("--out", type=str, default="whole_concat_bars.pdf")
    args = parser.parse_args()

    here = Path(__file__).parent
    results = aggregate(here)

    parsed = {}
    for k, v in results.items():
        m = SETTING_RE.match(k)
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
        raise SystemExit(f"No results found for lrs={args.lrs}")

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
            va = "bottom" if v >= 0 else "top"
            ax.text(
                b.get_x() + b.get_width() / 2,
                v,
                f"{r['adapted']['wer'] * 100:.1f}%",
                ha="center",
                va=va,
                fontsize=9,
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"epoch {e}" for e in epochs])
    ax.set_ylabel("WER improvement vs unadapted (%)")
    if n > 1:
        ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    title_lrs = ", ".join(f"lr={lr}" for lr in args.lrs)
    ax.set_title(f"Whole-concat adapt-only (earnings22 test, {title_lrs})", fontsize=10)

    out_path = (here / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
