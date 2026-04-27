"""Render the crossdataset transfer results as a bar chart.

Adaptation is always on dataset A (earnings22, one record at a time, LOO).
For each setting this script plots, per test set, the relative WER decrease
after adaptation vs the unadapted baseline of that test set:
  - Test: A (earnings22) -> a_to_a_loo vs a_baseline   (within-dataset LOO)
  - Test: B (tedlium)    -> a_to_b vs b_baseline       (cross-dataset transfer)

Default config: epochs=1,5 for lr_tag=lr9e6. Override with `--epochs`
and `--lr-tag` (use `--lr-tag ""` for default-LR runs, `--lr-tag ao0` for the
adapt_overlap=0 run).

Run: `python plot_crossdataset_bars.py [--epochs 1 5] [--lr-tag lr9e6] [--out crossdataset_bars.pdf]`
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggregate import aggregate


def setting_key(epoch: int, lr_tag: str) -> str:
    return f"earnings22_tedlium-epoch-{epoch}{('-' + lr_tag) if lr_tag else ''}-test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, nargs="*", default=None,
                        help="Epochs to plot. Defaults to 1 5 for lr9e6, otherwise 1.")
    parser.add_argument("--lr-tag", type=str, default="lr9e6",
                        help='Filename tag, e.g. "lr9e6" or "ao0". Use "" for the default-LR runs.')
    parser.add_argument("--out", type=str, default="crossdataset_bars.pdf",
                        help="Output figure path (extension picks the format).")
    parser.add_argument("--name-a", type=str, default="earnings22")
    parser.add_argument("--name-b", type=str, default="tedlium")
    args = parser.parse_args()

    here = Path(__file__).parent
    results = aggregate(here)
    epochs = args.epochs if args.epochs else ([1, 5] if args.lr_tag == "lr9e6" else [1])
    rows = []
    for epoch in epochs:
        k = setting_key(epoch, args.lr_tag)
        if k not in results:
            raise SystemExit(f"Setting {k!r} not found. Available: {sorted(results)}")
        rows.append((epoch, results[k]))

    test_groups = [f"Test: {args.name_a}\n(within-dataset LOO)",
                   f"Test: {args.name_b}\n(cross-dataset)"]

    x = np.arange(len(test_groups))
    n_epochs = len(rows)
    group_width = 0.72
    bar_width = group_width / max(n_epochs, 1)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    offsets = (np.arange(n_epochs) - (n_epochs - 1) / 2.0) * bar_width
    all_decreases = []

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for idx, (epoch, row) in enumerate(rows):
        a_baseline = row["a_baseline"]["wer"]
        b_baseline = row["b_baseline"]["wer"]
        a_adapted = row["a_to_a_loo"]["wer"]
        b_adapted = row["a_to_b"]["wer"]

        pdecrease = np.array([
            (a_baseline - a_adapted) / a_baseline * 100.0,
            (b_baseline - b_adapted) / b_baseline * 100.0,
        ])
        all_decreases.append(pdecrease)
        baselines = [a_baseline, b_baseline]
        adapted = [a_adapted, b_adapted]
        bars = ax.bar(
            x + offsets[idx],
            pdecrease,
            bar_width,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.5,
            label=f"epoch {epoch}",
        )
        for b, v, base, ada in zip(bars, pdecrease, baselines, adapted):
            va = "bottom" if v >= 0 else "top"
            ax.text(
                b.get_x() + b.get_width() / 2,
                v,
                f"adapted {ada * 100:.2f}%\nbaseline {base * 100:.2f}%",
                ha="center",
                va=va,
                fontsize=9,
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(test_groups)
    ax.set_ylabel("WER decrease vs unadapted (%)")
    stacked = np.concatenate(all_decreases) if all_decreases else np.array([0.0])
    ymax = max(abs(stacked.min()), abs(stacked.max())) * 1.5 + 0.5
    ax.set_ylim(-ymax, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    epoch_label = ", ".join(str(epoch) for epoch, _ in rows)
    ax.set_title(f"Adapt on {args.name_a}  (epochs {epoch_label}, {args.lr_tag or 'default'})", fontsize=10)
    if n_epochs > 1:
        ax.legend(frameon=False, fontsize=9)

    out_path = (here / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
