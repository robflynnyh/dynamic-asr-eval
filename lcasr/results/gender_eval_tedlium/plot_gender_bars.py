"""Render the leave-one-speaker-out gender adaptation results as a grouped bar chart.

Default config: epoch=1, lr_tag=lr9e6 -> setting key `tedlium-epoch-1-lr9e6-test`.
Override with `--epoch` and `--lr-tag` (use empty `--lr-tag ""` for the
default-LR runs whose filenames have no lr suffix, e.g. `tedlium-epoch-1-test`).

Run: `python plot_gender_bars.py [--epoch 1] [--lr-tag lr9e6] [--out gender_bars.pdf]`
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggregate import aggregate


def setting_key(epoch: int, lr_tag: str) -> str:
    return f"tedlium-epoch-{epoch}{('-' + lr_tag) if lr_tag else ''}-test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lr-tag", type=str, default="lr9e6",
                        help='LR filename tag, e.g. "lr9e6". Use "" for default-LR runs.')
    parser.add_argument("--out", type=str, default="gender_bars.pdf",
                        help="Output figure path (extension picks the format).")
    args = parser.parse_args()

    here = Path(__file__).parent
    results = aggregate(here)
    key = setting_key(args.epoch, args.lr_tag)
    if key not in results:
        raise SystemExit(f"Setting {key!r} not found. Available: {sorted(results)}")
    row = results[key]

    test_groups = ["Test: Male", "Test: Female"]
    adapt_conditions = ["Adapt: Male", "Adapt: Female"]
    male_baseline = row["male_baseline"]["wer"]
    female_baseline = row["female_baseline"]["wer"]
    adapted = np.array([
        [row["male_to_male"]["wer"],   row["male_to_female"]["wer"]],
        [row["female_to_male"]["wer"], row["female_to_female"]["wer"]],
    ])
    baselines = np.array([male_baseline, female_baseline])
    rel_change = (adapted - baselines) / baselines * 100.0

    n_groups = len(test_groups)
    n_bars = len(adapt_conditions)
    x = np.arange(n_groups)
    width = 0.35
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * width

    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, (cond, off, c) in enumerate(zip(adapt_conditions, offsets, colors)):
        bars = ax.bar(x + off, rel_change[i], width, label=cond, color=c,
                      edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, rel_change[i]):
            va = "bottom" if v >= 0 else "top"
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.2f}%",
                    ha="center", va=va, fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(test_groups)
    ax.set_ylabel("WER change vs unadapted (%)")
    ymax = max(abs(rel_change.min()), abs(rel_change.max())) * 1.4 + 0.5
    ax.set_ylim(-ymax, ymax)
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    out_path = (here / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
