"""Render cross-dataset transfer results for epoch 1 in a single plot.

This script compares "Adapt on Tedlium" and "Adapt on Earnings" on both test sets.
X-axis: Test datasets (Tedlium, Earnings)
Legend: Adaptation source (Adapt on Tedlium, Adapt on Earnings)

Run:
  `python plot_crossdataset_combined.py`
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggregate import aggregate


def get_decrease(row, adapt_type, test_on):
    """
    row: the aggregated results for a setting (e.g. 'tedlium_earnings22-epoch-1-lr9e6-test')
    adapt_type: 'a' or 'b' (not used here directly, but conceptually)
    test_on: 'a' or 'b'
    """
    if test_on == 'a':
        baseline = row["a_baseline"]["wer"]
        # If it's a_to_a_loo, it means we adapted on a and tested on a
        # If it's a_to_b, it means we adapted on a and tested on b
        # Wait, the 'row' is for a specific adaptation direction.
        # So if row is 'tedlium_earnings22', then a=tedlium, b=earnings.
        # Adapt on Ted, Test on Ted: a_to_a_loo
        # Adapt on Ted, Test on Earnings: a_to_b
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1, help="Epoch to plot.")
    parser.add_argument("--lr-tag", type=str, default="lr9e6", help='LR tag.')
    parser.add_argument("--out", type=str, default="crossdataset_combined_epoch1.pdf", help="Output path.")
    args = parser.parse_args()

    here = Path(__file__).parent
    results = aggregate(here)

    # We expect two directions for the same pair of datasets
    # e.g. tedlium_earnings22 and earnings22_tedlium
    
    # Hardcode the datasets for now as requested or detect them
    d1 = "tedlium"
    d2 = "earnings22"
    
    tag = f"-{args.lr_tag}" if args.lr_tag else ""
    key1 = f"{d1}_{d2}-epoch-{args.epoch}{tag}-test"
    key2 = f"{d2}_{d1}-epoch-{args.epoch}{tag}-test"

    if key1 not in results or key2 not in results:
        print(f"Missing one of: {key1}, {key2}")
        print(f"Available keys: {list(results.keys())}")
        return

    res1 = results[key1] # Adapt on d1
    res2 = results[key2] # Adapt on d2

    # Group 1: Test on d1 (Tedlium)
    # - Adapt on d1: res1['a_to_a_loo'] vs res1['a_baseline']
    # - Adapt on d2: res2['a_to_b'] vs res2['b_baseline'] 
    #   (Wait, in res2, a=d2, b=d1. So a_to_b is Adapt d2, Test d1. b_baseline is d1 baseline)
    
    test_d1_adapt_d1 = (res1['a_baseline']['wer'] - res1['a_to_a_loo']['wer']) / res1['a_baseline']['wer'] * 100
    test_d1_adapt_d2 = (res2['b_baseline']['wer'] - res2['a_to_b']['wer']) / res2['b_baseline']['wer'] * 100

    # Group 2: Test on d2 (Earnings22)
    # - Adapt on d1: res1['a_to_b'] vs res1['b_baseline']
    #   (In res1, a=d1, b=d2. So a_to_b is Adapt d1, Test d2. b_baseline is d2 baseline)
    # - Adapt on d2: res2['a_to_a_loo'] vs res2['a_baseline']

    test_d2_adapt_d1 = (res1['b_baseline']['wer'] - res1['a_to_b']['wer']) / res1['b_baseline']['wer'] * 100
    test_d2_adapt_d2 = (res2['a_baseline']['wer'] - res2['a_to_a_loo']['wer']) / res2['a_baseline']['wer'] * 100

    base1 = res1['a_baseline']['wer'] * 100
    base2 = res1['b_baseline']['wer'] * 100

    test_groups = [
        f"Test on {d1.capitalize()}\n(Baseline: {base1:.1f}%)",
        f"Test on Earnings-22\n(Baseline: {base2:.1f}%)"
    ]
    adapt_labels = [f"Adapt on {d1.capitalize()}", "Adapt on Earnings-22"]

    # Data for plotting
    # Rows: Adapt on d1, Adapt on d2
    # Columns: Test on d1, Test on d2
    data = [
        [test_d1_adapt_d1, test_d2_adapt_d1], # Adapt on d1
        [test_d1_adapt_d2, test_d2_adapt_d2]  # Adapt on d2
    ]
    
    # Absolute WERs for labels
    abs_wers = [
        [res1['a_to_a_loo']['wer'] * 100, res1['a_to_b']['wer'] * 100], # Adapt on d1
        [res2['a_to_b']['wer'] * 100, res2['a_to_a_loo']['wer'] * 100]  # Adapt on d2
    ]

    x = np.arange(len(test_groups))
    width = 0.35
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(7, 5))
    
    rects1 = ax.bar(x - width/2, data[0], width, label=adapt_labels[0], color=colors[0], edgecolor="black", linewidth=0.5)
    rects2 = ax.bar(x + width/2, data[1], width, label=adapt_labels[1], color=colors[1], edgecolor="black", linewidth=0.5)

    def autolabel(rects, absolute_wers):
        for rect, abs_wer in zip(rects, absolute_wers):
            height = rect.get_height()
            ax.annotate(f'{abs_wer:.1f}%\n({height:+.1f}%)',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -3),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9)

    autolabel(rects1, abs_wers[0])
    autolabel(rects2, abs_wers[1])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Relative WER decrease vs unadapted (%)")
    # ax.set_title(f"Cross-dataset Adaptation (Epoch {args.epoch}, {args.lr_tag})")
    ax.set_xticks(x)
    ax.set_xticklabels(test_groups)
    ax.legend()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, which="both")
    ax.set_axisbelow(True)

    # Set y-axis limits with some padding
    all_vals = [v for row in data for v in row]
    ymax = max(abs(min(all_vals)), abs(max(all_vals))) * 1.3 + 1
    ax.set_ylim(-ymax, ymax)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    # Also save as png
    png_out = Path(args.out).with_suffix(".png")
    fig.savefig(png_out, bbox_inches="tight")
    
    print(f"Saved {args.out} and {png_out}")

if __name__ == "__main__":
    main()
