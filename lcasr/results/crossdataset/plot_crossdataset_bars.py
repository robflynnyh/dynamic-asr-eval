"""Render the crossdataset transfer results as bar charts.

Adaptation is always on dataset A (one record at a time, LOO).
For each setting this script plots, per test set, the relative WER decrease
after adaptation vs the unadapted baseline of that test set:
  - Test: A -> a_to_a_loo vs a_baseline   (within-dataset LOO)
  - Test: B -> a_to_b vs b_baseline       (cross-dataset transfer)

By default this writes plots for all matching directions in the aggregate
results. Use `--dataset-a` and `--dataset-b` to plot one direction.

Run:
  `python plot_crossdataset_bars.py`
  `python plot_crossdataset_bars.py --dataset-a tedlium --dataset-b earnings22 --out tedlium_earnings22_bars.pdf`
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggregate import aggregate


SETTING_RE = re.compile(
    r"^(?P<dataset_a>.+)_(?P<dataset_b>.+)-epoch-(?P<epoch>\d+)"
    r"(?P<tag>(?:-[^-]+)*)-test$"
)


def setting_key(dataset_a: str, dataset_b: str, epoch: int, lr_tag: str) -> str:
    return f"{dataset_a}_{dataset_b}-epoch-{epoch}{('-' + lr_tag) if lr_tag else ''}-test"


def available_directions(results: dict, lr_tag: str, epochs: list[int]) -> list[tuple[str, str]]:
    directions = set()
    for key in results:
        match = SETTING_RE.match(key)
        if match is None:
            continue
        tag = match.group("tag").lstrip("-")
        epoch = int(match.group("epoch"))
        if tag == lr_tag and epoch in epochs:
            directions.add((match.group("dataset_a"), match.group("dataset_b")))
    return sorted(directions)


def plot_direction(
        results: dict,
        dataset_a: str,
        dataset_b: str,
        epochs: list[int],
        lr_tag: str,
        out: Path,
    ) -> None:
    rows = []
    for epoch in epochs:
        key = setting_key(dataset_a, dataset_b, epoch, lr_tag)
        if key not in results:
            print(f"Skipping missing setting {key!r}")
            continue
        rows.append((epoch, results[key]))

    if not rows:
        raise SystemExit(
            f"No settings found for {dataset_a}->{dataset_b}, epochs={epochs}, lr_tag={lr_tag!r}. "
            f"Available: {sorted(results)}"
        )

    test_groups = [f"Test: {dataset_a}\n(within-dataset LOO)",
                   f"Test: {dataset_b}\n(cross-dataset)"]

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
        for bar, value, base, ada in zip(bars, pdecrease, baselines, adapted):
            va = "bottom" if value >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
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
    ax.set_title(f"Adapt on {dataset_a}  (epochs {epoch_label}, {lr_tag or 'default'})", fontsize=10)
    if n_epochs > 1:
        ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, nargs="*", default=None,
                        help="Epochs to plot. Defaults to 1 5 for lr9e6, otherwise 1.")
    parser.add_argument("--lr-tag", type=str, default="lr9e6",
                        help='Filename tag, e.g. "lr9e6" or "ao0". Use "" for the default-LR runs.')
    parser.add_argument("--out", type=str, default="crossdataset_bars.pdf",
                        help="Output figure path. When plotting all directions, this is used only for the legacy earnings22->tedlium direction.")
    parser.add_argument("--dataset-a", type=str, default=None,
                        help="Adaptation dataset. If omitted, plot all available directions for the requested epochs/tag.")
    parser.add_argument("--dataset-b", type=str, default=None,
                        help="Cross-dataset test dataset. Must be set with --dataset-a.")
    args = parser.parse_args()

    here = Path(__file__).parent
    results = aggregate(here)
    epochs = args.epochs if args.epochs else ([1, 5] if args.lr_tag == "lr9e6" else [1])

    if (args.dataset_a is None) != (args.dataset_b is None):
        raise SystemExit("--dataset-a and --dataset-b must be set together")

    if args.dataset_a is not None:
        out_path = (here / args.out) if not Path(args.out).is_absolute() else Path(args.out)
        plot_direction(results, args.dataset_a, args.dataset_b, epochs, args.lr_tag, out_path)
        return

    directions = available_directions(results, args.lr_tag, epochs)
    if not directions:
        raise SystemExit(f"No directions found for epochs={epochs}, lr_tag={args.lr_tag!r}")

    for dataset_a, dataset_b in directions:
        if dataset_a == "earnings22" and dataset_b == "tedlium":
            out_name = args.out
        else:
            suffix = Path(args.out).suffix or ".pdf"
            out_name = f"{dataset_a}_{dataset_b}_bars{suffix}"
        out_path = (here / out_name) if not Path(out_name).is_absolute() else Path(out_name)
        plot_direction(results, dataset_a, dataset_b, epochs, args.lr_tag, out_path)


if __name__ == "__main__":
    main()
