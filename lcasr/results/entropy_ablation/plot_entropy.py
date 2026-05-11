#!/usr/bin/env python3
"""Plot ROB-57 entropy traces by adaptation update count."""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PER_STEP_CSV = ROOT / "entropy_by_update.csv"
OUT_PDF = ROOT / "entropy_by_update.pdf"
OUT_PNG = ROOT / "entropy_by_update.png"

DATASET_LABELS = {
    "earnings22": "Earnings22 test",
    "tedlium": "TED-LIUM test",
}

SETTING_LABELS = {
    "freq_mask": "frequency masking",
    "no_aug": "no augmentation",
}

COLORS = {
    "freq_mask": "#4C72B0",
    "no_aug": "#DD8452",
}


def ensure_summary(refresh: bool) -> None:
    if refresh or not PER_STEP_CSV.exists():
        subprocess.run(["python", str(ROOT / "aggregate_entropy.py")], cwd=ROOT, check=True)


def load_rows() -> list[dict[str, str]]:
    with PER_STEP_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    selected = []
    for row in rows:
        measurement = row.get("measurement", "")
        update_step = int(row.get("update_step", 0))
        if measurement == "post_update" or (measurement == "pre_update" and update_step == 0):
            selected.append(row)
    return selected


def plot(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise SystemExit("No entropy rows found. Run the ablation first.")

    datasets = [dataset for dataset in ("tedlium", "earnings22") if any(row["dataset"] == dataset for row in rows)]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(8.5, max(3.2, 3.0 * len(datasets))), squeeze=False)

    for ax, dataset in zip(axes[:, 0], datasets):
        by_setting: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row in rows:
            if row["dataset"] != dataset:
                continue
            by_setting[row["setting"]].append((int(row["update_step"]), float(row["mean_entropy"])))

        for setting in ("freq_mask", "no_aug"):
            points = sorted(by_setting.get(setting, []))
            if not points:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=2.5,
                linewidth=1.4,
                color=COLORS.get(setting),
                label=SETTING_LABELS.get(setting, setting),
            )

        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xlabel("Adaptation updates")
        ax.set_ylabel("Mean frame entropy")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=180)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Regenerate entropy_by_update.csv first.")
    args = parser.parse_args()
    ensure_summary(args.refresh)
    plot(load_rows())


if __name__ == "__main__":
    main()
