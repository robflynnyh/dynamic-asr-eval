#!/usr/bin/env python3
"""Compare ROB-110 no-adapt 16384 CTC baseline checkpoint repeats."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
DEFAULT_SOURCES = {
    "rp_1": ROOT / "summary.csv",
    "rp_2": REPO_ROOT / "lcasr/results/ctc_seq16384_unadapted_baseline_repeats/rp_2/summary.csv",
    "rp_3": REPO_ROOT / "lcasr/results/ctc_seq16384_unadapted_baseline_repeats/rp_3/summary.csv",
}
DEFAULT_CHECKPOINTS = {
    "rp_1": "/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_1/step_105360.pt",
    "rp_2": "/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_2/step_105360.pt",
    "rp_3": "/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_16384_rp_3/step_105360.pt",
}


def read_summary(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary for {label}: {path}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["checkpoint_repeat"] = label
        row["checkpoint"] = DEFAULT_CHECKPOINTS.get(label, "")
        row["source_summary"] = str(path)
    return rows


def format_delta(value: str, anchor: str) -> tuple[str, str]:
    if value == "" or anchor == "":
        return "", ""
    wer = float(value)
    anchor_wer = float(anchor)
    absolute = wer - anchor_wer
    relative = (absolute / anchor_wer * 100.0) if anchor_wer else 0.0
    return f"{absolute:.10f}", f"{relative:.4f}"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "dataset",
        "split",
        "checkpoint_repeat",
        "seq_len",
        "overlap",
        "mode",
        "epochs",
        "n",
        "wer_mean",
        "wer_delta_vs_rp1_abs",
        "wer_delta_vs_rp1_rel_pct",
        "ins_rate_mean",
        "del_rate_mean",
        "sub_rate_mean",
        "words",
        "paths",
        "checkpoint",
        "source_summary",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    by_key = {(row["dataset"], row["checkpoint_repeat"]): row for row in rows}
    lines = [
        "# CTC 16384 No-Adapt Checkpoint Repeat Comparison",
        "",
        "Generated from ROB-110 no-adapt 16384-context CTC summary CSVs.",
        "",
        "| Dataset | rp_1 WER | rp_2 WER | rp_3 WER |",
        "|---|---:|---:|---:|",
    ]
    for dataset in datasets:
        values: list[str] = []
        for label in ("rp_1", "rp_2", "rp_3"):
            row = by_key.get((dataset, label), {})
            wer = row.get("wer_mean", "")
            values.append(f"{100 * float(wer):.2f}%" if wer != "" else "")
        lines.append(f"| {dataset} | {values[0]} | {values[1]} | {values[2]} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "lcasr/results/ctc_seq16384_unadapted_baseline_repeats",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for label, path in DEFAULT_SOURCES.items():
        rows.extend(read_summary(path, label))

    anchor_by_dataset = {
        row["dataset"]: row.get("wer_mean", "")
        for row in rows
        if row.get("checkpoint_repeat") == "rp_1"
    }
    for row in rows:
        delta, relative = format_delta(row.get("wer_mean", ""), anchor_by_dataset.get(row["dataset"], ""))
        row["wer_delta_vs_rp1_abs"] = delta
        row["wer_delta_vs_rp1_rel_pct"] = relative

    rows = sorted(rows, key=lambda row: (row["dataset"], row["checkpoint_repeat"]))
    write_csv(out_dir / "checkpoint_repeat_comparison.csv", rows)
    write_markdown(out_dir / "checkpoint_repeat_comparison.md", rows)
    print(f"Wrote {len(rows)} checkpoint-repeat comparison rows")
    print(out_dir / "checkpoint_repeat_comparison.csv")
    print(out_dir / "checkpoint_repeat_comparison.md")


if __name__ == "__main__":
    main()
