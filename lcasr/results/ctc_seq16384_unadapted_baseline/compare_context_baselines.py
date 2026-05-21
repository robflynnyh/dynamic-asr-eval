#!/usr/bin/env python3
"""Compare no-adapt CTC baseline summaries for 2048, 16384, and 65536 context."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
DEFAULT_SOURCES = {
    "2048": REPO_ROOT / "lcasr/results/ctc_seq2048_unadapted_baseline/summary.csv",
    "16384": ROOT / "summary.csv",
    "65536": REPO_ROOT / "lcasr/results/seq_65536_investigation/unadapted_baseline/summary.csv",
}


def read_summary(path: Path, label: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["source_seq_len"] = label
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
        "source_seq_len",
        "seq_len",
        "overlap",
        "mode",
        "epochs",
        "n",
        "wer_mean",
        "wer_delta_vs_seq16384_abs",
        "wer_delta_vs_seq16384_rel_pct",
        "ins_rate_mean",
        "del_rate_mean",
        "sub_rate_mean",
        "words",
        "paths",
        "source_summary",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    by_key = {(row["dataset"], row["source_seq_len"]): row for row in rows}
    lines = [
        "# CTC No-Adapt Context Baseline Comparison",
        "",
        "Generated from summary CSVs for the 2048, 16384, and 65536 no-adapt CTC baselines.",
        "",
        "| Dataset | 2048 WER | 16384 WER | 65536 WER |",
        "|---|---:|---:|---:|",
    ]
    for dataset in datasets:
        values: list[str] = []
        for label in ("2048", "16384", "65536"):
            row = by_key.get((dataset, label), {})
            wer = row.get("wer_mean", "")
            values.append(f"{100 * float(wer):.2f}%" if wer != "" else "")
        lines.append(f"| {dataset} | {values[0]} | {values[1]} | {values[2]} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()

    rows: list[dict[str, str]] = []
    for label, path in DEFAULT_SOURCES.items():
        rows.extend(read_summary(path, label))

    anchor_by_dataset = {
        row["dataset"]: row.get("wer_mean", "")
        for row in rows
        if row.get("source_seq_len") == "16384"
    }
    for row in rows:
        delta, relative = format_delta(row.get("wer_mean", ""), anchor_by_dataset.get(row["dataset"], ""))
        row["wer_delta_vs_seq16384_abs"] = delta
        row["wer_delta_vs_seq16384_rel_pct"] = relative

    rows = sorted(rows, key=lambda row: (row["dataset"], int(row["source_seq_len"])))
    write_csv(out_dir / "context_baseline_comparison.csv", rows)
    write_markdown(out_dir / "context_baseline_comparison.md", rows)
    print(f"Wrote {len(rows)} comparison rows")
    print(out_dir / "context_baseline_comparison.csv")
    print(out_dir / "context_baseline_comparison.md")


if __name__ == "__main__":
    main()
