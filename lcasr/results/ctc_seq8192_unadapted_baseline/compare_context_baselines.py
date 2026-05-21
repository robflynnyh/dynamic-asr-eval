#!/usr/bin/env python3
"""Compare no-adapt CTC context-length baselines for thesis checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUTS = [
    ("2048", ROOT / "lcasr/results/ctc_seq2048_unadapted_baseline/summary.csv"),
    ("8192", ROOT / "lcasr/results/ctc_seq8192_unadapted_baseline/summary.csv"),
    ("16384", ROOT / "lcasr/results/ctc_seq16384_unadapted_baseline/summary.csv"),
    ("65536", ROOT / "lcasr/results/seq_65536_investigation/unadapted_baseline/summary.csv"),
]


def read_summary(seq_len: str, path: Path) -> tuple[list[dict[str, str]], str]:
    if not path.exists():
        return [], f"missing: {path}"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, str]] = []
    for row in rows:
        out.append({
            "dataset": row.get("dataset", ""),
            "split": row.get("split", ""),
            "seq_len": row.get("seq_len") or seq_len,
            "overlap": row.get("overlap", ""),
            "stride": stride(row.get("seq_len") or seq_len, row.get("overlap", "")),
            "mode": row.get("mode", "no_adapt"),
            "epochs": row.get("epochs", "0"),
            "n": row.get("n", ""),
            "wer_mean": row.get("wer_mean", ""),
            "ins_rate_mean": row.get("ins_rate_mean", ""),
            "del_rate_mean": row.get("del_rate_mean", ""),
            "sub_rate_mean": row.get("sub_rate_mean", ""),
            "words": row.get("words", ""),
            "source_summary": str(path),
        })
    return out, f"loaded {len(out)} rows: {path}"


def stride(seq_len: str, overlap: str) -> str:
    try:
        return str(int(seq_len) - int(overlap))
    except (TypeError, ValueError):
        return ""


def pct(value: str) -> str:
    return f"{100 * float(value):.2f}%" if value else ""


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "dataset",
        "split",
        "seq_len",
        "overlap",
        "stride",
        "mode",
        "epochs",
        "n",
        "wer_mean",
        "ins_rate_mean",
        "del_rate_mean",
        "sub_rate_mean",
        "words",
        "source_summary",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, rows: list[dict[str, str]], notes: list[str]) -> None:
    lines = [
        "# CTC No-Adapt Context Baseline Comparison",
        "",
        "| Dataset | Split | Seq Len | Overlap | Stride | WER | Ins | Del | Sub |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['seq_len']} | "
            f"{row['overlap']} | {row['stride']} | {pct(row['wer_mean'])} | "
            f"{pct(row['ins_rate_mean'])} | {pct(row['del_rate_mean'])} | "
            f"{pct(row['sub_rate_mean'])} |"
        )
    lines.extend(["", "## Sources", ""])
    lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for context_baseline_comparison.csv/md",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    notes: list[str] = []
    for seq_len, path in DEFAULT_INPUTS:
        loaded, note = read_summary(seq_len, path)
        rows.extend(loaded)
        notes.append(note)

    rows.sort(key=lambda row: (row["dataset"], row["split"], int(row["seq_len"])))
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "context_baseline_comparison.csv", rows)
    write_markdown(out_dir / "context_baseline_comparison.md", rows, notes)
    print(f"Wrote {len(rows)} comparison rows")
    print(out_dir / "context_baseline_comparison.csv")
    print(out_dir / "context_baseline_comparison.md")


if __name__ == "__main__":
    main()
