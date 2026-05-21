#!/usr/bin/env python3
"""Write a snapshot comparison for ROB-115 CTC context rows."""

from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parent
OUT_MD = ROOT / "comparison_snapshot.md"

SOURCES = [
    ("2048 no-adapt", REPO_ROOT / "lcasr/results/ctc_seq2048_unadapted_baseline/summary.csv", "summary"),
    ("2048 adapted", REPO_ROOT / "lcasr/results/ctc_seq2048_self_training_eval/summary_by_setting.csv", "grouped"),
    ("8192 adapted", ROOT / "summary_by_setting.csv", "grouped"),
    ("16384 adapted RMM", REPO_ROOT / "lcasr/results/rmm_eval/ctc_seq16384/summary_by_setting.csv", "grouped"),
    ("65536 no-adapt", REPO_ROOT / "lcasr/results/seq_65536_investigation/unadapted_baseline/summary.csv", "summary"),
    ("65536 adapted stride2048", REPO_ROOT / "lcasr/results/seq_65536_investigation/self_training_stride2048/summary_by_setting.csv", "grouped"),
]


def pct(value: str) -> str:
    return f"{100 * float(value):.2f}%" if value else ""


def read_rows(label: str, path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [{**row, "source": label, "summary_path": str(path.relative_to(REPO_ROOT))} for row in reader]


def main() -> None:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for label, path, _kind in SOURCES:
        source_rows = read_rows(label, path)
        if not source_rows:
            missing.append(f"- {label}: `{path.relative_to(REPO_ROOT)}`")
            continue
        rows.extend(source_rows)

    lines = [
        "# ROB-115 CTC Context Comparison Snapshot",
        "",
        "This is a snapshot, not a final thesis-ready comparison. It is generated",
        "from committed summary artifacts available in this checkout.",
        "",
        "| Source | Dataset | Split | Seq Len | Overlap | Epochs | LR | N | WER | Summary |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: (
        int(item.get("seq_len") or 0),
        item.get("source", ""),
        item.get("dataset", ""),
        int(item.get("epochs") or 0),
        item.get("lr", ""),
    )):
        lines.append(
            f"| {row.get('source', '')} | {row.get('dataset', '')} | {row.get('split', '')} | "
            f"{row.get('seq_len', '')} | {row.get('overlap', '')} | {row.get('epochs', '')} | "
            f"{row.get('lr', '')} | {row.get('n', '')} | {pct(row.get('wer_mean', ''))} | "
            f"`{row.get('summary_path', '')}` |"
        )

    if missing:
        lines.extend([
            "",
            "## Missing Snapshot Inputs",
            "",
            *missing,
            "- 8192 no-adapt baseline: no committed summary artifact matching seq_len `8192` was found in this checkout.",
            "- 16384 no-adapt baseline: no exact all-dataset committed CTC no-adapt summary was found in this checkout.",
        ])

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(OUT_MD)


if __name__ == "__main__":
    main()
