#!/usr/bin/env python3
"""Write a snapshot comparison for ROB-115 CTC context rows."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parent
OUT_MD = ROOT / "comparison_snapshot.md"

SOURCES = [
    ("2048 no-adapt", REPO_ROOT / "lcasr/results/ctc_seq2048_unadapted_baseline/summary.csv", "summary"),
    ("2048 adapted", REPO_ROOT / "lcasr/results/ctc_seq2048_self_training_eval/summary_by_setting.csv", "grouped"),
    ("8192 no-adapt", REPO_ROOT / "lcasr/results/ctc_seq8192_unadapted_baseline/summary.csv", "summary"),
    ("8192 adapted", ROOT / "summary_by_setting.csv", "grouped"),
    ("16384 no-adapt", REPO_ROOT / "lcasr/results/ctc_seq16384_unadapted_baseline/summary.csv", "summary"),
    ("16384 adapted RMM", REPO_ROOT / "lcasr/results/rmm_eval/ctc_seq16384/summary_by_setting.csv", "grouped"),
    ("65536 no-adapt", REPO_ROOT / "lcasr/results/seq_65536_investigation/unadapted_baseline/summary.csv", "summary"),
    ("65536 adapted stride2048", REPO_ROOT / "lcasr/results/seq_65536_investigation/self_training_stride2048/summary_by_setting.csv", "grouped"),
]

PARTIAL_PKL_SOURCES = [
    ("16384 adapted freq-mask partial", REPO_ROOT / "lcasr/results/entropy_ablation/pkl/tedlium-test-epoch-5-freq_mask_1.pkl"),
    ("16384 adapted freq-mask partial", REPO_ROOT / "lcasr/results/entropy_ablation/pkl/earnings22-test-epoch-5-freq_mask_1.pkl"),
]


def pct(value: str) -> str:
    return f"{100 * float(value):.2f}%" if value else ""


def lr_label(value: object) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.0e}".replace("e-0", "em").replace("e-", "em").replace("e+0", "e")


def read_rows(label: str, path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [{**row, "source": label, "summary_path": str(path.relative_to(REPO_ROOT))} for row in reader]


def read_pkl_row(label: str, path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        result = pickle.load(f)
    args = result.get("args_dict", {})
    return {
        "source": label,
        "dataset": str(args.get("dataset", "")),
        "split": str(args.get("split", "")),
        "seq_len": str(args.get("seq_len", "")),
        "overlap": str(args.get("overlap", "")),
        "epochs": str(args.get("epochs", "")),
        "lr": lr_label(args.get("optim_lr", "")),
        "n": "1",
        "wer_mean": str(result.get("wer", "")),
        "summary_path": str(path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for label, path, _kind in SOURCES:
        source_rows = read_rows(label, path)
        if not source_rows:
            missing.append(f"- {label}: `{path.relative_to(REPO_ROOT)}`")
            continue
        rows.extend(source_rows)
    for label, path in PARTIAL_PKL_SOURCES:
        row = read_pkl_row(label, path)
        if row is None:
            missing.append(f"- {label}: `{path.relative_to(REPO_ROOT)}`")
            continue
        rows.append(row)

    lines = [
        "# ROB-115 CTC Context Comparison Snapshot",
        "",
        "This is a snapshot, not a final thesis-ready comparison. It is generated",
        "from committed summary and result artifacts available in this checkout.",
        "",
        "Policy notes:",
        "",
        "- The 2048, 8192, and 65536 adapted rows use the standard frequency-mask",
        "  self-training policy (`spec_augment_n_freq_masks=6`,",
        "  `spec_augment_freq_mask_param=34`, `spec_augment_n_time_masks=0`).",
        "- The 16384 adapted freq-mask partial rows are exact standard",
        "  frequency-mask rows from the entropy-ablation artifacts, but only",
        "  TEDLIUM and Earnings22 have committed exact-context rows in that",
        "  source, so they are not a full four-dataset replacement for the RMM",
        "  block.",
        "- The 16384 adapted rows are explicitly labeled `16384 adapted RMM` because",
        "  they come from the RMM random mixed-mask result family, not the",
        "  frequency-mask-only policy.",
        "- The 65536 adapted row uses the stride-2048 follow-up overlap, so its",
        "  adapted/no-adapt pair is useful context but not a perfectly matched",
        "  stride comparison.",
        "",
        "| Source | Dataset | Split | Seq Len | Overlap | Epochs | LR | N | WER | Artifact |",
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
        ])

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(OUT_MD)


if __name__ == "__main__":
    main()
