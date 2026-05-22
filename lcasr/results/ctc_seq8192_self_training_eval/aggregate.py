#!/usr/bin/env python3
"""Aggregate ROB-115 CTC 8192-context self-training eval results."""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE_SUMMARY = ROOT.parent / "ctc_seq8192_unadapted_baseline" / "summary.csv"
RESULT_RE = re.compile(
    r"(?P<dataset>.+)-(?P<split>dev|test)-ctc-seq(?P<seq>\d+)"
    r"-overlap(?P<overlap>\d+)-epoch-(?P<epoch>\d+)"
    r"-lr-(?P<lr>[^_]+)_(?P<repeat>\d+)\.pkl$"
)


def mean(values: list[float]) -> str:
    return f"{statistics.mean(values):.10f}" if values else ""


def stdev(values: list[float]) -> str:
    return f"{statistics.stdev(values):.10f}" if len(values) > 1 else ""


def load_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*.pkl")):
        match = RESULT_RE.match(path.name)
        if not match:
            continue
        try:
            with path.open("rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            rows.append({
                **match.groupdict(),
                "wer": "",
                "ins_rate": "",
                "del_rate": "",
                "sub_rate": "",
                "words": "",
                "path": path.name,
                "error": repr(exc),
            })
            continue

        rows.append({
            "dataset": match.group("dataset"),
            "split": match.group("split"),
            "seq_len": match.group("seq"),
            "overlap": match.group("overlap"),
            "epochs": match.group("epoch"),
            "lr": match.group("lr"),
            "repeat": match.group("repeat"),
            "wer": data.get("wer", "") if isinstance(data, dict) else "",
            "ins_rate": data.get("ins_rate", "") if isinstance(data, dict) else "",
            "del_rate": data.get("del_rate", "") if isinstance(data, dict) else "",
            "sub_rate": data.get("sub_rate", "") if isinstance(data, dict) else "",
            "words": data.get("words", "") if isinstance(data, dict) else "",
            "path": path.name,
            "error": "",
        })
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        key = (
            str(row["dataset"]),
            str(row["split"]),
            str(row["seq_len"]),
            str(row["overlap"]),
            str(row["epochs"]),
            str(row["lr"]),
        )
        grouped[key].append(row)

    out: list[dict[str, object]] = []
    for (dataset, split, seq_len, overlap, epochs, lr), items in grouped.items():
        def floats(field: str) -> list[float]:
            vals: list[float] = []
            for item in items:
                try:
                    vals.append(float(item.get(field, "")))
                except Exception:
                    pass
            return vals

        out.append({
            "dataset": dataset,
            "split": split,
            "seq_len": seq_len,
            "overlap": overlap,
            "epochs": epochs,
            "lr": lr,
            "n": len(floats("wer")),
            "wer_mean": mean(floats("wer")),
            "wer_std": stdev(floats("wer")),
            "ins_rate_mean": mean(floats("ins_rate")),
            "del_rate_mean": mean(floats("del_rate")),
            "sub_rate_mean": mean(floats("sub_rate")),
            "words": items[0].get("words", ""),
            "repeats": " ".join(str(item.get("repeat", "")) for item in items),
            "paths": " ".join(str(item.get("path", "")) for item in items),
        })
    return sorted(out, key=lambda row: (str(row["dataset"]), int(row["epochs"]), str(row["lr"])))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_baseline_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pct_value(value: object) -> str:
    if value in ("", None):
        return ""
    return f"{100 * float(value):.2f}%"


def signed_pct_value(value: float) -> str:
    return f"{100 * value:+.2f}%"


def write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    source: Path,
    baseline_summary: Path = BASELINE_SUMMARY,
) -> None:
    grouped_rows = summarize(rows)
    baseline_rows = read_baseline_rows(baseline_summary)
    baseline_by_dataset = {
        row["dataset"]: row for row in baseline_rows if row.get("split") == "test"
    }
    lines = [
        "# CTC 8192-Context Self-Training Summary",
        "",
        f"Generated from `{source}`.",
        "",
        f"Per-repeat rows: `{len(rows)}`.",
        f"Grouped rows: `{len(grouped_rows)}`.",
        "",
        "| Dataset | Epochs | LR | N | WER Mean | WER Std | Ins | Del | Sub | Words |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped_rows:
        lines.append(
            f"| {row.get('dataset', '')} | {row.get('epochs', '')} | {row.get('lr', '')} | "
            f"{row.get('n', '')} | {pct_value(row.get('wer_mean', ''))} | "
            f"{pct_value(row.get('wer_std', ''))} | "
            f"{pct_value(row.get('ins_rate_mean', ''))} | "
            f"{pct_value(row.get('del_rate_mean', ''))} | "
            f"{pct_value(row.get('sub_rate_mean', ''))} | {row.get('words', '')} |"
        )
    lines.extend([
        "",
        "## ROB-110 8192 No-Adapt Baseline",
        "",
    ])
    if baseline_rows:
        lines.extend([
            f"Imported from `{baseline_summary.resolve()}`.",
            "",
            "| Dataset | Split | Epochs | N | WER Mean | Ins | Del | Sub | Words |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in sorted(baseline_rows, key=lambda item: item.get("dataset", "")):
            lines.append(
                f"| {row.get('dataset', '')} | {row.get('split', '')} | "
                f"{row.get('epochs', '')} | {row.get('n', '')} | "
                f"{pct_value(row.get('wer_mean', ''))} | "
                f"{pct_value(row.get('ins_rate_mean', ''))} | "
                f"{pct_value(row.get('del_rate_mean', ''))} | "
                f"{pct_value(row.get('sub_rate_mean', ''))} | {row.get('words', '')} |"
            )
        lines.extend([
            "",
            "## Adapted Delta From 8192 No-Adapt",
            "",
            "| Dataset | LR | Adapted WER | No-Adapt WER | Absolute Delta | Relative Delta |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in grouped_rows:
            baseline = baseline_by_dataset.get(str(row.get("dataset", "")))
            if not baseline:
                continue
            adapted_wer = float(row.get("wer_mean", ""))
            baseline_wer = float(baseline.get("wer_mean", ""))
            absolute_delta = adapted_wer - baseline_wer
            relative_delta = absolute_delta / baseline_wer if baseline_wer else 0.0
            lines.append(
                f"| {row.get('dataset', '')} | {row.get('lr', '')} | "
                f"{pct_value(adapted_wer)} | {pct_value(baseline_wer)} | "
                f"{signed_pct_value(absolute_delta)} | {signed_pct_value(relative_delta)} |"
            )
    else:
        lines.append(f"Missing baseline summary: `{baseline_summary.resolve()}`.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Result directory containing ROB-115 CTC self-training PKL files",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_rows(root)
    grouped_rows = summarize(rows)
    row_fields = [
        "dataset", "split", "seq_len", "overlap", "epochs", "lr", "repeat",
        "wer", "ins_rate", "del_rate", "sub_rate", "words", "path", "error",
    ]
    grouped_fields = [
        "dataset", "split", "seq_len", "overlap", "epochs", "lr", "n", "wer_mean",
        "wer_std", "ins_rate_mean", "del_rate_mean", "sub_rate_mean", "words",
        "repeats", "paths",
    ]
    write_csv(root / "summary.csv", rows, row_fields)
    write_csv(root / "summary_by_setting.csv", grouped_rows, grouped_fields)
    write_markdown(root / "summary.md", rows, root)
    print(f"Wrote {len(rows)} per-repeat rows and {len(grouped_rows)} grouped rows")
    print(root / "summary.csv")
    print(root / "summary_by_setting.csv")
    print(root / "summary.md")


if __name__ == "__main__":
    main()
