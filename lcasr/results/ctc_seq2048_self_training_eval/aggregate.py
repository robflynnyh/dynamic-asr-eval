#!/usr/bin/env python3
"""Aggregate ROB-56 CTC 2048-context self-training eval results."""

from __future__ import annotations

import csv
import argparse
import pickle
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_CSV = ROOT / "summary.csv"
OUT_GROUPED_CSV = ROOT / "summary_by_setting.csv"
OUT_MD = ROOT / "summary.md"


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(ROOT.glob("*.pkl")):
        match = re.match(
            r"(?P<dataset>.+)-(?P<split>dev|test)-ctc-seq(?P<seq>\d+)-overlap(?P<overlap>\d+)-epoch-(?P<epoch>\d+)-lr-(?P<lr>[^_]+)_(?P<repeat>\d+)\.pkl$",
            path.name,
        )
        try:
            with path.open("rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            rows.append({
                "dataset": match.group("dataset") if match else "",
                "split": match.group("split") if match else "",
                "seq_len": match.group("seq") if match else "",
                "overlap": match.group("overlap") if match else "",
                "epochs": match.group("epoch") if match else "",
                "lr": match.group("lr") if match else "",
                "repeat": match.group("repeat") if match else "",
                "wer": "",
                "ins_rate": "",
                "del_rate": "",
                "sub_rate": "",
                "words": "",
                "path": path.name,
                "error": repr(exc),
            })
            continue

        args = data.get("args_dict", {}) if isinstance(data, dict) else {}
        rows.append({
            "dataset": match.group("dataset") if match else args.get("dataset", ""),
            "split": match.group("split") if match else args.get("split", ""),
            "seq_len": match.group("seq") if match else args.get("seq_len", ""),
            "overlap": match.group("overlap") if match else args.get("overlap", ""),
            "epochs": match.group("epoch") if match else args.get("epochs", ""),
            "lr": match.group("lr") if match else "",
            "repeat": match.group("repeat") if match else data.get("repeat", ""),
            "wer": data.get("wer", "") if isinstance(data, dict) else "",
            "ins_rate": data.get("ins_rate", "") if isinstance(data, dict) else "",
            "del_rate": data.get("del_rate", "") if isinstance(data, dict) else "",
            "sub_rate": data.get("sub_rate", "") if isinstance(data, dict) else "",
            "words": data.get("words", "") if isinstance(data, dict) else "",
            "path": path.name,
            "error": "",
        })
    return rows


def mean(values: list[float]) -> str:
    return f"{statistics.mean(values):.10f}" if values else ""


def stdev(values: list[float]) -> str:
    return f"{statistics.stdev(values):.10f}" if len(values) > 1 else ""


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        key = (
            str(row.get("dataset", "")),
            str(row.get("split", "")),
            str(row.get("seq_len", "")),
            str(row.get("overlap", "")),
            str(row.get("epochs", "")),
            str(row.get("lr", "")),
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
    return sorted(out, key=lambda row: (row["dataset"], int(row["epochs"] or 0), row["lr"]))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(rows: list[dict[str, object]], grouped_rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTC 2048-Context Self-Training Summary",
        "",
        f"Generated from `{ROOT}`.",
        "",
        f"Per-repeat rows: `{len(rows)}`.",
        "",
        "| Dataset | Epochs | LR | WER | Ins | Del | Sub |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped_rows:
        def pct(field: str) -> str:
            value = row.get(field, "")
            return f"{100 * float(value):.2f}%" if value != "" else ""

        lines.append(
            f"| {row.get('dataset', '')} | {row.get('epochs', '')} | {row.get('lr', '')} | {pct('wer_mean')} | {pct('ins_rate_mean')} | {pct('del_rate_mean')} | {pct('sub_rate_mean')} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    global ROOT, OUT_CSV, OUT_GROUPED_CSV, OUT_MD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Result directory containing ROB-56 CTC self-training PKL files",
    )
    args = parser.parse_args()
    ROOT = args.root.resolve()
    OUT_CSV = ROOT / "summary.csv"
    OUT_GROUPED_CSV = ROOT / "summary_by_setting.csv"
    OUT_MD = ROOT / "summary.md"

    rows = load_rows()
    row_fields = [
        "dataset", "split", "seq_len", "overlap", "epochs", "lr", "repeat",
        "wer", "ins_rate", "del_rate", "sub_rate", "words", "path", "error",
    ]
    grouped_rows = summarize(rows)
    grouped_fields = [
        "dataset", "split", "seq_len", "overlap", "epochs", "lr", "n", "wer_mean",
        "wer_std", "ins_rate_mean", "del_rate_mean", "sub_rate_mean", "words",
        "repeats", "paths",
    ]
    write_csv(OUT_CSV, rows, row_fields)
    write_csv(OUT_GROUPED_CSV, grouped_rows, grouped_fields)
    write_markdown(rows, grouped_rows)
    print(f"Wrote {len(rows)} rows")
    print(OUT_CSV)
    print(OUT_GROUPED_CSV)
    print(OUT_MD)


if __name__ == "__main__":
    main()
