#!/usr/bin/env python3
"""Aggregate ROB-110 CTC 8192-context unadapted baseline results."""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent

RESULT_RE = re.compile(
    r"(?P<dataset>.+)-(?P<split>dev|test)-ctc-seq(?P<seq>\d+)"
    r"-overlap(?P<overlap>\d+)-(?P<mode>no_adapt)_(?P<repeat>\d+)\.pkl$"
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

        args = data.get("args_dict", {}) if isinstance(data, dict) else {}
        rows.append({
            "dataset": match.group("dataset"),
            "split": match.group("split"),
            "seq_len": match.group("seq"),
            "overlap": match.group("overlap"),
            "mode": match.group("mode"),
            "epochs": args.get("epochs", 0),
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
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("error"):
            continue
        key = (
            str(row["dataset"]),
            str(row["split"]),
            str(row["seq_len"]),
            str(row["overlap"]),
            str(row["mode"]),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for (dataset, split, seq_len, overlap, mode), items in grouped.items():
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
            "mode": mode,
            "epochs": "0",
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
    return sorted(out, key=lambda row: str(row["dataset"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "dataset",
        "split",
        "seq_len",
        "overlap",
        "mode",
        "epochs",
        "n",
        "wer_mean",
        "wer_std",
        "ins_rate_mean",
        "del_rate_mean",
        "sub_rate_mean",
        "words",
        "repeats",
        "paths",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, rows: list[dict[str, object]], source: Path) -> None:
    lines = [
        "# CTC 8192-Context Unadapted Baseline Summary",
        "",
        f"Generated from `{source}`.",
        "",
        "| Dataset | Split | WER | Ins | Del | Sub | Words |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def pct(field: str) -> str:
            value = row.get(field, "")
            return f"{100 * float(value):.2f}%" if value != "" else ""

        lines.append(
            f"| {row.get('dataset', '')} | {row.get('split', '')} | "
            f"{pct('wer_mean')} | {pct('ins_rate_mean')} | "
            f"{pct('del_rate_mean')} | {pct('sub_rate_mean')} | "
            f"{row.get('words', '')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Result directory containing ROB-110 8192 no-adapt baseline PKL files",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_rows(root)
    summary_rows = summarize(rows)
    write_csv(root / "summary.csv", summary_rows)
    write_markdown(root / "summary.md", summary_rows, root)
    print(f"Wrote {len(summary_rows)} dataset rows from {len(rows)} result pickles")
    print(root / "summary.csv")
    print(root / "summary.md")


if __name__ == "__main__":
    main()
