#!/usr/bin/env python3
"""Aggregate ROB-57 entropy trace JSONL files."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
TRACE_ROWS_CSV = ROOT / "trace_rows.csv"
PER_STEP_CSV = ROOT / "entropy_by_update.csv"


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        with path.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    rows.append({
                        "path": str(path.relative_to(ROOT)),
                        "line": line_no,
                        "error": str(exc),
                    })
                    continue
                row["path"] = str(path.relative_to(ROOT))
                row["line"] = line_no
                row["error"] = ""
                rows.append(row)
    return rows


def write_trace_rows(rows: list[dict[str, object]]) -> None:
    fields = [
        "dataset",
        "split",
        "setting",
        "measurement",
        "update_step",
        "epoch",
        "chunk_key",
        "chunk_order",
        "record_index",
        "record_id",
        "repeat_index",
        "repeats",
        "mean_entropy",
        "std_entropy",
        "num_frames",
        "chunk_len",
        "loss",
        "seq_len",
        "overlap",
        "epochs",
        "optim_lr",
        "spec_augment_n_freq_masks",
        "spec_augment_freq_mask_param",
        "spec_augment_n_time_masks",
        "checkpoint",
        "path",
        "line",
        "error",
    ]
    with TRACE_ROWS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def mean(values: list[float]) -> str:
    return f"{statistics.mean(values):.10f}" if values else ""


def std(values: list[float]) -> str:
    return f"{statistics.stdev(values):.10f}" if len(values) > 1 else ""


def write_per_step(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        try:
            key = (
                str(row.get("dataset", "")),
                str(row.get("split", "")),
                str(row.get("setting", "")),
                str(row.get("measurement", "")),
                int(row.get("update_step", 0)),
            )
            float(row.get("mean_entropy", ""))
        except (TypeError, ValueError):
            continue
        grouped[key].append(row)

    fields = [
        "dataset",
        "split",
        "setting",
        "measurement",
        "update_step",
        "n",
        "mean_entropy",
        "std_entropy",
        "min_entropy",
        "max_entropy",
        "epochs",
        "optim_lr",
        "spec_augment_n_freq_masks",
        "spec_augment_freq_mask_param",
        "spec_augment_n_time_masks",
        "paths",
    ]
    out_rows = []
    for (dataset, split, setting, measurement, update_step), items in grouped.items():
        values = [float(item["mean_entropy"]) for item in items]
        out_rows.append({
            "dataset": dataset,
            "split": split,
            "setting": setting,
            "measurement": measurement,
            "update_step": update_step,
            "n": len(values),
            "mean_entropy": mean(values),
            "std_entropy": std(values),
            "min_entropy": f"{min(values):.10f}",
            "max_entropy": f"{max(values):.10f}",
            "epochs": items[0].get("epochs", ""),
            "optim_lr": items[0].get("optim_lr", ""),
            "spec_augment_n_freq_masks": items[0].get("spec_augment_n_freq_masks", ""),
            "spec_augment_freq_mask_param": items[0].get("spec_augment_freq_mask_param", ""),
            "spec_augment_n_time_masks": items[0].get("spec_augment_n_time_masks", ""),
            "paths": " ".join(sorted({str(item.get("path", "")) for item in items})),
        })

    out_rows = sorted(out_rows, key=lambda row: (
        row["dataset"],
        row["setting"],
        row["measurement"],
        int(row["update_step"]),
    ))
    with PER_STEP_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    rows = load_rows()
    write_trace_rows(rows)
    write_per_step(rows)
    print(f"Wrote {TRACE_ROWS_CSV}")
    print(f"Wrote {PER_STEP_CSV}")


if __name__ == "__main__":
    main()
