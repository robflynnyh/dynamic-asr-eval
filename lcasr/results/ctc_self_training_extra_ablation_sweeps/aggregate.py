#!/usr/bin/env python3
"""Aggregate CTC self-training extra ablation sweep results.

Run from this directory or anywhere:
    python aggregate.py

Outputs:
  - summary.csv
  - summary.md
"""

from __future__ import annotations

import csv
import os
import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_CSV = ROOT / "summary.csv"
OUT_MD = ROOT / "summary.md"


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(ROOT.glob("**/*.pkl")):
        # Ignore nested/generated junk if any; current layout is group/*.pkl.
        rel = path.relative_to(ROOT)
        if len(rel.parts) < 2:
            group = "root"
        else:
            group = rel.parts[0]
        try:
            with path.open("rb") as f:
                data = pickle.load(f)
        except Exception as exc:  # keep aggregation robust
            rows.append({
                "group": group,
                "setting": path.stem,
                "lr": "",
                "wer": "",
                "ins_rate": "",
                "del_rate": "",
                "sub_rate": "",
                "words": "",
                "repeat": "",
                "path": str(rel),
                "error": repr(exc),
            })
            continue

        name = path.name.removesuffix(".pkl")
        name = re.sub(r"_\d+$", "", name)
        match = re.search(r"lr-(9em\d+)-(.+)$", name)
        lr = match.group(1) if match else ""
        setting = match.group(2) if match else name
        args = data.get("args_dict", {}) if isinstance(data, dict) else {}

        rows.append({
            "group": group,
            "setting": setting,
            "lr": lr,
            "wer": data.get("wer", "") if isinstance(data, dict) else "",
            "ins_rate": data.get("ins_rate", "") if isinstance(data, dict) else "",
            "del_rate": data.get("del_rate", "") if isinstance(data, dict) else "",
            "sub_rate": data.get("sub_rate", "") if isinstance(data, dict) else "",
            "words": data.get("words", "") if isinstance(data, dict) else "",
            "repeat": data.get("repeat", "") if isinstance(data, dict) else "",
            "dataset": args.get("dataset", "") if isinstance(args, dict) else "",
            "split": args.get("split", "") if isinstance(args, dict) else "",
            "epochs": args.get("epochs", "") if isinstance(args, dict) else "",
            "path": str(rel),
            "error": "",
        })
    return rows


def sort_key(row: dict[str, object]):
    wer = row.get("wer", "")
    try:
        return (float(wer), str(row.get("group", "")), str(row.get("lr", "")), str(row.get("setting", "")))
    except Exception:
        return (float("inf"), str(row.get("group", "")), str(row.get("lr", "")), str(row.get("setting", "")))


def fmt(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def write_outputs(rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=sort_key)
    fields = ["group", "lr", "setting", "wer", "ins_rate", "del_rate", "sub_rate", "words", "repeat", "dataset", "split", "epochs", "path", "error"]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    def wer_pct(row: dict[str, object]) -> str:
        try:
            return f"{100 * float(row.get('wer', '')):.2f}%"
        except Exception:
            return str(row.get('wer', ''))

    def lr_display(lr: object) -> str:
        return str(lr).replace("em", "e-")

    def clean_setting(setting: object) -> str:
        return str(setting).replace("freeze-subsampling-through-layer-", "freeze-through-")

    def add_table(title: str, subset: list[dict[str, object]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| WER | Setting |")
        lines.append("|---:|---|")
        for row in subset:
            lines.append(f"| {wer_pct(row)} | {clean_setting(row.get('setting',''))} |")
        lines.append("")

    lines = []
    lines.append("# CTC Self-Training Extra Ablation Sweep Summary")
    lines.append("")
    lines.append(f"Generated from `{ROOT}`.")
    lines.append("")
    lines.append("Focused view: **9e-5 only**, all rows in each category. Full metrics remain in `summary.csv`.")
    lines.append("")

    ok_rows = [r for r in rows if r.get('error', '') == '' and r.get('lr') == '9em5']
    for group in ["train_only", "progressive_top", "layer_type", "layer_drop_lr_sweep"]:
        subset = [r for r in ok_rows if r.get("group") == group]
        if subset:
            add_table(f"{group}, 9e-5", subset)

    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = load_rows()
    write_outputs(rows)
    print(f"Wrote {len(rows)} rows")
    print(OUT_CSV)
    print(OUT_MD)


if __name__ == "__main__":
    main()
