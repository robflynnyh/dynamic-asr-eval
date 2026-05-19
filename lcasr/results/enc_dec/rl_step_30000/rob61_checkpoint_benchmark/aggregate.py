"""Aggregate ROB-61 encoder-decoder checkpoint benchmark pickles."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPEAT_RE = re.compile(r"_(\d+)\.pkl$")
SETTING_RE = re.compile(
    r"^(?P<dataset>.+)-(?P<split>dev|test)-(?P<checkpoint>.+)-"
    r"(?P<decode>greedy|beam5_lp0p5)-seq(?P<seq>\d+)-overlap(?P<overlap>\d+)$"
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5


def parse_setting(setting: str) -> dict[str, Any]:
    match = SETTING_RE.match(setting)
    if match is None:
        return {"setting": setting}
    row = match.groupdict()
    row["setting"] = setting
    row["seq"] = int(row["seq"])
    row["overlap"] = int(row["overlap"])
    return row


def load_groups(directory: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(directory.glob("*.pkl")):
        match = REPEAT_RE.search(path.name)
        if match is None:
            continue
        setting = path.name[: match.start()]
        with open(path, "rb") as handle:
            groups[setting].append(pickle.load(handle))
    return groups


def aggregate(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for setting, repeats in sorted(load_groups(directory).items()):
        row = parse_setting(setting)
        wers = [float(rep["wer"]) for rep in repeats]
        row.update({
            "wer": mean(wers),
            "wer_std": std(wers),
            "n_repeats": len(repeats),
            "words": mean([float(rep["words"]) for rep in repeats]),
            "ins_rate": mean([float(rep["ins_rate"]) for rep in repeats]),
            "del_rate": mean([float(rep["del_rate"]) for rep in repeats]),
            "sub_rate": mean([float(rep["sub_rate"]) for rep in repeats]),
            "repeat_ids": [rep.get("repeat") for rep in repeats],
        })
        rows.append(row)

    baselines = {
        (row.get("dataset"), row.get("split"), row.get("decode")): row["wer"]
        for row in rows
        if row.get("checkpoint") == "old_seed"
    }
    for row in rows:
        baseline = baselines.get((row.get("dataset"), row.get("split"), row.get("decode")))
        row["delta_vs_old_seed"] = None if baseline is None else row["wer"] - baseline
        row["relative_delta_vs_old_seed"] = (
            None if baseline in (None, 0) else (row["wer"] - baseline) / baseline
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outcome(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ROB-61 checkpoint benchmark",
        "",
        "Normal encoder-decoder inference benchmark; no self-training or test-time adaptation.",
        "Greedy uses the default autoregressive decode. Beam uses `beam_width=5` and `length_penalty=0.5`.",
        "",
        "| Dataset | Split | Decode | Checkpoint | WER | Delta vs old | Relative delta | Ins | Del | Sub |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (
        str(item.get("dataset", "")),
        str(item.get("split", "")),
        str(item.get("decode", "")),
        str(item.get("checkpoint", "")),
    )):
        delta = row.get("delta_vs_old_seed")
        rel_delta = row.get("relative_delta_vs_old_seed")
        lines.append(
            "| {dataset} | {split} | {decode} | {checkpoint} | {wer:.5f} | {delta} | {rel_delta} | "
            "{ins:.5f} | {dele:.5f} | {sub:.5f} |".format(
                dataset=row.get("dataset", ""),
                split=row.get("split", ""),
                decode=row.get("decode", ""),
                checkpoint=row.get("checkpoint", ""),
                wer=row["wer"],
                delta="" if delta is None else f"{delta:+.5f}",
                rel_delta="" if rel_delta is None else f"{rel_delta:+.2%}",
                ins=row["ins_rate"],
                dele=row["del_rate"],
                sub=row["sub_rate"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No result pickles found.", file=sys.stderr)
        return
    print("\t".join(("dataset", "split", "decode", "checkpoint", "wer", "delta_vs_old", "rel_delta", "n")))
    for row in sorted(rows, key=lambda item: (
        str(item.get("dataset", "")),
        str(item.get("split", "")),
        str(item.get("decode", "")),
        float(item.get("wer", 0.0)),
    )):
        delta = row.get("delta_vs_old_seed")
        rel_delta = row.get("relative_delta_vs_old_seed")
        print("\t".join([
            str(row.get("dataset", "")),
            str(row.get("split", "")),
            str(row.get("decode", "")),
            str(row.get("checkpoint", "")),
            f"{row['wer']:.5f}",
            "" if delta is None else f"{delta:+.5f}",
            "" if rel_delta is None else f"{rel_delta:+.5%}",
            str(row["n_repeats"]),
        ]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent / "pkl")
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "summary.csv")
    parser.add_argument("--outcome", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = aggregate(args.directory)
    if args.csv is not None:
        write_csv(rows, args.csv)
    if args.outcome is not None:
        write_outcome(rows, args.outcome)
    if args.json:
        print(json.dumps(rows, indent=2, default=float))
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
