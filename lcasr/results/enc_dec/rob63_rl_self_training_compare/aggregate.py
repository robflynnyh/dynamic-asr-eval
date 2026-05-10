"""Aggregate ROB-63 encoder-decoder self-training comparison pickles."""
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
    r"^(?P<dataset>.+)-(?P<split>dev|test)-(?P<checkpoint>old_seed|rl_step_30000)-"
    r"(?P<training_mode>teacher_ce|teacher_kl)-(?P<decode>.+)-epoch-(?P<epoch>\d+)"
    r"-lr-(?P<lr>[^-]+)-(?P<augmentation>.+)$"
)
DEFAULT_NORMAL_BASELINE_CSV = Path(__file__).parent.parent / "rob61_checkpoint_benchmark" / "summary.csv"


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
    row: dict[str, Any] = match.groupdict()
    row["setting"] = setting
    row["epoch"] = int(row["epoch"])
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


def load_normal_baselines(path: Path | None) -> dict[tuple[Any, ...], float]:
    if path is None or not path.exists():
        return {}
    baselines: dict[tuple[Any, ...], float] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("dataset"), row.get("split"), row.get("checkpoint"), row.get("decode"))
            baselines[key] = float(row["wer"])
    return baselines


def aggregate(directory: Path, normal_baseline_csv: Path | None = DEFAULT_NORMAL_BASELINE_CSV) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for setting, repeats in sorted(load_groups(directory).items()):
        wers = [float(rep["wer"]) for rep in repeats]
        row = parse_setting(setting)
        row.update(
            {
                "wer": mean(wers),
                "wer_std": std(wers),
                "n_repeats": len(repeats),
                "words": mean([float(rep["words"]) for rep in repeats]),
                "ins_rate": mean([float(rep["ins_rate"]) for rep in repeats]),
                "del_rate": mean([float(rep["del_rate"]) for rep in repeats]),
                "sub_rate": mean([float(rep["sub_rate"]) for rep in repeats]),
                "repeat_ids": [rep.get("repeat") for rep in repeats],
            }
        )
        rows.append(row)

    seed_by_cell = {
        (
            row.get("dataset"),
            row.get("split"),
            row.get("training_mode"),
            row.get("decode"),
            row.get("epoch"),
            row.get("lr"),
            row.get("augmentation"),
        ): row["wer"]
        for row in rows
        if row.get("checkpoint") == "old_seed"
    }
    normal_baselines = load_normal_baselines(normal_baseline_csv)
    for row in rows:
        seed = seed_by_cell.get(
            (
                row.get("dataset"),
                row.get("split"),
                row.get("training_mode"),
                row.get("decode"),
                row.get("epoch"),
                row.get("lr"),
                row.get("augmentation"),
            )
        )
        normal = normal_baselines.get(
            (row.get("dataset"), row.get("split"), row.get("checkpoint"), row.get("decode"))
        )
        row["delta_vs_old_seed"] = None if seed is None else row["wer"] - seed
        row["relative_delta_vs_old_seed"] = None if seed in (None, 0) else (row["wer"] - seed) / seed
        row["normal_decode_wer"] = normal
        row["delta_vs_normal_decode"] = None if normal is None else row["wer"] - normal
        row["relative_delta_vs_normal_decode"] = None if normal in (None, 0) else (row["wer"] - normal) / normal
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
        "# ROB-63 RL self-training comparison",
        "",
        "Snapshot generated from completed ROB-63 pickles.",
        "RL deltas compare each RL `step_30000` self-training cell against the matching old-seed cell.",
        "Normal deltas compare each self-training cell against the same checkpoint's normal decoding WER.",
        "",
        "| Dataset | Mode | LR | Augmentation | Checkpoint | WER | Normal WER | Delta vs normal | Relative vs normal | Delta vs old seed | Relative vs old seed | Ins | Del | Sub |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            str(item.get("dataset", "")),
            str(item.get("training_mode", "")),
            str(item.get("lr", "")),
            str(item.get("augmentation", "")),
            str(item.get("checkpoint", "")),
        ),
    ):
        delta = row.get("delta_vs_old_seed")
        rel_delta = row.get("relative_delta_vs_old_seed")
        normal = row.get("normal_decode_wer")
        normal_delta = row.get("delta_vs_normal_decode")
        normal_rel_delta = row.get("relative_delta_vs_normal_decode")
        lines.append(
            "| {dataset} | {mode} | {lr} | {aug} | {checkpoint} | {wer:.5f} | {normal} | "
            "{normal_delta} | {normal_rel_delta} | {delta} | {rel_delta} | "
            "{ins:.5f} | {dele:.5f} | {sub:.5f} |".format(
                dataset=row.get("dataset", ""),
                mode=row.get("training_mode", ""),
                lr=row.get("lr", ""),
                aug=row.get("augmentation", ""),
                checkpoint=row.get("checkpoint", ""),
                wer=row["wer"],
                normal="" if normal is None else f"{normal:.5f}",
                normal_delta="" if normal_delta is None else f"{normal_delta:+.5f}",
                normal_rel_delta="" if normal_rel_delta is None else f"{normal_rel_delta:+.2%}",
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
    print(
        "\t".join(
            (
                "dataset",
                "mode",
                "lr",
                "aug",
                "checkpoint",
                "wer",
                "normal_wer",
                "delta_vs_normal",
                "rel_vs_normal",
                "delta_vs_old",
                "rel_vs_old",
                "n",
            )
        )
    )
    for row in sorted(
        rows,
        key=lambda item: (
            str(item.get("dataset", "")),
            str(item.get("training_mode", "")),
            str(item.get("lr", "")),
            str(item.get("augmentation", "")),
            float(item.get("wer", 0.0)),
        ),
    ):
        delta = row.get("delta_vs_old_seed")
        rel_delta = row.get("relative_delta_vs_old_seed")
        normal = row.get("normal_decode_wer")
        normal_delta = row.get("delta_vs_normal_decode")
        normal_rel_delta = row.get("relative_delta_vs_normal_decode")
        print(
            "\t".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("training_mode", "")),
                    str(row.get("lr", "")),
                    str(row.get("augmentation", "")),
                    str(row.get("checkpoint", "")),
                    f"{row['wer']:.5f}",
                    "" if normal is None else f"{normal:.5f}",
                    "" if normal_delta is None else f"{normal_delta:+.5f}",
                    "" if normal_rel_delta is None else f"{normal_rel_delta:+.5%}",
                    "" if delta is None else f"{delta:+.5f}",
                    "" if rel_delta is None else f"{rel_delta:+.5%}",
                    str(row["n_repeats"]),
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent / "pkl")
    parser.add_argument(
        "--normal-baseline-csv",
        type=Path,
        default=DEFAULT_NORMAL_BASELINE_CSV,
        help="Normal decoding summary CSV used for per-checkpoint self-training deltas.",
    )
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "summary.csv")
    parser.add_argument("--outcome", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = aggregate(args.directory, args.normal_baseline_csv)
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
