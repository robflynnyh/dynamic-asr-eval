"""Aggregate encoder-decoder dynamic-eval pickles by setting.

For each setting (filename minus trailing `_<repeat>.pkl`), this reports the
mean WER and error components across repeats. Filenames are parsed to expose
dataset, split, training mode, epoch, LR, augmentation, and agreement threshold
when present.

Run: `python aggregate.py` from this directory.
"""
import argparse
import csv
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path


REPEAT_RE = re.compile(r"_(\d+)\.pkl$")
SETTING_RE = re.compile(
    r"^(?P<dataset>.+)-(?P<split>dev|test)-(?P<mode>.+)-epoch-(?P<epoch>\d+)"
    r"-lr-(?P<lr>[^-]+)-(?P<aug>.+?)(?:-agree(?P<agreement>[^-]+))?$"
)


def _mean(values):
    return sum(values) / len(values)


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def parse_setting(setting: str) -> dict:
    match = SETTING_RE.match(setting)
    if match is None:
        return {"setting": setting}
    out = match.groupdict()
    out["setting"] = setting
    out["epoch"] = int(out["epoch"])
    mode = out.get("mode") or ""
    out["base_mode"] = mode
    out["decode"] = "greedy"
    for base_mode in (
        "adaptive_ce_ctc_aux_epoch_relabel",
        "teacher_kl_epoch_relabel",
        "teacher_ce_epoch_relabel",
        "ctc_aux_epoch_relabel",
        "adaptive_ce_ctc_aux",
        "teacher_kl",
        "teacher_ce",
        "ctc_aux",
        "grpo",
        "maxrl",
        "no_adapt",
        "baseline",
    ):
        if mode == base_mode:
            break
        if mode.startswith(base_mode + "-"):
            out["base_mode"] = base_mode
            out["decode"] = mode[len(base_mode) + 1:]
            break
    return out


def load_groups(directory: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(directory.glob("*.pkl")):
        match = REPEAT_RE.search(path.name)
        if match is None:
            continue
        setting = path.name[: match.start()]
        with open(path, "rb") as f:
            groups[setting].append(pickle.load(f))
    return groups


def aggregate(directory: Path) -> list[dict]:
    rows = []
    for setting, repeats in sorted(load_groups(directory).items()):
        wers = [float(rep["wer"]) for rep in repeats]
        row = parse_setting(setting)
        row.update({
            "wer": _mean(wers),
            "wer_std": _std(wers),
            "n_repeats": len(repeats),
            "words": _mean([float(rep["words"]) for rep in repeats]),
            "ins_rate": _mean([float(rep["ins_rate"]) for rep in repeats]),
            "del_rate": _mean([float(rep["del_rate"]) for rep in repeats]),
            "sub_rate": _mean([float(rep["sub_rate"]) for rep in repeats]),
            "repeat_ids": [rep.get("repeat") for rep in repeats],
        })
        rows.append(row)

    baselines = {
        (row.get("dataset"), row.get("split"), row.get("decode", "greedy")): row["wer"]
        for row in rows
        if row.get("base_mode") in {"baseline", "no_adapt"}
    }
    for row in rows:
        baseline = baselines.get((row.get("dataset"), row.get("split"), row.get("decode", "greedy")))
        row["delta_vs_baseline"] = None if baseline is None else row["wer"] - baseline

    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No result pickles found.", file=sys.stderr)
        return

    sorted_rows = sorted(rows, key=lambda r: (
        r.get("dataset", ""),
        r.get("mode", ""),
        r.get("lr", ""),
        r.get("aug", ""),
        r.get("agreement") or "",
    ))
    header = (
        "dataset", "mode", "lr", "aug", "agreement", "wer", "delta",
        "wer_std", "ins", "del", "sub", "n",
    )
    print("\t".join(header))
    for row in sorted_rows:
        print("\t".join([
            str(row.get("dataset", "")),
            str(row.get("mode", "")),
            str(row.get("lr", "")),
            str(row.get("aug", "")),
            str(row.get("agreement") or ""),
            f"{row['wer']:.5f}",
            "" if row["delta_vs_baseline"] is None else f"{row['delta_vs_baseline']:.5f}",
            f"{row['wer_std']:.5f}",
            f"{row['ins_rate']:.5f}",
            f"{row['del_rate']:.5f}",
            f"{row['sub_rate']:.5f}",
            str(row["n_repeats"]),
        ]))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the compact table")
    parser.add_argument("--csv", type=Path, default="summary.csv", help="Optional path to write CSV summary")
    args = parser.parse_args()

    results = aggregate(Path(__file__).parent)
    if args.csv is not None:
        write_csv(results, args.csv)
    if args.json:
        print(json.dumps(results, indent=2, default=float))
    else:
        print_table(results)
