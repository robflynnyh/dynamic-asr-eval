"""Aggregate ROB-122 ROB-81 Floras-finetuned encoder-decoder eval rows."""
from __future__ import annotations

import argparse
import csv
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULT_ROOT = Path(__file__).parent
REPEAT_RE = re.compile(r"_(\d+)\.pkl$")
SETTING_RE = re.compile(
    r"^(?P<dataset>.+)-(?P<split>dev|test)-"
    r"(?P<checkpoint>rob81_floras50)-"
    r"(?P<mode>no_adapt|teacher_ce)-(?P<decode>.+)-"
    r"epoch-(?P<epoch>\d+)-lr-(?P<lr>[^-]+)-(?P<augmentation>.+)$"
)
CHECKPOINT = "rob81_floras50"
CHECKPOINT_PATH = (
    "/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/"
    "supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0/"
    "step_323484.pt"
)
DATASETS = ("tedlium", "earnings22", "chime6", "rev16")
SPLITS = ("dev", "test")
MODES = ("no_adapt", "teacher_ce")
FIXED_SETTING = {
    "decode": "beam5_lp0p5",
    "seq_len": 2048,
    "overlap": 0,
}
MODE_SETTINGS = {
    "no_adapt": {"epoch": 0, "lr": "none", "augmentation": "no_aug"},
    "teacher_ce": {"epoch": 1, "lr": "1em7", "augmentation": "freq3_width24_time0"},
}


def source_label(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def parse_setting(path: Path) -> dict[str, Any] | None:
    match = REPEAT_RE.search(path.name)
    if match is None:
        return None
    setting = path.name[: match.start()]
    parsed = SETTING_RE.match(setting)
    if parsed is None:
        return None
    row: dict[str, Any] = parsed.groupdict()
    row["epoch"] = int(row["epoch"])
    row["setting"] = setting
    return row


def is_expected_setting(row: dict[str, Any]) -> bool:
    mode_settings = MODE_SETTINGS.get(row["mode"])
    if mode_settings is None:
        return False
    return (
        row["checkpoint"] == CHECKPOINT
        and row["decode"] == FIXED_SETTING["decode"]
        and row["epoch"] == mode_settings["epoch"]
        and row["lr"] == mode_settings["lr"]
        and row["augmentation"] == mode_settings["augmentation"]
    )


def load_rows(directory: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    if not directory.exists():
        return []
    for path in sorted(directory.glob("*.pkl")):
        parsed = parse_setting(path)
        if parsed is None or not is_expected_setting(parsed):
            continue
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        key = (parsed["dataset"], parsed["split"], parsed["mode"])
        grouped[key].append(
            {
                **parsed,
                "wer": float(payload["wer"]),
                "words": float(payload.get("words", 0.0)),
                "ins_rate": float(payload.get("ins_rate", 0.0)),
                "del_rate": float(payload.get("del_rate", 0.0)),
                "sub_rate": float(payload.get("sub_rate", 0.0)),
                "repeat": payload.get("repeat", ""),
                "source": source_label(path),
            }
        )

    rows = []
    for (dataset, split, mode), repeats in sorted(grouped.items()):
        rows.append(
            {
                "checkpoint": CHECKPOINT,
                "checkpoint_path": CHECKPOINT_PATH,
                "dataset": dataset,
                "split": split,
                "mode": mode,
                "status": "complete",
                **FIXED_SETTING,
                **MODE_SETTINGS[mode],
                "wer": sum(row["wer"] for row in repeats) / len(repeats),
                "words": sum(row["words"] for row in repeats) / len(repeats),
                "ins_rate": sum(row["ins_rate"] for row in repeats) / len(repeats),
                "del_rate": sum(row["del_rate"] for row in repeats) / len(repeats),
                "sub_rate": sum(row["sub_rate"] for row in repeats) / len(repeats),
                "n_repeats": len(repeats),
                "repeat_ids": ";".join(str(row["repeat"]) for row in repeats),
                "source": ";".join(row["source"] for row in repeats),
                "delta_vs_no_adapt": "",
                "note": "",
            }
        )
    return rows


def expected_rows(observed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["dataset"], row["split"], row["mode"]): row
        for row in observed
    }
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for split in SPLITS:
            for mode in MODES:
                key = (dataset, split, mode)
                if key in by_key:
                    rows.append(by_key[key])
                    continue
                status = "missing"
                note = ""
                if dataset == "rev16" and split == "dev":
                    status = "unavailable"
                    note = "Current lcasr/rev16 loader exposes test only; no dev manifest is wired into the runner."
                rows.append(
                    {
                        "checkpoint": CHECKPOINT,
                        "checkpoint_path": CHECKPOINT_PATH,
                        "dataset": dataset,
                        "split": split,
                        "mode": mode,
                        "status": status,
                        **FIXED_SETTING,
                        **MODE_SETTINGS[mode],
                        "wer": "",
                        "words": "",
                        "ins_rate": "",
                        "del_rate": "",
                        "sub_rate": "",
                        "n_repeats": 0,
                        "repeat_ids": "",
                        "source": "",
                        "delta_vs_no_adapt": "",
                        "note": note,
                    }
                )
    return rows


def add_deltas(rows: list[dict[str, Any]]) -> None:
    baselines = {
        (row["dataset"], row["split"]): row
        for row in rows
        if row["mode"] == "no_adapt" and row["status"] == "complete"
    }
    for row in rows:
        if row["mode"] != "teacher_ce" or row["status"] != "complete":
            continue
        baseline = baselines.get((row["dataset"], row["split"]))
        if baseline is None:
            continue
        row["delta_vs_no_adapt"] = float(row["wer"]) - float(baseline["wer"])


def row_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        DATASETS.index(row["dataset"]),
        SPLITS.index(row["split"]),
        MODES.index(row["mode"]),
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "dataset",
        "split",
        "mode",
        "status",
        "wer",
        "delta_vs_no_adapt",
        "decode",
        "epoch",
        "lr",
        "augmentation",
        "seq_len",
        "overlap",
        "n_repeats",
        "words",
        "ins_rate",
        "del_rate",
        "sub_rate",
        "checkpoint_path",
        "source",
        "note",
        "repeat_ids",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any) -> str:
    return "" if value == "" else f"{float(value):.5f}"


def write_outcome(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = [row for row in rows if row["status"] == "complete"]
    missing = [row for row in rows if row["status"] == "missing"]
    unavailable = [row for row in rows if row["status"] == "unavailable"]
    lines = [
        "# ROB-122 ROB-81 Floras-finetuned encoder-decoder eval",
        "",
        "Fixed adaptation setting: `teacher_ce`, `lr=1e-7`, `freq3_width24_time0`, `beam5_lp0p5`, epoch `1`, no filtering.",
        "Matched baseline setting: `no_adapt`, `beam5_lp0p5`, epoch `0`, `optim_lr=0.0`, no augmentation.",
        "",
        f"Completed rows: {len(complete)}/16. Missing rows: {len(missing)}. Unavailable rows: {len(unavailable)}.",
        "",
        "Rev16 dev rows are marked unavailable because the current `lcasr/rev16` loader exposes test only.",
        "",
        "## Checkpoint",
        "",
        f"`{CHECKPOINT_PATH}`",
        "",
        "## Rows",
        "",
        "| Dataset | Split | Mode | Status | WER | Delta vs no-adapt | Source / note |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        source_or_note = row["source"] or row["note"]
        lines.append(
            "| {dataset} | {split} | `{mode}` | {status} | {wer} | {delta} | {source} |".format(
                dataset=row["dataset"],
                split=row["split"],
                mode=row["mode"],
                status=row["status"],
                wer=format_float(row["wer"]),
                delta=format_float(row["delta_vs_no_adapt"]),
                source=source_or_note,
            )
        )
    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "From `lcasr/`:",
            "",
            "```bash",
            "python results/enc_dec/rob81_floras50_finetune_eval/aggregate.py \\",
            "  --csv results/enc_dec/rob81_floras50_finetune_eval/summary.csv \\",
            "  --outcome results/enc_dec/rob81_floras50_finetune_eval/OUTCOME.md",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=RESULT_ROOT / "pkl")
    parser.add_argument("--csv", type=Path, default=RESULT_ROOT / "summary.csv")
    parser.add_argument("--outcome", type=Path, default=RESULT_ROOT / "OUTCOME.md")
    args = parser.parse_args()

    rows = expected_rows(load_rows(args.directory))
    add_deltas(rows)
    rows.sort(key=row_sort_key)
    write_csv(rows, args.csv)
    write_outcome(rows, args.outcome)


if __name__ == "__main__":
    main()
