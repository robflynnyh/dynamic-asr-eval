"""Aggregate ROB-94 fixed-setting encoder-decoder thesis rows."""
from __future__ import annotations

import argparse
import csv
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULT_ROOT = Path(__file__).parent
ENC_DEC_ROOT = RESULT_ROOT.parent
REPEAT_RE = re.compile(r"_(\d+)\.pkl$")
SETTING_RE = re.compile(
    r"^(?P<dataset>.+)-(?P<split>dev|test)-"
    r"(?:(?P<checkpoint>enc_dec_v2|old_seed|rl_step_30000)-)?"
    r"(?P<training_mode>teacher_ce|teacher_kl)-(?P<decode>.+)-"
    r"epoch-(?P<epoch>\d+)-lr-(?P<lr>[^-]+)-(?P<augmentation>.+)$"
)
FIXED_SETTING = {
    "training_mode": "teacher_ce",
    "decode": "beam5_lp0p5",
    "epoch": 1,
    "lr": "1em7",
    "augmentation": "freq3_width24_time0",
}
CHECKPOINT_PATHS = {
    "enc_dec_v2": "/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt",
    "old_seed": "/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt",
    "rl_step_30000": "/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt",
}
CHECKPOINTS = ("enc_dec_v2", "old_seed", "rl_step_30000")
DATASETS = ("tedlium", "earnings22", "chime6", "rev16")
SPLITS = ("dev", "test")
DEFAULT_SOURCES = (
    (RESULT_ROOT / "pkl", None),
    (ENC_DEC_ROOT / "enc_dec_v2" / "enc_dec_dynamic_eval", "enc_dec_v2"),
    (ENC_DEC_ROOT / "rl_step_30000" / "rob63_rl_self_training_compare" / "pkl", None),
    (ENC_DEC_ROOT / "rl_step_30000" / "rob63_best_ce_remaining_datasets" / "pkl", None),
)


def parse_setting(path: Path, default_checkpoint: str | None) -> dict[str, Any] | None:
    match = REPEAT_RE.search(path.name)
    if match is None:
        return None
    setting = path.name[: match.start()]
    parsed = SETTING_RE.match(setting)
    if parsed is None:
        return None
    row: dict[str, Any] = parsed.groupdict()
    if row["checkpoint"] is None:
        row["checkpoint"] = default_checkpoint
    if row["checkpoint"] is None:
        return None
    row["epoch"] = int(row["epoch"])
    row["setting"] = setting
    return row


def is_fixed_row(row: dict[str, Any]) -> bool:
    return all(row.get(key) == value for key, value in FIXED_SETTING.items())


def source_label(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_rows(sources: list[tuple[Path, str | None]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for directory, default_checkpoint in sources:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.pkl")):
            parsed = parse_setting(path, default_checkpoint)
            if parsed is None or not is_fixed_row(parsed):
                continue
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            key = (parsed["checkpoint"], parsed["dataset"], parsed["split"])
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
    for key, repeats in sorted(grouped.items()):
        checkpoint, dataset, split = key
        rows.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_path": CHECKPOINT_PATHS[checkpoint],
                "dataset": dataset,
                "split": split,
                **FIXED_SETTING,
                "status": "complete",
                "wer": sum(row["wer"] for row in repeats) / len(repeats),
                "words": sum(row["words"] for row in repeats) / len(repeats),
                "ins_rate": sum(row["ins_rate"] for row in repeats) / len(repeats),
                "del_rate": sum(row["del_rate"] for row in repeats) / len(repeats),
                "sub_rate": sum(row["sub_rate"] for row in repeats) / len(repeats),
                "n_repeats": len(repeats),
                "repeat_ids": ";".join(str(row["repeat"]) for row in repeats),
                "source": ";".join(row["source"] for row in repeats),
                "note": "",
            }
        )
    return rows


def expected_rows(observed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["checkpoint"], row["dataset"], row["split"]): row
        for row in observed
    }
    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        for dataset in DATASETS:
            for split in SPLITS:
                key = (checkpoint, dataset, split)
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
                        "checkpoint": checkpoint,
                        "checkpoint_path": CHECKPOINT_PATHS[checkpoint],
                        "dataset": dataset,
                        "split": split,
                        **FIXED_SETTING,
                        "status": status,
                        "wer": "",
                        "words": "",
                        "ins_rate": "",
                        "del_rate": "",
                        "sub_rate": "",
                        "n_repeats": 0,
                        "repeat_ids": "",
                        "source": "",
                        "note": note,
                    }
                )
    return rows


def row_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        CHECKPOINTS.index(row["checkpoint"]),
        DATASETS.index(row["dataset"]),
        SPLITS.index(row["split"]),
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "dataset",
        "split",
        "status",
        "wer",
        "training_mode",
        "lr",
        "augmentation",
        "decode",
        "epoch",
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


def format_wer(value: Any) -> str:
    return "" if value == "" else f"{float(value):.5f}"


def write_outcome(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = [row for row in rows if row["status"] == "complete"]
    missing = [row for row in rows if row["status"] == "missing"]
    unavailable = [row for row in rows if row["status"] == "unavailable"]
    lines = [
        "# ROB-94 fixed-setting encoder-decoder thesis rows",
        "",
        "Fixed setting: `teacher_ce`, `lr=1e-7`, `freq3_width24_time0`, `beam5_lp0p5`, epoch `1`, no filtering.",
        "",
        f"Completed rows: {len(complete)}/24. Missing rows: {len(missing)}. Unavailable rows: {len(unavailable)}.",
        "",
        "Rev16 dev rows are marked unavailable because the current `lcasr/rev16` loader and `enc_dec_dynamic_eval_test.py` expose Rev16 test only.",
        "",
        "## Thesis Table",
        "",
        "| Checkpoint | Dataset | Split | Status | WER | Source / note |",
        "|---|---|---|---|---:|---|",
    ]
    for row in rows:
        source_note = row["source"] if row["status"] == "complete" else row["note"]
        lines.append(
            "| {checkpoint} | {dataset} | {split} | {status} | {wer} | {source_note} |".format(
                checkpoint=row["checkpoint"],
                dataset=row["dataset"],
                split=row["split"],
                status=row["status"],
                wer=format_wer(row["wer"]),
                source_note=source_note,
            )
        )
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
        ]
    )
    for checkpoint in CHECKPOINTS:
        lines.append(f"- `{checkpoint}`: `{CHECKPOINT_PATHS[checkpoint]}`")
    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "From `lcasr/`:",
            "",
            "```bash",
            "python results/enc_dec/rob94_fixed_setting_thesis/aggregate.py \\",
            "  --csv results/enc_dec/rob94_fixed_setting_thesis/summary.csv \\",
            "  --outcome results/enc_dec/rob94_fixed_setting_thesis/OUTCOME.md",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_source(value: str) -> tuple[Path, str | None]:
    if ":" not in value:
        return Path(value), None
    checkpoint, directory = value.split(":", 1)
    return Path(directory), checkpoint or None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Optional source as '<default-checkpoint>:<directory>' or '<directory>'. Defaults cover ROB-94 and existing ROB-63/enc_dec_v2 artifacts.",
    )
    parser.add_argument("--csv", type=Path, default=RESULT_ROOT / "summary.csv")
    parser.add_argument("--outcome", type=Path, default=RESULT_ROOT / "OUTCOME.md")
    args = parser.parse_args()

    sources = [parse_source(item) for item in args.source] if args.source else list(DEFAULT_SOURCES)
    rows = sorted(expected_rows(load_rows(sources)), key=row_sort_key)
    write_csv(rows, args.csv)
    write_outcome(rows, args.outcome)
    print(f"wrote {args.csv} and {args.outcome}")
    print(f"complete={sum(row['status'] == 'complete' for row in rows)} total={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
