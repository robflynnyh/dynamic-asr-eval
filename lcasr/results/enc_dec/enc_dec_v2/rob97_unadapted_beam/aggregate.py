"""Aggregate ROB-97 enc_dec_v2 unadapted beam-search rows."""
from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path
from typing import Any


RESULT_ROOT = Path(__file__).parent
REPEAT_RE = re.compile(r"_(\d+)\.pkl$")
SETTING_RE = re.compile(
    r"^(?P<dataset>.+)-(?P<split>dev|test)-enc_dec_v2-"
    r"(?P<training_mode>no_adapt)-(?P<decode>.+)-"
    r"epoch-(?P<epoch>\d+)-lr-(?P<lr>[^-]+)-(?P<augmentation>.+)$"
)
CHECKPOINT = "enc_dec_v2"
CHECKPOINT_PATH = "/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt"
EXPECTED_ROWS = (("chime6", "test"), ("rev16", "test"))
FIXED_SETTING = {
    "training_mode": "no_adapt",
    "decode": "beam5_lp0p5",
    "epoch": 0,
    "lr": "none",
    "augmentation": "no_aug",
}


def parse_setting(path: Path) -> dict[str, Any] | None:
    match = REPEAT_RE.search(path.name)
    if match is None:
        return None
    parsed = SETTING_RE.match(path.name[: match.start()])
    if parsed is None:
        return None
    row: dict[str, Any] = parsed.groupdict()
    row["epoch"] = int(row["epoch"])
    return row


def is_fixed_row(row: dict[str, Any]) -> bool:
    return all(row.get(key) == value for key, value in FIXED_SETTING.items())


def source_label(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_rows(directory: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if directory.exists():
        for path in sorted(directory.glob("*.pkl")):
            parsed = parse_setting(path)
            if parsed is None or not is_fixed_row(parsed):
                continue
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            key = (parsed["dataset"], parsed["split"])
            grouped.setdefault(key, []).append(
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

    rows: list[dict[str, Any]] = []
    for dataset, split in EXPECTED_ROWS:
        repeats = grouped.get((dataset, split), [])
        if not repeats:
            rows.append(
                {
                    "checkpoint": CHECKPOINT,
                    "checkpoint_path": CHECKPOINT_PATH,
                    "dataset": dataset,
                    "split": split,
                    **FIXED_SETTING,
                    "status": "missing",
                    "wer": "",
                    "words": "",
                    "ins_rate": "",
                    "del_rate": "",
                    "sub_rate": "",
                    "n_repeats": 0,
                    "repeat_ids": "",
                    "source": "",
                    "note": "Expected ROB-97 result artifact is not present yet.",
                }
            )
            continue
        rows.append(
            {
                "checkpoint": CHECKPOINT,
                "checkpoint_path": CHECKPOINT_PATH,
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
    lines = [
        "# ROB-97 enc_dec_v2 unadapted beam-search rows",
        "",
        "Fixed setting: `no_adapt`, `beam5_lp0p5`, epoch `0`, no augmentation.",
        "",
        f"Completed rows: {len(complete)}/2. Missing rows: {len(missing)}.",
        "",
        "## ROB-96 Rows",
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
            "## Checkpoint",
            "",
            f"- `{CHECKPOINT}`: `{CHECKPOINT_PATH}`",
            "",
            "## Regeneration",
            "",
            "From `lcasr/`:",
            "",
            "```bash",
            "python results/enc_dec/enc_dec_v2/rob97_unadapted_beam/aggregate.py \\",
            "  --csv results/enc_dec/enc_dec_v2/rob97_unadapted_beam/summary.csv \\",
            "  --outcome results/enc_dec/enc_dec_v2/rob97_unadapted_beam/OUTCOME.md",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=RESULT_ROOT / "pkl")
    parser.add_argument("--csv", type=Path, default=RESULT_ROOT / "summary.csv")
    parser.add_argument("--outcome", type=Path, default=RESULT_ROOT / "OUTCOME.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.directory)
    write_csv(rows, args.csv)
    write_outcome(rows, args.outcome)
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"Wrote {args.csv} and {args.outcome}; complete={complete} total={len(rows)}")


if __name__ == "__main__":
    main()
