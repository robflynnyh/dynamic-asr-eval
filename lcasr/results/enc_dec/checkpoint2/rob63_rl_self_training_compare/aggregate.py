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
CHECKPOINT_PATHS = {
    "old_seed": "/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_no_anorm_V2_lr_2e3_ctcw_0_05/step_210720.pt",
    "rl_step_30000": "/store/store5/data/acp21rjf_checkpoints/lcasr/rob61_rl_floras50_30k_b36_r24_grpo_wer70_cer30_std001_const_lr_1e-5/step_30000.pt",
}
CHECKPOINT_DESCRIPTIONS = {
    "old_seed": "normal encoder-decoder seed checkpoint",
    "rl_step_30000": "30K RL-trained checkpoint from the ROB-61/PR #11 lineage",
}
CHECKPOINT_VIEW_NAMES = {
    "old_seed": "checkpoint1",
    "rl_step_30000": "checkpoint2",
}
CHECKPOINT_VIEW_TITLES = {
    "old_seed": "Checkpoint 1: old_seed",
    "rl_step_30000": "Checkpoint 2: rl_step_30000",
}


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


def load_groups(directories: list[Path]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for directory in directories:
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


def aggregate(
    directory: Path | list[Path],
    normal_baseline_csv: Path | None = DEFAULT_NORMAL_BASELINE_CSV,
) -> list[dict[str, Any]]:
    directories = [directory] if isinstance(directory, Path) else directory
    rows: list[dict[str, Any]] = []
    for setting, repeats in sorted(load_groups(directories).items()):
        wers = [float(rep["wer"]) for rep in repeats]
        row = parse_setting(setting)
        row.update(
            {
                "checkpoint_path": CHECKPOINT_PATHS.get(str(row.get("checkpoint", ""))),
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


def format_cell(row: dict[str, Any]) -> str:
    return (
        f"{row.get('checkpoint')} {row.get('training_mode')} "
        f"lr={row.get('lr')} {row.get('augmentation')}"
    )


def format_range(values: list[float]) -> str:
    return f"{min(values):+.5f} to {max(values):+.5f}"


def format_wer(value: float | None) -> str:
    return "" if value is None else f"{value:.5f}"


def format_signed(value: float | None) -> str:
    return "" if value is None else f"{value:+.5f}"


def format_percent(value: float | None) -> str:
    return "" if value is None else f"{value:+.2%}"


def comparable_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("dataset"),
        row.get("split"),
        row.get("training_mode"),
        row.get("lr"),
        row.get("augmentation"),
        row.get("decode"),
        row.get("epoch"),
    )


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("dataset", "")),
        str(row.get("split", "")),
        str(row.get("training_mode", "")),
        str(row.get("lr", "")),
        str(row.get("augmentation", "")),
        str(row.get("checkpoint", "")),
    )


def best_rows_by_dataset(rows: list[dict[str, Any]], checkpoint: str) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("checkpoint") != checkpoint:
            continue
        key = (str(row.get("dataset", "")), str(row.get("split", "")))
        current = best.get(key)
        if current is None or float(row["wer"]) < float(current["wer"]):
            best[key] = row
    return [best[key] for key in sorted(best)]


def write_checkpoint_readme(checkpoint: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    view_name = CHECKPOINT_VIEW_NAMES[checkpoint]
    lines = [
        f"# ROB-63 {CHECKPOINT_VIEW_TITLES[checkpoint]}",
        "",
        "This folder is the checkpoint-specific view of the ROB-63 encoder-decoder",
        "self-training results. The top-level files are generated from the combined",
        "ROB-63 result pickles and normal-decoding baselines. Nested folders under",
        "`checkpoint1/` and `checkpoint2/` hold the moved raw logs, PKLs, and",
        "per-run outcomes.",
        "",
        "## Checkpoint",
        "",
        f"- Folder: `results/enc_dec/{view_name}/`",
        f"- Key: `{checkpoint}`",
        f"- Description: {CHECKPOINT_DESCRIPTIONS[checkpoint]}",
        f"- Path: `{CHECKPOINT_PATHS[checkpoint]}`",
        "",
        "## Files",
        "",
        "| File | Meaning |",
        "|---|---|",
        "| `summary.csv` | All ROB-63 adapted rows for this checkpoint, with matching normal/unadapted WER where available. |",
        "| `OUTCOME.md` | Human-readable best-row summary plus the full checkpoint-specific table. |",
        "| Nested result folders | Raw/result folders grouped by checkpoint family; only `checkpoint1/` and `checkpoint2/` should be direct children of `results/enc_dec/`. |",
        "",
        "Regenerate from `lcasr/` with:",
        "",
        "```bash",
        "python results/enc_dec/checkpoint2/rob63_rl_self_training_compare/aggregate.py \\",
        "  --directory results/enc_dec/checkpoint2/rob63_rl_self_training_compare/pkl \\",
        "  --extra-directory results/enc_dec/checkpoint2/rob63_best_ce_remaining_datasets/pkl \\",
        "  --extra-directory results/enc_dec/checkpoint2/rob63_lower_lr_dev_followup/pkl \\",
        "  --extra-directory results/enc_dec/checkpoint2/rob63_aug_followup/pkl \\",
        "  --extra-directory results/enc_dec/checkpoint2/rob63_strong_aug_filter_followup/pkl \\",
        "  --extra-directory results/enc_dec/checkpoint2/rob63_targeted_high_aug_followup/pkl \\",
        "  --csv results/enc_dec/checkpoint2/rob63_rl_self_training_compare/combined_summary.csv \\",
        "  --outcome results/enc_dec/checkpoint2/rob63_rl_self_training_compare/COMBINED_OUTCOME.md \\",
        "  --checkpoint-view-root results/enc_dec \\",
        "  --top-level-outcome results/enc_dec/OUTCOME.md",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_checkpoint_outcome(checkpoint: str, rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_rows = [row for row in rows if row.get("checkpoint") == checkpoint]
    lines = [
        f"# {CHECKPOINT_VIEW_TITLES[checkpoint]}",
        "",
        f"`{checkpoint}` is the {CHECKPOINT_DESCRIPTIONS[checkpoint]}.",
        "",
        f"Checkpoint path: `{CHECKPOINT_PATHS[checkpoint]}`",
        "",
        "Rows are one-epoch self-training snapshots. `Normal WER` is the matching",
        "unadapted beam5/lp0.5 decode for the same checkpoint when available.",
        "",
        "## Best Rows",
        "",
        "| Dataset | Split | Mode | LR | Augmentation | Normal WER | Adapted WER | Adapted vs normal |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for row in best_rows_by_dataset(rows, checkpoint):
        lines.append(
            "| {dataset} | {split} | {mode} | {lr} | {aug} | {normal} | {wer} | {delta} |".format(
                dataset=row.get("dataset", ""),
                split=row.get("split", ""),
                mode=row.get("training_mode", ""),
                lr=row.get("lr", ""),
                aug=row.get("augmentation", ""),
                normal=format_wer(row.get("normal_decode_wer")),
                wer=format_wer(float(row["wer"])),
                delta=format_signed(row.get("delta_vs_normal_decode")),
            )
        )
    lines.extend(
        [
            "",
            "## Full Checkpoint Table",
            "",
            "| Dataset | Split | Mode | LR | Augmentation | Adapted WER | Normal WER | Delta vs normal | Relative vs normal | Ins | Del | Sub |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(checkpoint_rows, key=row_sort_key):
        lines.append(
            "| {dataset} | {split} | {mode} | {lr} | {aug} | {wer} | {normal} | {delta} | {rel_delta} | "
            "{ins:.5f} | {dele:.5f} | {sub:.5f} |".format(
                dataset=row.get("dataset", ""),
                split=row.get("split", ""),
                mode=row.get("training_mode", ""),
                lr=row.get("lr", ""),
                aug=row.get("augmentation", ""),
                wer=format_wer(float(row["wer"])),
                normal=format_wer(row.get("normal_decode_wer")),
                delta=format_signed(row.get("delta_vs_normal_decode")),
                rel_delta=format_percent(row.get("relative_delta_vs_normal_decode")),
                ins=row["ins_rate"],
                dele=row["del_rate"],
                sub=row["sub_rate"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def write_checkpoint_views(rows: list[dict[str, Any]], root: Path) -> None:
    for checkpoint, view_name in CHECKPOINT_VIEW_NAMES.items():
        checkpoint_rows = [row for row in rows if row.get("checkpoint") == checkpoint]
        directory = root / view_name
        write_csv(checkpoint_rows, directory / "summary.csv")
        write_checkpoint_readme(checkpoint, directory / "README.md")
        write_checkpoint_outcome(checkpoint, rows, directory / "OUTCOME.md")


def write_top_level_checkpoint_outcome(rows: list[dict[str, Any]], path: Path) -> None:
    by_key = {(row.get("checkpoint"), comparable_key(row)): row for row in rows}
    lines = [
        "# Encoder-decoder ROB-63 checkpoint outcome",
        "",
        "This top-level readout is intentionally checkpoint-focused. It shows the best",
        "completed adapted row for the newest checkpoint and the matched old-seed row",
        "for the same dataset, split, mode, learning rate, augmentation, decode, and",
        "epoch. All rows are single-repeat snapshots.",
        "",
        "## Checkpoint Folders",
        "",
        "| Folder | Checkpoint | Meaning |",
        "|---|---|---|",
    ]
    for checkpoint, view_name in CHECKPOINT_VIEW_NAMES.items():
        lines.append(
            f"| `{view_name}/` | `{checkpoint}` | {CHECKPOINT_DESCRIPTIONS[checkpoint]} at `{CHECKPOINT_PATHS[checkpoint]}` |"
        )
    lines.extend(
        [
            "",
        "Checkpoint-specific summaries live in `checkpoint1/OUTCOME.md` and",
        "`checkpoint2/OUTCOME.md`. The full paired ROB-63 table remains in",
        "`checkpoint2/rob63_rl_self_training_compare/COMBINED_OUTCOME.md`.",
            "",
            "## Best Newest-Checkpoint Rows",
            "",
            "`checkpoint2` is the newest RL-trained checkpoint. `checkpoint1` is the",
            "normal encoder-decoder seed checkpoint. Baselines are unadapted beam5/lp0.5",
            "decodes for each checkpoint.",
            "",
            "| Dataset | Split | Best checkpoint2 setting | Checkpoint1 normal WER | Checkpoint1 matched adapted WER | Checkpoint2 normal WER | Checkpoint2 best adapted WER | Checkpoint2 adapted vs normal | Checkpoint2 adapted vs checkpoint1 adapted |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rl in best_rows_by_dataset(rows, "rl_step_30000"):
        old = by_key.get(("old_seed", comparable_key(rl)))
        setting = f"{rl.get('training_mode')} lr={rl.get('lr')} {rl.get('augmentation')}"
        lines.append(
            "| {dataset} | {split} | `{setting}` | {old_normal} | {old_adapted} | {rl_normal} | "
            "{rl_adapted} | {rl_delta} | {rl_vs_old} |".format(
                dataset=rl.get("dataset", ""),
                split=rl.get("split", ""),
                setting=setting,
                old_normal="" if old is None else format_wer(old.get("normal_decode_wer")),
                old_adapted="" if old is None else format_wer(float(old["wer"])),
                rl_normal=format_wer(rl.get("normal_decode_wer")),
                rl_adapted=format_wer(float(rl["wer"])),
                rl_delta=format_signed(rl.get("delta_vs_normal_decode")),
                rl_vs_old=format_signed(rl.get("delta_vs_old_seed")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- TED-LIUM and Earnings22 are the cleanest wins for the RL checkpoint under",
            "  the completed one-epoch self-training grid.",
            "- Rev16 is sensitive to learning rate and masking: the best RL row is a small",
            "  gain over both its own normal decode and the matched old-seed adapted row,",
            "  but nearby higher-LR cells collapse.",
            "- CHiME-6 remains deletion-dominated or near 1.0 WER after adaptation, so it",
            "  should be treated as a failed adaptation setting rather than a checkpoint win.",
            "- The older `checkpoint1/enc_dec_dynamic_eval/` folder uses a different historical",
            "  checkpoint, `enc_dec_v2/step_105360.pt`; use the ROB-63 checkpoint folders",
            "  for seed-vs-RL comparisons.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def describe_grid(rows: list[dict[str, Any]]) -> str:
    dimensions = [
        ("datasets", "dataset"),
        ("splits", "split"),
        ("checkpoints", "checkpoint"),
        ("training modes", "training_mode"),
        ("learning rates", "lr"),
        ("frequency-mask settings", "augmentation"),
    ]
    parts = []
    for label, key in dimensions:
        count = len({str(row.get(key, "")) for row in rows if row.get(key) is not None})
        parts.append(f"{count} {label}")
    return ", ".join(parts)


def write_summary(lines: list[str], rows: list[dict[str, Any]]) -> None:
    has_filtered_rows = any("filter" in str(row.get("augmentation", "")) for row in rows)
    filter_note = (
        "- Teacher filtering is encoded in the augmentation label when present; "
        "`basic_repeat_filter` rows use the light repeat/length teacher filter."
        if has_filtered_rows
        else "- The sweep used no teacher filtering."
    )
    lines.extend(
        [
            "## Summary",
            "",
            f"- Completed {len(rows)} one-epoch result rows covering {describe_grid(rows)}.",
            f"{filter_note} `Delta vs old seed` is only meaningful for the RL rows "
            "because old-seed rows are the matching-cell reference.",
        ]
    )
    dataset_splits = sorted(
        {
            (str(row.get("dataset", "")), str(row.get("split", "")))
            for row in rows
        }
    )
    for dataset, split in dataset_splits:
        dataset_rows = [
            row for row in rows if row.get("dataset") == dataset and row.get("split") == split
        ]
        rl_rows = [row for row in dataset_rows if row.get("checkpoint") == "rl_step_30000"]
        old_rows = [row for row in dataset_rows if row.get("checkpoint") == "old_seed"]
        if not rl_rows or not old_rows:
            continue
        rl_deltas = [
            float(row["delta_vs_old_seed"])
            for row in rl_rows
            if row.get("delta_vs_old_seed") is not None
        ]
        better_count = sum(delta < 0 for delta in rl_deltas)
        best_rl = min(rl_rows, key=lambda row: float(row["wer"]))
        best_old = min(old_rows, key=lambda row: float(row["wer"]))
        old_normal_rel = [
            float(row["relative_delta_vs_normal_decode"])
            for row in old_rows
            if row.get("relative_delta_vs_normal_decode") is not None
        ]
        rl_normal_rel = [
            float(row["relative_delta_vs_normal_decode"])
            for row in rl_rows
            if row.get("relative_delta_vs_normal_decode") is not None
        ]
        summary = (
            f"- {dataset}/{split}: RL `step_30000` beats the matching old-seed cell in "
            f"{better_count}/{len(rl_rows)} cells; RL-vs-old absolute WER deltas span "
            f"{format_range(rl_deltas)}. Best RL cell is {format_cell(best_rl)} at "
            f"{best_rl['wer']:.5f} WER; best old-seed cell is {format_cell(best_old)} "
            f"at {best_old['wer']:.5f} WER."
        )
        if old_normal_rel and rl_normal_rel:
            summary += (
                " Relative change vs normal decoding spans "
                f"{min(rl_normal_rel):+.2%} to {max(rl_normal_rel):+.2%} for RL and "
                f"{min(old_normal_rel):+.2%} to {max(old_normal_rel):+.2%} for old seed."
            )
        else:
            summary += " Matching normal-decoding baselines were not available for this dataset in the ROB-61 CSV."
        lines.append(summary)

    outliers = [
        row
        for row in rows
        if row.get("relative_delta_vs_normal_decode") is not None
        and float(row["relative_delta_vs_normal_decode"]) > 0.25
    ]
    for row in sorted(outliers, key=lambda item: float(item["relative_delta_vs_normal_decode"]), reverse=True):
        lines.append(
            f"- Sanity note: {row.get('dataset')} {format_cell(row)} is an outlier versus normal decoding "
            f"({float(row['relative_delta_vs_normal_decode']):+.2%}, WER {row['wer']:.5f}); "
            f"the error mix is dominated by deletion rate {row['del_rate']:.5f}."
        )
    lines.append("")


def write_paired_comparison(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "## Unadapted vs adapted WER",
            "",
            "This is the main readout. `Old normal WER` uses the `old_seed` checkpoint from "
            "the checkpoint key above, and `RL normal WER` uses the `rl_step_30000` checkpoint "
            "from the same key. Both are unadapted beam5/lp0.5 decoding baselines for the "
            "matching checkpoint. The adapted columns are the one-epoch self-training WERs "
            "for the listed setting.",
            "",
            "| Dataset | Split | Mode | LR | Augmentation | Old normal WER | Old adapted WER | Old adapted vs normal | RL normal WER | RL adapted WER | RL adapted vs normal | RL adapted vs old adapted |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            row.get("dataset"),
            row.get("split"),
            row.get("training_mode"),
            row.get("lr"),
            row.get("augmentation"),
            row.get("decode"),
            row.get("epoch"),
        )
        grouped[key][str(row.get("checkpoint"))] = row

    for key in sorted(grouped):
        dataset, split, mode, lr, augmentation, _decode, _epoch = key
        old = grouped[key].get("old_seed")
        rl = grouped[key].get("rl_step_30000")
        if old is None and rl is None:
            continue
        old_normal = None if old is None else old.get("normal_decode_wer")
        old_adapted = None if old is None else float(old["wer"])
        old_delta = None if old is None else old.get("delta_vs_normal_decode")
        rl_normal = None if rl is None else rl.get("normal_decode_wer")
        rl_adapted = None if rl is None else float(rl["wer"])
        rl_delta = None if rl is None else rl.get("delta_vs_normal_decode")
        rl_vs_old = None if rl is None else rl.get("delta_vs_old_seed")
        lines.append(
            "| {dataset} | {split} | {mode} | {lr} | {augmentation} | {old_normal} | {old_adapted} | "
            "{old_delta} | {rl_normal} | {rl_adapted} | {rl_delta} | {rl_vs_old} |".format(
                dataset=dataset,
                split=split,
                mode=mode,
                lr=lr,
                augmentation=augmentation,
                old_normal=format_wer(old_normal),
                old_adapted=format_wer(old_adapted),
                old_delta=format_signed(old_delta),
                rl_normal=format_wer(rl_normal),
                rl_adapted=format_wer(rl_adapted),
                rl_delta=format_signed(rl_delta),
                rl_vs_old=format_signed(rl_vs_old),
            )
        )
    lines.append("")


def write_outcome(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ROB-63 RL self-training comparison",
        "",
        "Snapshot generated from completed ROB-63 pickles.",
        "RL deltas compare each RL `step_30000` self-training cell against the matching old-seed cell.",
        "Normal/unadapted deltas compare each self-training cell against the same checkpoint's normal decoding WER.",
        "",
        "## Checkpoint Key",
        "",
    ]
    for checkpoint, checkpoint_path in CHECKPOINT_PATHS.items():
        lines.append(f"- `{checkpoint}`: {CHECKPOINT_DESCRIPTIONS[checkpoint]} at `{checkpoint_path}`.")
    lines.extend(
        [
            "",
            "Every row in the full table also carries the exact `checkpoint_path` used for that row.",
            "",
        ]
    )
    if rows:
        write_summary(lines, rows)
        write_paired_comparison(lines, rows)
    lines.extend(
        [
            "## Full table",
            "",
            "| Dataset | Split | Mode | LR | Augmentation | Checkpoint | Checkpoint path | Adapted WER | Unadapted WER | Delta vs unadapted | Relative vs unadapted | Delta vs old seed | Relative vs old seed | Ins | Del | Sub |",
            "|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        rows,
        key=lambda item: (
            str(item.get("dataset", "")),
            str(item.get("split", "")),
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
            "| {dataset} | {split} | {mode} | {lr} | {aug} | {checkpoint} | `{checkpoint_path}` | {wer:.5f} | {normal} | "
            "{normal_delta} | {normal_rel_delta} | {delta} | {rel_delta} | "
            "{ins:.5f} | {dele:.5f} | {sub:.5f} |".format(
                dataset=row.get("dataset", ""),
                split=row.get("split", ""),
                mode=row.get("training_mode", ""),
                lr=row.get("lr", ""),
                aug=row.get("augmentation", ""),
                checkpoint=row.get("checkpoint", ""),
                checkpoint_path=row.get("checkpoint_path", ""),
                wer=row["wer"],
                normal=format_wer(normal),
                normal_delta=format_signed(normal_delta),
                normal_rel_delta=format_percent(normal_rel_delta),
                delta=format_signed(delta),
                rel_delta=format_percent(rel_delta),
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
                "split",
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
            str(item.get("split", "")),
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
                    str(row.get("split", "")),
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
        "--extra-directory",
        type=Path,
        action="append",
        default=[],
        help="Additional pickle directories to include in the same aggregate.",
    )
    parser.add_argument(
        "--normal-baseline-csv",
        type=Path,
        default=DEFAULT_NORMAL_BASELINE_CSV,
        help="Normal decoding summary CSV used for per-checkpoint self-training deltas.",
    )
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "summary.csv")
    parser.add_argument("--outcome", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-view-root",
        type=Path,
        default=None,
        help="Optional results/enc_dec root where checkpoint1/ and checkpoint2/ views are written.",
    )
    parser.add_argument(
        "--top-level-outcome",
        type=Path,
        default=None,
        help="Optional top-level results/enc_dec/OUTCOME.md checkpoint-focused summary.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = aggregate([args.directory, *args.extra_directory], args.normal_baseline_csv)
    if args.csv is not None:
        write_csv(rows, args.csv)
    if args.outcome is not None:
        write_outcome(rows, args.outcome)
    if args.checkpoint_view_root is not None:
        write_checkpoint_views(rows, args.checkpoint_view_root)
    if args.top_level_outcome is not None:
        write_top_level_checkpoint_outcome(rows, args.top_level_outcome)
    if args.json:
        print(json.dumps(rows, indent=2, default=float))
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
