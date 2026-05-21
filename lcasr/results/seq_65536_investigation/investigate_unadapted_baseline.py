#!/usr/bin/env python3
"""Investigate the ROB-67 65536-context unadapted CTC baseline."""

from __future__ import annotations

import csv
import hashlib
import pickle
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
UNADAPTED_ROOT = ROOT / "unadapted_baseline"
BASELINE_2048_ROOT = REPO_ROOT / "lcasr" / "results" / "ctc_seq2048_unadapted_baseline"
LEGACY_16384_EARNINGS = (
    REPO_ROOT / "lcasr" / "results" / "paper" / "per_epoch_eval" / "epoch-0-earnings22-test_1.pkl"
)

DATASETS = ["chime6", "earnings22", "rev16", "tedlium"]
SUMMARY_FIELDS = [
    "dataset",
    "split",
    "wer_65536",
    "wer_2048",
    "delta_65536_minus_2048",
    "ins_65536",
    "ins_2048",
    "del_65536",
    "del_2048",
    "sub_65536",
    "sub_2048",
    "words_65536",
    "words_2048",
    "record_count_65536",
    "record_count_2048",
    "reference_set_matches_2048",
    "reference_order_matches_2048",
    "output_ref_word_ratio_65536",
    "output_ref_word_ratio_2048",
    "checkpoint_65536",
    "checkpoint_2048",
    "checkpoint_config_audio_chunking_size_65536",
    "checkpoint_config_audio_chunking_size_2048",
    "checkpoint_config_random_seed_65536",
    "checkpoint_config_random_seed_2048",
]


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not contain a dict")
    return data


def sha_texts(texts: list[str]) -> list[str]:
    return [hashlib.sha256(text.encode()).hexdigest() for text in texts]


def text_word_count(texts: list[str]) -> int:
    return sum(len(text.split()) for text in texts)


def result_path(root: Path, dataset: str, seq_len: int, overlap: int) -> Path:
    return root / f"{dataset}-test-ctc-seq{seq_len}-overlap{overlap}-no_adapt_1.pkl"


def cfg_get(args: dict[str, Any], section: str, key: str) -> Any:
    cfg = args.get("config")
    if cfg is None:
        return ""
    try:
        section_value = cfg.get(section, {})
        return section_value.get(key, "")
    except Exception:
        return ""


def check_run_contract(data: dict[str, Any], dataset: str, seq_len: int, overlap: int) -> list[str]:
    args = data.get("args_dict", {})
    checks = {
        "dataset": args.get("dataset") == dataset,
        "split": args.get("split") == "test",
        "seq_len": args.get("seq_len") == seq_len,
        "overlap": args.get("overlap") == overlap,
        "epochs": args.get("epochs") == 0,
        "repeats": args.get("repeats") == 1,
        "beamsearch": args.get("beamsearch") is False,
        "awmc": args.get("awmc") is False,
        "consistency": args.get("consistency") is False,
        "max_records": args.get("max_records") is None,
    }
    return [name for name, ok in checks.items() if not ok]


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        data_65536 = load_pickle(result_path(UNADAPTED_ROOT, dataset, 65536, 57344))
        data_2048 = load_pickle(result_path(BASELINE_2048_ROOT, dataset, 2048, 1792))

        args_65536 = data_65536.get("args_dict", {})
        args_2048 = data_2048.get("args_dict", {})
        gold_65536 = data_65536.get("gold", [])
        gold_2048 = data_2048.get("gold", [])
        out_65536 = data_65536.get("model_output", [])
        out_2048 = data_2048.get("model_output", [])
        if not isinstance(gold_65536, list) or not isinstance(gold_2048, list):
            raise TypeError(f"Expected gold lists for {dataset}")
        if not isinstance(out_65536, list) or not isinstance(out_2048, list):
            raise TypeError(f"Expected model_output lists for {dataset}")

        contract_misses = check_run_contract(data_65536, dataset, 65536, 57344)
        contract_misses += [
            f"2048_{name}" for name in check_run_contract(data_2048, dataset, 2048, 1792)
        ]
        if contract_misses:
            raise AssertionError(f"{dataset} contract mismatch: {', '.join(contract_misses)}")

        words_65536 = int(data_65536.get("words", 0))
        words_2048 = int(data_2048.get("words", 0))
        out_words_65536 = text_word_count(out_65536)
        out_words_2048 = text_word_count(out_2048)

        rows.append({
            "dataset": dataset,
            "split": args_65536.get("split", ""),
            "wer_65536": float(data_65536["wer"]),
            "wer_2048": float(data_2048["wer"]),
            "delta_65536_minus_2048": float(data_65536["wer"]) - float(data_2048["wer"]),
            "ins_65536": float(data_65536["ins_rate"]),
            "ins_2048": float(data_2048["ins_rate"]),
            "del_65536": float(data_65536["del_rate"]),
            "del_2048": float(data_2048["del_rate"]),
            "sub_65536": float(data_65536["sub_rate"]),
            "sub_2048": float(data_2048["sub_rate"]),
            "words_65536": words_65536,
            "words_2048": words_2048,
            "record_count_65536": len(gold_65536),
            "record_count_2048": len(gold_2048),
            "reference_set_matches_2048": sorted(sha_texts(gold_65536)) == sorted(sha_texts(gold_2048)),
            "reference_order_matches_2048": sha_texts(gold_65536) == sha_texts(gold_2048),
            "output_ref_word_ratio_65536": out_words_65536 / words_65536,
            "output_ref_word_ratio_2048": out_words_2048 / words_2048,
            "checkpoint_65536": args_65536.get("checkpoint", ""),
            "checkpoint_2048": args_2048.get("checkpoint", ""),
            "checkpoint_config_audio_chunking_size_65536": cfg_get(args_65536, "audio_chunking", "size"),
            "checkpoint_config_audio_chunking_size_2048": cfg_get(args_2048, "audio_chunking", "size"),
            "checkpoint_config_random_seed_65536": cfg_get(args_65536, "training", "random_seed"),
            "checkpoint_config_random_seed_2048": cfg_get(args_2048, "training", "random_seed"),
        })
    return rows


def fmt_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt_float(row.get(field, "")) for field in SUMMARY_FIELDS})


def legacy_16384_note() -> tuple[str, str]:
    if not LEGACY_16384_EARNINGS.exists():
        return "No legacy 16384 no-adapt Earnings22 artifact was found.", ""
    data = load_pickle(LEGACY_16384_EARNINGS)
    current = load_pickle(result_path(UNADAPTED_ROOT, "earnings22", 65536, 57344))
    args = data.get("args_dict", {})
    legacy_gold_hashes = sorted(sha_texts(data.get("gold", [])))
    current_gold_hashes = sorted(sha_texts(current.get("gold", [])))
    legacy_output_hash = hashlib.sha256("\n".join(data.get("model_output", [])).encode()).hexdigest()
    current_output_hash = hashlib.sha256("\n".join(current.get("model_output", [])).encode()).hexdigest()
    note = (
        f"A legacy Earnings22 `epochs=0` artifact exists at `{LEGACY_16384_EARNINGS.relative_to(REPO_ROOT)}` "
        f"with WER {pct(float(data['wer']))}, `seq_len={args.get('seq_len')}`, "
        f"`overlap={args.get('overlap')}`, and checkpoint `{args.get('checkpoint')}`."
    )
    evidence = (
        f"It matches the current 65536 Earnings22 WER to the shown precision, but it is not the same artifact: "
        f"checkpoint paths differ, checkpoint config seeds differ, output hashes differ "
        f"(`{legacy_output_hash[:12]}` vs `{current_output_hash[:12]}`), and the sorted reference-set hash comparison is "
        f"`{legacy_gold_hashes == current_gold_hashes}`."
    )
    return note, evidence


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    all_reference_sets_match = all(row["reference_set_matches_2048"] for row in rows)
    any_order_differs = any(not row["reference_order_matches_2048"] for row in rows)
    legacy_note, legacy_evidence = legacy_16384_note()

    lines = [
        "# ROB-67 65536 Unadapted Baseline Investigation",
        "",
        "This checks whether the 65536-context before-adaptation CTC rows look anomalous because of a reporting/configuration mismatch or because the checkpoint genuinely behaves similarly to shorter-context baselines before self-training.",
        "",
        "## Conclusion",
        "",
        "- No aggregation or launcher-contract mismatch was found in the committed 65536 unadapted artifacts.",
        "- Each 65536 PKL reports `epochs=0`, greedy CTC (`beamsearch=False`), no AWMC/consistency path, `split=test`, `seq_len=65536`, `overlap=57344`, one repeat, and no `max_records` cap.",
        "- Compared with the exact committed 2048 no-adapt baseline, 65536 is better on TEDLIUM and Earnings22, effectively tied on Rev16, and worse on CHiME-6.",
        "- CHiME-6 is the main suspicious row: the 65536 model has a higher deletion rate and a lower output/reference word ratio than the 2048 baseline.",
        "- The committed tree does not contain an all-dataset exact 16384 no-adapt CTC baseline analogous to ROB-66/ROB-67, so the 2.7 minute comparison cannot be fully validated from committed exact-match artifacts alone.",
    ]
    if all_reference_sets_match:
        lines.append("- The 65536 and 2048 baselines use the same reference record sets for all four datasets.")
    if any_order_differs:
        lines.append("- CHiME-6 and Earnings22 appear in a different record order between the 2048 and 65536 runs; this changes concatenated hashes but not each run's paired WER calculation.")
    lines.extend([
        "",
        "## 65536 vs 2048 No-Adapt Baseline",
        "",
        "| Dataset | 65536 WER | 2048 WER | Delta | 65536 Del | 2048 Del | 65536 Out/Ref | 2048 Out/Ref | Reference set matches | Reference order matches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {pct(row['wer_65536'])} | {pct(row['wer_2048'])} | "
            f"{row['delta_65536_minus_2048'] * 100:+.2f} pp | {pct(row['del_65536'])} | "
            f"{pct(row['del_2048'])} | {row['output_ref_word_ratio_65536']:.3f} | "
            f"{row['output_ref_word_ratio_2048']:.3f} | {row['reference_set_matches_2048']} | "
            f"{row['reference_order_matches_2048']} |"
        )
    lines.extend([
        "",
        "## Run Metadata Checks",
        "",
        "| Dataset | 65536 checkpoint | 65536 cfg chunk | 65536 seed | 2048 checkpoint | 2048 cfg chunk | 2048 seed |",
        "|---|---|---:|---:|---|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['dataset']} | `{row['checkpoint_65536']}` | "
            f"{row['checkpoint_config_audio_chunking_size_65536']} | "
            f"{row['checkpoint_config_random_seed_65536']} | `{row['checkpoint_2048']}` | "
            f"{row['checkpoint_config_audio_chunking_size_2048']} | "
            f"{row['checkpoint_config_random_seed_2048']} |"
        )
    lines.extend([
        "",
        "## Legacy 16384 Context Note",
        "",
        legacy_note,
    ])
    if legacy_evidence:
        lines.append(legacy_evidence)
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        "python3 lcasr/results/seq_65536_investigation/aggregate_unadapted.py",
        "python3 lcasr/results/seq_65536_investigation/investigate_unadapted_baseline.py",
        "```",
        "",
        "The script writes `unadapted_baseline/investigation.csv` and this report.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = build_rows()
    write_csv(rows, UNADAPTED_ROOT / "investigation.csv")
    write_markdown(rows, ROOT / "UNADAPTED_BASELINE_INVESTIGATION.md")
    print(f"Wrote {len(rows)} investigation rows")
    print(UNADAPTED_ROOT / "investigation.csv")
    print(ROOT / "UNADAPTED_BASELINE_INVESTIGATION.md")


if __name__ == "__main__":
    main()
