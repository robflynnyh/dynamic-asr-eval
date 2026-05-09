#!/usr/bin/env python3
"""Run per-utterance ROB-55 majority-vote diagnostics on TEDLIUM dev."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

LCASR_ROOT = Path(__file__).resolve().parents[3]
if str(LCASR_ROOT) not in sys.path:
    sys.path.insert(0, str(LCASR_ROOT))

import lcasr
import lib
from enc_dec_teacher_filters import add_enc_dec_teacher_filter_args
from lcasr.eval.utils import zero_out_spectogram
from lcasr.eval.wer import word_error_rate_detail
from lcasr.utils.audio_tools import load_tokenizer, processing_chain
from lcasr.utils.general import get_model_class, load_model
from tedlium.run import fetch_utterances, get_text_and_audio, proc_stm_and_timings


normalize = EnglishTextNormalizer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt",
    )
    parser.add_argument("--output-dir", default="results/enc_dec/enc_dec_majority_vote_utterance_diagnostic")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--max-utterances", type=int, default=0, help="0 means all matching utterances")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-duration", type=float, default=0.0, help="0 means no duration cap")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--training-mode", default="teacher_kl", choices=["teacher_ce", "teacher_kl"])
    parser.add_argument("--optim-lr", type=float, default=1e-7)
    parser.add_argument("--teacher-kl-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-vote-num-samples", type=int, default=8)
    parser.add_argument("--teacher-vote-temperature", type=float, default=0.7)
    parser.add_argument("--teacher-vote-min-count", type=int, default=2)
    parser.add_argument("--teacher-vote-similarity", type=float, default=1.0)
    parser.add_argument("--teacher-vote-representative-strategy", default="medoid", choices=["first", "medoid"])
    parser.add_argument("--enc-dec-beam-width", type=int, default=5)
    parser.add_argument("--enc-dec-length-penalty", type=float, default=0.5)
    parser.add_argument("--enc-dec-eos-bias", type=float, default=0.0)
    parser.add_argument("--enc-dec-repetition-penalty", type=float, default=0.0)
    parser.add_argument("--enc-dec-no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--enc-dec-max-generate", type=int, default=-1)
    parser.add_argument("--spec-augment-freq-mask-param", type=int, default=24)
    parser.add_argument("--spec-augment-n-time-masks", type=int, default=0)
    parser.add_argument("--spec-augment-n-freq-masks", type=int, default=3)
    parser.add_argument("--disable-flash-attention", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    add_enc_dec_teacher_filter_args(parser)
    return parser.parse_args()


def load_enc_dec_model(args: argparse.Namespace):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if args.disable_flash_attention:
        config.model.flash_attn = False
    tokenizer = load_tokenizer()
    model = load_model(config, model_class=get_model_class(config), vocab_size=len(tokenizer))
    model.print_total_params()
    model.load_state_dict(checkpoint["model"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.device = device
    model = model.to(device).eval()
    print(f"Loaded model from {args.checkpoint} on {device}")
    return model, tokenizer, config


def make_eval_args(args: argparse.Namespace, config, epochs: int, diagnostics_path: str = ""):
    return SimpleNamespace(
        config=config,
        epochs=epochs,
        seq_len=args.seq_len,
        overlap=args.overlap,
        training_mode=args.training_mode,
        teacher_kl_temperature=args.teacher_kl_temperature,
        teacher_epoch_relabel=False,
        teacher_vote_num_samples=args.teacher_vote_num_samples,
        teacher_vote_temperature=args.teacher_vote_temperature,
        teacher_vote_min_count=args.teacher_vote_min_count,
        teacher_vote_similarity=args.teacher_vote_similarity,
        teacher_vote_include_deterministic=False,
        teacher_vote_representative_strategy=args.teacher_vote_representative_strategy,
        teacher_diagnostics_path=diagnostics_path,
        teacher_diagnostics_context={},
        teacher_filter_max_length=args.teacher_filter_max_length,
        teacher_min_frames_per_token=args.teacher_min_frames_per_token,
        teacher_filter_max_consecutive_token_repeat=args.teacher_filter_max_consecutive_token_repeat,
        teacher_max_consecutive_token_repeat=args.teacher_max_consecutive_token_repeat,
        teacher_filter_repeated_token_ngrams=args.teacher_filter_repeated_token_ngrams,
        teacher_repeated_token_ngram_sizes=args.teacher_repeated_token_ngram_sizes,
        teacher_repeated_token_ngram_min_repeats=args.teacher_repeated_token_ngram_min_repeats,
        teacher_filter_decode_agreement=args.teacher_filter_decode_agreement,
        teacher_decode_agreement_temperature=args.teacher_decode_agreement_temperature,
        teacher_decode_agreement_min_similarity=args.teacher_decode_agreement_min_similarity,
        teacher_filter_low_confidence=args.teacher_filter_low_confidence,
        teacher_min_mean_max_prob=args.teacher_min_mean_max_prob,
        teacher_max_mean_entropy=args.teacher_max_mean_entropy,
        teacher_filter_repeated_words=args.teacher_filter_repeated_words,
        teacher_max_consecutive_word_repeat=args.teacher_max_consecutive_word_repeat,
        teacher_filter_ctc_agreement=args.teacher_filter_ctc_agreement,
        teacher_ctc_agreement_min_similarity=args.teacher_ctc_agreement_min_similarity,
        enc_dec_beam_width=args.enc_dec_beam_width,
        enc_dec_length_penalty=args.enc_dec_length_penalty,
        enc_dec_eos_bias=args.enc_dec_eos_bias,
        enc_dec_repetition_penalty=args.enc_dec_repetition_penalty,
        enc_dec_no_repeat_ngram_size=args.enc_dec_no_repeat_ngram_size,
        enc_dec_max_generate=args.enc_dec_max_generate,
        spec_augment_freq_mask_param=args.spec_augment_freq_mask_param,
        spec_augment_n_time_masks=args.spec_augment_n_time_masks,
        spec_augment_n_freq_masks=args.spec_augment_n_freq_masks,
        optim_lr=args.optim_lr,
        shuffle=False,
        freeze_decoder=False,
    )


def normalized_wer(hypothesis: str, reference: str) -> float:
    return word_error_rate_detail([hypothesis], [reference])[0]


def file_offset(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def read_events_since(path: Path, offset: int) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with path.open() as f:
        f.seek(offset)
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def write_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def completed_units(path: Path) -> set[str]:
    if not path.exists():
        return set()
    units = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            units.add(json.loads(line)["unit_id"])
    return units


def iter_utterances(args: argparse.Namespace):
    seen = 0
    yielded = 0
    for rec in get_text_and_audio(args.split):
        _, _, remove_timings = proc_stm_and_timings(rec["text"])
        spec = processing_chain(rec["audio"])
        spec = zero_out_spectogram(spec=spec, remove_timings=remove_timings)
        utterances, _ = fetch_utterances(rec["text"], spec)
        for utt_idx, utterance in enumerate(utterances):
            duration = utterance["end"] - utterance["start"]
            if args.max_duration > 0.0 and duration > args.max_duration:
                continue
            if seen < args.start_index:
                seen += 1
                continue
            if args.max_utterances > 0 and yielded >= args.max_utterances:
                return
            unit_id = f"{Path(rec['audio']).stem}:{utt_idx:04d}:{utterance['start']:.2f}-{utterance['end']:.2f}"
            yield {
                "unit_id": unit_id,
                "recording_id": Path(rec["audio"]).stem,
                "utterance_index": utt_idx,
                "start": utterance["start"],
                "end": utterance["end"],
                "duration": duration,
                "gold": normalize(utterance["text"]).lower(),
                "spectogram": utterance["spectogram"],
            }
            seen += 1
            yielded += 1


def summarize(rows_path: Path, summary_json: Path, summary_csv: Path) -> None:
    rows = []
    with rows_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    scalar_fields = [
        "unit_id",
        "recording_id",
        "utterance_index",
        "duration",
        "word_count",
        "baseline_wer",
        "adapted_wer",
        "wer_delta",
        "improved",
        "teacher_selected_wer",
        "teacher_accepted_updates",
        "teacher_skipped_updates",
        "mean_vote_count",
        "mean_vote_fraction",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in scalar_fields})

    baseline_texts = [row["baseline_text"] for row in rows]
    adapted_texts = [row["adapted_text"] for row in rows]
    golds = [row["gold"] for row in rows]
    bins = {
        "teacher_exact": [],
        "teacher_wer_le_0p10": [],
        "teacher_wer_le_0p25": [],
        "teacher_wer_le_0p50": [],
        "teacher_wer_gt_0p50": [],
        "no_teacher_update": [],
    }
    for row in rows:
        teacher_wer = row.get("teacher_selected_wer")
        if teacher_wer is None:
            bins["no_teacher_update"].append(row)
        elif teacher_wer == 0.0:
            bins["teacher_exact"].append(row)
        elif teacher_wer <= 0.10:
            bins["teacher_wer_le_0p10"].append(row)
        elif teacher_wer <= 0.25:
            bins["teacher_wer_le_0p25"].append(row)
        elif teacher_wer <= 0.50:
            bins["teacher_wer_le_0p50"].append(row)
        else:
            bins["teacher_wer_gt_0p50"].append(row)

    def bin_summary(items: list[dict]) -> dict:
        if not items:
            return {"count": 0, "improved": 0, "mean_wer_delta": None}
        return {
            "count": len(items),
            "improved": sum(1 for row in items if row["improved"]),
            "mean_wer_delta": sum(row["wer_delta"] for row in items) / len(items),
        }

    summary = {
        "num_utterances": len(rows),
        "baseline_corpus_wer": word_error_rate_detail(baseline_texts, golds)[0] if rows else None,
        "adapted_corpus_wer": word_error_rate_detail(adapted_texts, golds)[0] if rows else None,
        "num_improved": sum(1 for row in rows if row["improved"]),
        "num_worsened": sum(1 for row in rows if row["wer_delta"] > 0),
        "num_unchanged": sum(1 for row in rows if row["wer_delta"] == 0),
        "teacher_bins": {name: bin_summary(items) for name, items in bins.items()},
    }
    if summary["baseline_corpus_wer"] is not None:
        summary["corpus_wer_delta"] = summary["adapted_corpus_wer"] - summary["baseline_corpus_wer"]
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "utterance_diagnostics.jsonl"
    teacher_events_path = output_dir / "teacher_events.jsonl"
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"

    model, tokenizer, config = load_enc_dec_model(args)
    baseline_args = make_eval_args(args, config, epochs=0)
    adapted_args = make_eval_args(args, config, epochs=args.epochs, diagnostics_path=str(teacher_events_path))

    done = completed_units(rows_path) if not args.no_resume else set()
    processed = 0
    started = time.time()
    for utterance in tqdm(iter_utterances(args), desc="utterances"):
        if utterance["unit_id"] in done:
            continue
        gold = utterance["gold"]
        spec = utterance["spectogram"]
        print(f"Processing {utterance['unit_id']} duration={utterance['duration']:.2f}s words={len(gold.split())}")

        baseline_text = normalize(lib.enc_dec_dynamic_eval(
            args=copy.copy(baseline_args),
            model=model,
            spec=spec,
            seq_len=args.seq_len,
            overlap=args.overlap,
            tokenizer=tokenizer,
            use_tqdm=False,
        )).lower()

        adapted_for_unit = copy.copy(adapted_args)
        adapted_for_unit.teacher_diagnostics_context = {
            "unit_id": utterance["unit_id"],
            "recording_id": utterance["recording_id"],
            "utterance_index": utterance["utterance_index"],
            "start": utterance["start"],
            "end": utterance["end"],
        }
        offset = file_offset(teacher_events_path)
        adapted_text = normalize(lib.enc_dec_dynamic_eval(
            args=adapted_for_unit,
            model=model,
            spec=spec,
            seq_len=args.seq_len,
            overlap=args.overlap,
            tokenizer=tokenizer,
            use_tqdm=False,
        )).lower()
        events = read_events_since(teacher_events_path, offset)
        accepted = [event for event in events if event.get("event") == "teacher_update_accepted"]
        skipped = [event for event in events if event.get("event") == "teacher_update_skipped"]
        teacher_texts = [event["teacher_text"] for event in accepted if event.get("teacher_text")]
        teacher_selected_text = " ".join(teacher_texts).strip() if teacher_texts else None
        teacher_selected_wer = normalized_wer(teacher_selected_text, gold) if teacher_selected_text else None
        vote_counts = [event.get("vote_count") for event in accepted if event.get("vote_count") is not None]
        vote_totals = [event.get("vote_total") for event in accepted if event.get("vote_total") is not None]

        baseline_wer = normalized_wer(baseline_text, gold)
        adapted_wer = normalized_wer(adapted_text, gold)
        row = {
            "unit_id": utterance["unit_id"],
            "recording_id": utterance["recording_id"],
            "utterance_index": utterance["utterance_index"],
            "start": utterance["start"],
            "end": utterance["end"],
            "duration": utterance["duration"],
            "gold": gold,
            "word_count": len(gold.split()),
            "baseline_text": baseline_text,
            "adapted_text": adapted_text,
            "baseline_wer": baseline_wer,
            "adapted_wer": adapted_wer,
            "wer_delta": adapted_wer - baseline_wer,
            "improved": adapted_wer < baseline_wer,
            "teacher_selected_text": teacher_selected_text,
            "teacher_selected_wer": teacher_selected_wer,
            "teacher_accepted_updates": len(accepted),
            "teacher_skipped_updates": len(skipped),
            "mean_vote_count": (sum(vote_counts) / len(vote_counts)) if vote_counts else None,
            "mean_vote_fraction": (
                sum(count / total for count, total in zip(vote_counts, vote_totals) if total)
                / len(vote_counts)
            ) if vote_counts else None,
            "skip_reasons": [event.get("reason") for event in skipped],
        }
        write_jsonl(rows_path, row)
        processed += 1
        if processed % 25 == 0:
            summarize(rows_path, summary_json, summary_csv)
            print(f"Intermediate summary written after {processed} new utterances")

    summarize(rows_path, summary_json, summary_csv)
    elapsed = time.time() - started
    print(f"Finished {processed} new utterances in {elapsed / 60:.1f} min")
    print(f"Wrote {rows_path}, {teacher_events_path}, {summary_json}, {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
