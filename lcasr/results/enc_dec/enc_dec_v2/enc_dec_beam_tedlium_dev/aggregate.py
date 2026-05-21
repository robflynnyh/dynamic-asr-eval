"""Aggregate TEDLIUM dev encoder-decoder beam-search sweep pickles.

For each setting (filename minus trailing `_<repeat>.pkl`), this reports the
mean WER and error components across repeats. It also exposes the decoder
arguments saved in `args_dict` and a simple output/reference word-length ratio.

Run: `python3.10 aggregate.py` from this directory.
"""
import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPEAT_RE = re.compile(r"_(\d+)\.pkl$")
SETTING_RE = re.compile(
    r"^(?P<dataset>.+)-(?P<split>dev|test)-(?P<run>.+)-seq(?P<seq>\d+)-overlap(?P<overlap>\d+)$"
)


DECODER_ARG_KEYS = (
    "decoding_mode",
    "enc_dec_beam_width",
    "enc_dec_length_penalty",
    "enc_dec_eos_bias",
    "enc_dec_repetition_penalty",
    "enc_dec_no_repeat_ngram_size",
    "enc_dec_max_generate",
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _word_count(texts: Any) -> int:
    if isinstance(texts, str):
        return len(texts.split())
    if isinstance(texts, (list, tuple)):
        return sum(len(str(text).split()) for text in texts)
    return 0


def _max_repeated_trigram(texts: Any) -> int:
    """Return the largest repeated word-trigram count across decoded texts."""
    if isinstance(texts, str):
        iterable = [texts]
    elif isinstance(texts, (list, tuple)):
        iterable = texts
    else:
        return 0

    max_count = 0
    for text in iterable:
        words = str(text).split()
        if len(words) < 3:
            continue
        counts = Counter(tuple(words[i : i + 3]) for i in range(len(words) - 2))
        max_count = max(max_count, max(counts.values(), default=0))
    return max_count


def parse_setting(setting: str) -> dict[str, Any]:
    match = SETTING_RE.match(setting)
    if match is None:
        return {"setting": setting, "run": setting}
    out = match.groupdict()
    out["setting"] = setting
    out["seq"] = int(out["seq"])
    out["overlap"] = int(out["overlap"])
    return out


def load_groups(directory: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(directory.glob("*.pkl")):
        match = REPEAT_RE.search(path.name)
        if match is None:
            continue
        setting = path.name[: match.start()]
        with open(path, "rb") as f:
            groups[setting].append(pickle.load(f))
    return groups


def _decoder_args(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    args_dict = repeats[0].get("args_dict") or {}
    return {key: args_dict.get(key) for key in DECODER_ARG_KEYS}


def aggregate(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for setting, repeats in sorted(load_groups(directory).items()):
        row = parse_setting(setting)
        row.update(_decoder_args(repeats))

        wers = [float(rep["wer"]) for rep in repeats]
        ref_words = [_word_count(rep.get("gold")) for rep in repeats]
        output_words = [_word_count(rep.get("model_output")) for rep in repeats]
        ratios = [
            output / ref
            for output, ref in zip(output_words, ref_words)
            if ref > 0
        ]

        row.update({
            "wer": _mean(wers),
            "wer_std": _std(wers),
            "n_repeats": len(repeats),
            "words": _mean([float(rep["words"]) for rep in repeats]),
            "ins_rate": _mean([float(rep["ins_rate"]) for rep in repeats]),
            "del_rate": _mean([float(rep["del_rate"]) for rep in repeats]),
            "sub_rate": _mean([float(rep["sub_rate"]) for rep in repeats]),
            "output_ref_ratio": _mean(ratios) if ratios else None,
            "max_repeated_trigram": max(_max_repeated_trigram(rep.get("model_output")) for rep in repeats),
            "repeat_ids": [rep.get("repeat") for rep in repeats],
        })
        rows.append(row)

    baselines = {
        (row.get("dataset"), row.get("split")): row["wer"]
        for row in rows
        if row.get("decoding_mode") == "default" or row.get("run") == "default"
    }
    for row in rows:
        baseline = baselines.get((row.get("dataset"), row.get("split")))
        row["delta_vs_default"] = None if baseline is None else row["wer"] - baseline

    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No result pickles found.", file=sys.stderr)
        return

    header = (
        "run", "mode", "beam", "lp", "eos", "rep", "ngram", "max_gen",
        "wer", "delta", "ins", "del", "sub", "len_ratio", "max_tri", "n",
    )
    print("\t".join(header))
    for row in sorted(rows, key=lambda item: item["wer"]):
        print("\t".join([
            str(row.get("run", "")),
            str(row.get("decoding_mode") or ""),
            "" if row.get("enc_dec_beam_width") is None else str(row["enc_dec_beam_width"]),
            "" if row.get("enc_dec_length_penalty") is None else str(row["enc_dec_length_penalty"]),
            "" if row.get("enc_dec_eos_bias") is None else str(row["enc_dec_eos_bias"]),
            "" if row.get("enc_dec_repetition_penalty") is None else str(row["enc_dec_repetition_penalty"]),
            "" if row.get("enc_dec_no_repeat_ngram_size") is None else str(row["enc_dec_no_repeat_ngram_size"]),
            "" if row.get("enc_dec_max_generate") is None else str(row["enc_dec_max_generate"]),
            f"{row['wer']:.5f}",
            "" if row["delta_vs_default"] is None else f"{row['delta_vs_default']:.5f}",
            f"{row['ins_rate']:.5f}",
            f"{row['del_rate']:.5f}",
            f"{row['sub_rate']:.5f}",
            "" if row["output_ref_ratio"] is None else f"{row['output_ref_ratio']:.3f}",
            str(row["max_repeated_trigram"]),
            str(row["n_repeats"]),
        ]))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
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
    parser.add_argument("--csv", type=Path, default="summary.csv", help="Path to write CSV summary")
    args = parser.parse_args()

    results = aggregate(Path(__file__).parent)
    if args.csv is not None:
        write_csv(results, args.csv)
    if args.json:
        print(json.dumps(results, indent=2, default=float))
    else:
        print_table(results)
