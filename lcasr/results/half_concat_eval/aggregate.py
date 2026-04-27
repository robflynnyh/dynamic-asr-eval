"""Aggregate half_concat_eval pickles into a dict keyed by setting.

Each pickle has `folds` (one per held-out half) where every fold contains
its own corpus-level `baseline`/`adapted` WER dicts. For each setting
(filename minus trailing `_<repeat>.pkl`):
- folds within a repeat are averaged into a single `baseline`/`adapted` pair,
- those per-repeat dicts are then averaged across repeats,
- `delta_wer = adapted.wer - baseline.wer`.

Run: `python aggregate.py` from this directory.
"""
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

REPEAT_RE = re.compile(r"_(\d+)\.pkl$")


def avg_dicts(ds):
    keys = ds[0].keys()
    return {k: sum(d[k] for d in ds) / len(ds) for k in keys}


def aggregate(directory: Path) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(directory.glob("*.pkl")):
        m = REPEAT_RE.search(p.name)
        if not m:
            continue
        setting = p.name[: m.start()]
        with open(p, "rb") as f:
            groups[setting].append(pickle.load(f))

    out: dict[str, dict] = {}
    for setting, repeats in groups.items():
        per_repeat = []
        for rep in repeats:
            folds = rep["folds"]
            per_repeat.append({
                "baseline": avg_dicts([f["baseline"] for f in folds]),
                "adapted": avg_dicts([f["adapted"] for f in folds]),
            })
        baseline = avg_dicts([r["baseline"] for r in per_repeat])
        adapted = avg_dicts([r["adapted"] for r in per_repeat])
        out[setting] = {
            "baseline": baseline,
            "adapted": adapted,
            "delta_wer": adapted["wer"] - baseline["wer"],
            "n_repeats": len(per_repeat),
            "n_folds_per_repeat": len(repeats[0]["folds"]),
        }
    return out


if __name__ == "__main__":
    results = aggregate(Path(__file__).parent)
    print(json.dumps(results, indent=2, default=float))
