"""Aggregate whole_concat_eval pickles into a dict keyed by setting.

For each setting (filename minus trailing `_<repeat>.pkl`):
- `baseline` and `adapted` are dataset-level WER dicts (already computed
  corpus-wide over every original recording with the adapted weights),
- both are averaged across repeats; `delta_wer = adapted.wer - baseline.wer`.

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
        baseline = avg_dicts([r["baseline"] for r in repeats])
        adapted = avg_dicts([r["adapted"] for r in repeats])
        out[setting] = {
            "baseline": baseline,
            "adapted": adapted,
            "delta_wer": adapted["wer"] - baseline["wer"],
            "n_repeats": len(repeats),
        }
    return out


if __name__ == "__main__":
    results = aggregate(Path(__file__).parent)
    print(json.dumps(results, indent=2, default=float))
