"""Aggregate within_loo pickles into a dict keyed by setting.

For each setting (filename minus trailing `_<repeat>.pkl`):
- `baseline` and `loo` are dataset-level WER dicts (the LOO stitched logits
  are already decoded into per-recording predictions and the WER is computed
  corpus-wide), so they are used as-is per repeat,
- both are averaged across repeats; `delta_wer = loo.wer - baseline.wer`.

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
        loo = avg_dicts([r["loo"] for r in repeats])
        out[setting] = {
            "baseline": baseline,
            "loo": loo,
            "delta_wer": loo["wer"] - baseline["wer"],
            "n_repeats": len(repeats),
        }
    return out


if __name__ == "__main__":
    results = aggregate(Path(__file__).parent)
    print(json.dumps(results, indent=2, default=float))
