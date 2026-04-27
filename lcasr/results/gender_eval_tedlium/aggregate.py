"""Aggregate gender_eval_tedlium pickles into a dict keyed by setting.

For each setting (filename minus trailing `_<repeat>.pkl`):
- per-repeat list fields (`male_to_male`, `female_to_female`,
  `male_to_female`, `female_to_male`) are averaged across adapt-speakers
  first, producing a single WER dict per repeat,
- baseline fields (`male_baseline`, `female_baseline`) are taken as-is,
- everything is then averaged across repeats.

Run: `python aggregate.py` from this directory.
"""
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

REPEAT_RE = re.compile(r"_(\d+)\.pkl$")
LIST_FIELDS = ("male_to_male", "female_to_female", "male_to_female", "female_to_male")
BASELINE_FIELDS = ("male_baseline", "female_baseline")


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
            row = {f: rep[f] for f in BASELINE_FIELDS if f in rep}
            for f in LIST_FIELDS:
                if f in rep and len(rep[f]) > 0:
                    row[f] = avg_dicts(rep[f])
            per_repeat.append(row)

        averaged = {}
        for f in BASELINE_FIELDS + LIST_FIELDS:
            present = [r[f] for r in per_repeat if f in r]
            if present:
                averaged[f] = avg_dicts(present)
        averaged["n_repeats"] = len(per_repeat)
        out[setting] = averaged
    return out


if __name__ == "__main__":
    results = aggregate(Path(__file__).parent)
    print(json.dumps(results, indent=2, default=float))
