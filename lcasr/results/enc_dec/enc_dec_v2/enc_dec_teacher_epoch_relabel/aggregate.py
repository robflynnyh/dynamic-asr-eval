"""Aggregate encoder-decoder teacher epoch-relabel pickles.

This wrapper reuses the parser and output formatting from
`results/enc_dec/enc_dec_v2/enc_dec_dynamic_eval/aggregate.py`.
"""
import argparse
import importlib.util
import json
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "enc_dec_dynamic_eval" / "aggregate.py"
SPEC = importlib.util.spec_from_file_location("enc_dec_dynamic_aggregate", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load aggregate helper from {SOURCE}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

aggregate = MODULE.aggregate
print_table = MODULE.print_table
write_csv = MODULE.write_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the compact table")
    parser.add_argument("--csv", type=Path, default="summary.csv", help="Optional path to write CSV summary")
    args = parser.parse_args()

    results = aggregate(Path(__file__).parent)
    if args.csv is not None:
        write_csv(results, args.csv)
    if args.json:
        print(json.dumps(results, indent=2, default=float))
    else:
        print_table(results)
