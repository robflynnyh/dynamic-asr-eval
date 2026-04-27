#!/usr/bin/env python3
"""Generate a LaTeX table for the epoch-1 cross-dataset / within-dataset result."""

import argparse
from pathlib import Path

from aggregate import aggregate


def setting_key(epoch: int, lr_tag: str) -> str:
    return f"earnings22_tedlium-epoch-{epoch}{('-' + lr_tag) if lr_tag else ''}-test"


def fmt_wer(x: float) -> str:
    return f"{x * 100:.1f}"


def build_table(name_a: str, name_b: str, row: dict, epoch: int, lr_tag: str) -> str:
    a_baseline = row["a_baseline"]["wer"]
    b_baseline = row["b_baseline"]["wer"]
    a_adapted = row["a_to_a_loo"]["wer"]
    b_adapted = row["a_to_b"]["wer"]

    lines = [
        r"\paragraph{Cross-Dataset and Within-Dataset}",
        r"\begin{table}[h!]",
        r"    \centering",
        r"    \begin{tabular}{lcc}",
        r"        \toprule",
        r"        \multirow{2}{*}{Adapt on} & \multicolumn{2}{c}{Test on:} \\",
        r"        \cmidrule(lr){2-3}",
        rf"         & {name_a} & {name_b} \\",
        r"        \midrule",
        rf"        {name_a} & {fmt_wer(a_adapted)} & {fmt_wer(b_adapted)} \\",
        r"        \midrule",
        rf"        Unadapted & {fmt_wer(a_baseline)} & {fmt_wer(b_baseline)} \\",
        r"        \bottomrule",
        r"    \end{tabular}",
        rf"    \caption{{WERs after adapting on {name_a} for epoch {epoch} ({lr_tag or 'default'}), reported as percentages.}}",
        rf"    \label{{tab:crossdataset:epoch{epoch}{('-' + lr_tag) if lr_tag else ''}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1, help="Epoch to summarize. Default: 1.")
    parser.add_argument("--lr-tag", type=str, default="lr9e6",
                        help='Filename tag, e.g. "lr9e6" or "ao0". Use "" for the default-LR runs.')
    parser.add_argument("--name-a", type=str, default="earnings22")
    parser.add_argument("--name-b", type=str, default="tedlium")
    parser.add_argument("--output", type=str, default="epoch1_crossdataset_table.tex")
    args = parser.parse_args()

    results = aggregate(Path(__file__).parent)
    key = setting_key(args.epoch, args.lr_tag)
    if key not in results:
        raise SystemExit(f"Setting {key!r} not found. Available: {sorted(results)}")

    latex = build_table(args.name_a, args.name_b, results[key], args.epoch, args.lr_tag)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path
    out_path.write_text(latex + "\n", encoding="utf-8")
    print(latex)
    print(f"\nWrote LaTeX table to {out_path}")


if __name__ == "__main__":
    main()
