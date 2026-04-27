#!/usr/bin/env python3
import argparse
import math
import os
import pickle
import statistics as st
from typing import Dict, List


METRIC_KEYS = ["wer", "ins_rate", "del_rate", "sub_rate"]


def load_results(results_dir: str, prefix: str) -> List[Dict]:
    files = sorted(
        f for f in os.listdir(results_dir)
        if f.startswith(prefix) and f.endswith('.pkl')
    )
    if not files:
        raise FileNotFoundError(f"No pickle files found in {results_dir!r} with prefix {prefix!r}")

    rows = []
    for fn in files:
        path = os.path.join(results_dir, fn)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        rows.append(data)
    return rows


def mean_metric(items: List[Dict], key: str) -> float:
    return st.mean(item[key] for item in items)


def pstdev_or_zero(values: List[float]) -> float:
    return st.pstdev(values) if len(values) > 1 else 0.0


def summarize(repeats: List[Dict]) -> Dict:
    a_baseline_values = [r['a_baseline']['wer'] for r in repeats]
    b_baseline_values = [r['b_baseline']['wer'] for r in repeats]
    a_to_b_repeat_means = [st.mean(x['wer'] for x in r['a_to_b']) for r in repeats]
    a_to_a_repeat_means = [st.mean(x['wer'] for x in r['a_to_a_loo']) for r in repeats]

    all_a_to_b = [item for r in repeats for item in r['a_to_b']]
    all_a_to_a = [item for r in repeats for item in r['a_to_a_loo']]

    summary = {
        'dataset_a': repeats[0]['dataset_a'],
        'dataset_b': repeats[0]['dataset_b'],
        'num_repeats': len(repeats),
        'num_cross_cases_per_repeat': len(repeats[0]['a_to_b']),
        'a_baseline_wer_mean': st.mean(a_baseline_values),
        'a_baseline_wer_std': pstdev_or_zero(a_baseline_values),
        'b_baseline_wer_mean': st.mean(b_baseline_values),
        'b_baseline_wer_std': pstdev_or_zero(b_baseline_values),
        'a_to_b_wer_mean': st.mean(a_to_b_repeat_means),
        'a_to_b_wer_std_across_repeats': pstdev_or_zero(a_to_b_repeat_means),
        'a_to_a_wer_mean': st.mean(a_to_a_repeat_means),
        'a_to_a_wer_std_across_repeats': pstdev_or_zero(a_to_a_repeat_means),
        'a_to_b_metrics_mean': {k: mean_metric(all_a_to_b, k) for k in METRIC_KEYS},
        'a_to_a_metrics_mean': {k: mean_metric(all_a_to_a, k) for k in METRIC_KEYS},
    }
    return summary


def fmt(x: float) -> str:
    return f"{x * 100:.2f}"


def build_latex_table(summary: Dict) -> str:
    dataset_a = summary['dataset_a']
    dataset_b = summary['dataset_b']
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Cross-dataset evaluation summary for epoch 1 ({dataset_a} $\rightarrow$ {dataset_b}), averaged over {summary['num_repeats']} repeats and {summary['num_cross_cases_per_repeat']} adaptation cases per repeat.}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Setting & WER & Ins. & Del. & Sub. \\",
        r"\midrule",
        rf"{dataset_a} baseline & {fmt(summary['a_baseline_wer_mean'])} & -- & -- & -- \\",
        rf"{dataset_b} baseline & {fmt(summary['b_baseline_wer_mean'])} & -- & -- & -- \\",
        rf"{dataset_a}$\rightarrow${dataset_b} & {fmt(summary['a_to_b_metrics_mean']['wer'])} & {fmt(summary['a_to_b_metrics_mean']['ins_rate'])} & {fmt(summary['a_to_b_metrics_mean']['del_rate'])} & {fmt(summary['a_to_b_metrics_mean']['sub_rate'])} \\",
        rf"{dataset_a}$\rightarrow${dataset_a} (leave-one-out) & {fmt(summary['a_to_a_metrics_mean']['wer'])} & {fmt(summary['a_to_a_metrics_mean']['ins_rate'])} & {fmt(summary['a_to_a_metrics_mean']['del_rate'])} & {fmt(summary['a_to_a_metrics_mean']['sub_rate'])} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results-dir',
        default='.',
        help='Directory containing the epoch-1 pickle files',
    )
    parser.add_argument(
        '--prefix',
        default='earnings22_tedlium-epoch-1-test_',
        help='Filename prefix for the pickle files to summarize',
    )
    parser.add_argument(
        '--output',
        default='epoch1_summary_table.tex',
        help='Output .tex file path',
    )
    args = parser.parse_args()

    repeats = load_results(args.results_dir, args.prefix)
    summary = summarize(repeats)
    latex = build_latex_table(summary)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(latex)
        f.write('\n')

    cross_pct_change = ((summary['a_to_b_wer_mean'] - summary['b_baseline_wer_mean']) / summary['b_baseline_wer_mean']) * 100.0
    loo_pct_change = ((summary['a_to_a_wer_mean'] - summary['a_baseline_wer_mean']) / summary['a_baseline_wer_mean']) * 100.0

    print(f"Loaded {summary['num_repeats']} repeats from {args.results_dir}")
    print(f"Dataset A baseline WER: {summary['a_baseline_wer_mean']:.6f}")
    print(f"Dataset B baseline WER: {summary['b_baseline_wer_mean']:.6f}")
    print(f"A->B averaged WER: {summary['a_to_b_wer_mean']:.6f}")
    print(f"A->A leave-one-out averaged WER: {summary['a_to_a_wer_mean']:.6f}")
    print(f"Relative WER change for A->B vs B baseline: {cross_pct_change:.4f}%")
    print(f"Relative WER change for A->A leave-one-out vs A baseline: {loo_pct_change:.4f}%")
    print(f"Wrote LaTeX table to {args.output}")


if __name__ == '__main__':
    main()
