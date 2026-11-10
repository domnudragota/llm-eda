from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from semantic_memory.analysis.bias import bias_evaluation
from semantic_memory.integrations.lwow import DEFAULT_REPO, load_original_functions, official_gender_matrix_path

FEMALE_PRIMES = ['woman', 'female', 'mother', 'girl', 'feminine']
MALE_PRIMES = ['man', 'male', 'father', 'boy', 'masculine']


def main() -> None:
    parser = argparse.ArgumentParser(
        description='A separate educational adaptation of the original 2025 LWOW bias-evaluation steps.'
    )
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--model', choices=['Mistral', 'Llama3', 'Haiku', 'Humans'], default='Mistral')
    parser.add_argument('--max-targets', type=int, default=8)

    args = parser.parse_args()
    if args.max_targets < 0:
        parser.error('--max-targets must be 0 or a positive integer')

    path = official_gender_matrix_path(args.repo, args.model)
    if not path.exists():
        parser.error(f'Original published activation matrix missing: {path}; run scripts/setup_lwow.ps1')

    original = load_original_functions(args.repo)
    raw = pd.read_csv(path)
    required = {'node', *FEMALE_PRIMES, *MALE_PRIMES}
    missing = required - set(raw.columns)
    if missing:
        parser.error(f'Expected prime columns missing in ORIGINAL matrix: {sorted(missing)}')

    targets = pd.read_csv(args.repo / 'reproducibility/data/LDT_analyses/gender_targets.csv')
    nodes = set(raw['node'])
    selected_targets = {}
    for label in ['Female', 'Male']:
        available = [str(word) for word in targets[label].dropna() if str(word) in nodes]
        selected_targets[label] = available[:args.max_targets] if args.max_targets else available
    target_groups = pd.DataFrame({label: pd.Series(words, dtype=str) for label, words in selected_targets.items()})
    results = bias_evaluation(raw, target_groups, original, FEMALE_PRIMES, MALE_PRIMES)

    print(f'Selected original model: {args.model}; max targets/group: {args.max_targets or "all"}')

    for label, alternative in [('Female', 'greater'), ('Male', 'less')]:
        chosen = selected_targets[label]
        if not chosen:
            print(f'\n{label}: no published target words found; skipping')
            continue

        result = results[label]
        if result is None:
            print(f'\n{label}: all paired differences are zero; signed-rank test is undefined')
            continue
        paired_differences = result['paired_differences']

        if not np.isfinite(paired_differences).all():
            raise ValueError('Non-finite original activation values encountered')
        effect = result['effect_z_over_sqrt_n']

        print(f'\n2025 {label}-related target group:')
        print(f'  target words displayed in test: {len(chosen)}')
        print(f'  prime-paired differences: {len(paired_differences)}')
        print(f'  F-prime minus M-prime difference: mean={np.mean(paired_differences):.6g}')
        print(f'  Wilcoxon alternative={alternative}; p={result["p_value"]:.6g}')
        print(f'  approximate Z/sqrt(N) effect (2025 sign convention) = {effect:.6g}')


if __name__ == '__main__':
    main()
