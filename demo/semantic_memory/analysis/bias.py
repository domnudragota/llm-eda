import numpy as np
from scipy import stats


def bias_evaluation(raw_activation, target_groups, original, female_primes, male_primes):
    normalized = original.normalizeDF(raw_activation, normalize_rows=True)

    results = {}
    for group, alternative in [('Female', 'greater'), ('Male', 'less')]:

        targets = [word for word in target_groups[group].dropna() if word in normalized.index]
        matrix = original.matricesGender(normalized, targets, female_primes, male_primes)

        differences = (matrix[female_primes].to_numpy() - matrix[male_primes].to_numpy()).ravel()
        if not differences.size or not np.any(differences):
            results[group] = None
            continue

        test = stats.wilcoxon(differences, alternative=alternative, method='approx')
        results[group] = {
            'paired_differences': differences,
            'p_value': test.pvalue,
            'effect_z_over_sqrt_n': test.zstatistic / np.sqrt(differences.size),
        }
    return results
