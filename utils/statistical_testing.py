"""
Statistical significance testing for scaffold split experiments.
Implements paired t-tests and bootstrap confidence intervals.
"""
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
import json
import os


def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Perform paired t-test between two sets of scores.

    Args:
        scores_a: Array of scores for method A (e.g., multimodal model)
        scores_b: Array of scores for method B (e.g., baseline)
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        Tuple of (t-statistic, p-value)
    """
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b, alternative=alternative)
    return float(t_stat), float(p_val)




def bootstrap_confidence_interval(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    statistic=np.mean
) -> Tuple[float, float, float]:
    """
    Compute bootstrapped confidence interval for a statistic.

    Args:
        scores: Array of scores
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        statistic: Function to compute statistic (default: mean)

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    n = len(scores)
    bootstrap_stats = []

    # Generate bootstrap samples
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    point_estimate = statistic(scores)
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)

    return float(point_estimate), float(lower_bound), float(upper_bound)


def bootstrap_difference_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrapped confidence interval for the difference between two methods.

    Args:
        scores_a: Array of scores for method A
        scores_b: Array of scores for method B
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level

    Returns:
        Tuple of (mean_difference, lower_bound, upper_bound)
    """
    n = len(scores_a)
    assert len(scores_b) == n, "Both arrays must have same length"

    bootstrap_diffs = []

    # Generate bootstrap samples
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        sample_a = scores_a[indices]
        sample_b = scores_b[indices]
        bootstrap_diffs.append(np.mean(sample_a) - np.mean(sample_b))

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    mean_diff = np.mean(scores_a) - np.mean(scores_b)
    lower_bound = np.percentile(bootstrap_diffs, lower_percentile)
    upper_bound = np.percentile(bootstrap_diffs, upper_percentile)

    return float(mean_diff), float(lower_bound), float(upper_bound)


def analyze_scaffold_splits(
    multimodal_scores: List[float],
    baseline_scores: List[float],
    model_name: str = "multimodal",
    baseline_name: str = "baseline",
    n_bootstrap: int = 10000
) -> Dict:
    """
    Comprehensive statistical analysis of scaffold split results.

    Args:
        multimodal_scores: List of scores for multimodal model across splits
        baseline_scores: List of scores for baseline across splits
        model_name: Name of the multimodal model
        baseline_name: Name of the baseline model
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary containing all statistical test results
    """
    scores_a = np.array(multimodal_scores)
    scores_b = np.array(baseline_scores)

    results = {
        'model_name': model_name,
        'baseline_name': baseline_name,
        'n_splits': len(multimodal_scores),
        'scores': {
            model_name: multimodal_scores,
            baseline_name: baseline_scores
        },
        'summary_statistics': {},
        'paired_t_test': {},
        'confidence_intervals': {},
        'bootstrap_difference': {}
    }

    # Summary statistics
    results['summary_statistics'] = {
        model_name: {
            'mean': float(np.mean(scores_a)),
            'std': float(np.std(scores_a, ddof=1)),
            'min': float(np.min(scores_a)),
            'max': float(np.max(scores_a))
        },
        baseline_name: {
            'mean': float(np.mean(scores_b)),
            'std': float(np.std(scores_b, ddof=1)),
            'min': float(np.min(scores_b)),
            'max': float(np.max(scores_b))
        },
        'mean_difference': float(np.mean(scores_a) - np.mean(scores_b))
    }

    # Paired t-test
    t_stat, p_val_t = paired_t_test(scores_a, scores_b, alternative='greater')
    results['paired_t_test'] = {
        't_statistic': t_stat,
        'p_value': p_val_t,
        'significant_at_0.05': p_val_t < 0.05,
        'significant_at_0.01': p_val_t < 0.01
    }

    # Bootstrap confidence intervals for each method
    mean_a, ci_a_lower, ci_a_upper = bootstrap_confidence_interval(
        scores_a, n_bootstrap=n_bootstrap
    )
    mean_b, ci_b_lower, ci_b_upper = bootstrap_confidence_interval(
        scores_b, n_bootstrap=n_bootstrap
    )

    results['confidence_intervals'] = {
        model_name: {
            'mean': mean_a,
            '95%_CI_lower': ci_a_lower,
            '95%_CI_upper': ci_a_upper
        },
        baseline_name: {
            'mean': mean_b,
            '95%_CI_lower': ci_b_lower,
            '95%_CI_upper': ci_b_upper
        }
    }

    # Bootstrap confidence interval for the difference
    diff_mean, diff_ci_lower, diff_ci_upper = bootstrap_difference_ci(
        scores_a, scores_b, n_bootstrap=n_bootstrap
    )

    results['bootstrap_difference'] = {
        'mean_difference': diff_mean,
        '95%_CI_lower': diff_ci_lower,
        '95%_CI_upper': diff_ci_upper,
        'CI_excludes_zero': diff_ci_lower > 0 or diff_ci_upper < 0
    }

    return results


def print_statistical_results(results: Dict):
    """
    Print formatted statistical test results.

    Args:
        results: Dictionary from analyze_scaffold_splits()
    """
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE ANALYSIS - SCAFFOLD SPLITS")
    print("=" * 70)

    print(f"\nModel: {results['model_name']}")
    print(f"Baseline: {results['baseline_name']}")
    print(f"Number of splits: {results['n_splits']}")

    # Summary statistics
    print("\n" + "-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)

    for method in [results['model_name'], results['baseline_name']]:
        stats_dict = results['summary_statistics'][method]
        print(f"\n{method}:")
        print(f"  Mean ± Std: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
        print(f"  Range: [{stats_dict['min']:.4f}, {stats_dict['max']:.4f}]")

    mean_diff = results['summary_statistics']['mean_difference']
    print(f"\nMean Difference: {mean_diff:.4f}")

    # Paired t-test
    print("\n" + "-" * 70)
    print("PAIRED T-TEST (one-sided: model > baseline)")
    print("-" * 70)
    t_test = results['paired_t_test']
    print(f"t-statistic: {t_test['t_statistic']:.4f}")
    print(f"p-value: {t_test['p_value']:.4f}")
    print(f"Significant at α=0.05: {t_test['significant_at_0.05']}")
    print(f"Significant at α=0.01: {t_test['significant_at_0.01']}")

    # Confidence intervals
    print("\n" + "-" * 70)
    print("95% BOOTSTRAP CONFIDENCE INTERVALS")
    print("-" * 70)

    for method in [results['model_name'], results['baseline_name']]:
        ci = results['confidence_intervals'][method]
        print(f"\n{method}:")
        print(f"  Mean: {ci['mean']:.4f}")
        print(f"  95% CI: [{ci['95%_CI_lower']:.4f}, {ci['95%_CI_upper']:.4f}]")

    # Difference CI
    print("\nDifference (Model - Baseline):")
    diff = results['bootstrap_difference']
    print(f"  Mean: {diff['mean_difference']:.4f}")
    print(f"  95% CI: [{diff['95%_CI_lower']:.4f}, {diff['95%_CI_upper']:.4f}]")
    print(f"  CI excludes zero: {diff['CI_excludes_zero']}")

    print("\n" + "=" * 70)


def save_statistical_results(results: Dict, save_path: str):
    """
    Save statistical test results to JSON file.

    Args:
        results: Dictionary from analyze_scaffold_splits()
        save_path: Path to save JSON file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nStatistical results saved to: {save_path}")
