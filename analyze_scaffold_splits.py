"""
Statistical analysis of scaffold split experiments.

This script collects AUC scores from the 5 scaffold splits and performs
statistical significance testing to demonstrate that improvements are
statistically meaningful.

Usage:
    python analyze_scaffold_splits.py --model moltitox --baseline graph
"""
import os
import sys
import argparse
import numpy as np
from utils.statistical_testing import (
    analyze_scaffold_splits,
    print_statistical_results,
    save_statistical_results
)


def parse_auc_from_results(result_txt_path: str, model_type: str) -> float:
    """
    Parse Mean AUC from result_test.txt file for a specific model.

    Args:
        result_txt_path: Path to result_test.txt file
        model_type: Type of model (e.g., 'moltitox', 'graph', 'smi_img')

    Returns:
        Mean AUC score as float, or None if not found
    """
    try:
        with open(result_txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Look for the model's test section
        model_markers = {
            'graph': 'experiments/graph/test.py',
            'smiles': 'experiments/smiles/test.py',
            'image': 'experiments/image/test.py',
            'spectrum': 'experiments/spectrum/test.py',
            'gph_smi': 'experiments/multimodal/gph_smi/test.py',
            'gph_img': 'experiments/multimodal/gph_img/test.py',
            'gph_spec': 'experiments/multimodal/gph_spec/test.py',
            'gph_smi_img': 'experiments/multimodal/gph_smi_img/test.py',
            'smi_img': 'experiments/multimodal/smi_img/test.py',
            'smi_spec': 'experiments/multimodal/smi_spec/test.py',
            'spec_img': 'experiments/multimodal/spec_img/test.py',
            'moltitox': 'experiments/multimodal/moltitox/test.py'
        }

        marker = model_markers.get(model_type)
        if not marker:
            print(f"Warning: Unknown model type '{model_type}'")
            return None

        # Find the section for this model
        marker_idx = content.find(marker)
        if marker_idx == -1:
            print(f"Warning: Could not find marker '{marker}' in {result_txt_path}")
            return None

        # Search for "Mean AUC" after the marker
        search_start = marker_idx
        search_text = content[search_start:search_start + 5000]  # Look ahead 5000 chars

        # Find "Mean AUC        : X.XXXX"
        for line in search_text.split('\n'):
            if 'Mean AUC' in line and ':' in line:
                try:
                    # Extract the number after the colon
                    auc_str = line.split(':')[-1].strip()
                    auc_val = float(auc_str)
                    return auc_val
                except (ValueError, IndexError):
                    continue

        print(f"Warning: Could not find Mean AUC for {model_type} in {result_txt_path}")
        return None

    except Exception as e:
        print(f"Error reading {result_txt_path}: {e}")
        return None


def collect_scaffold_split_results(
    scaffold_dirs: list,
    model_type: str
) -> list:
    """
    Collect AUC scores from multiple scaffold splits.

    Args:
        scaffold_dirs: List of paths to scaffold split directories
        model_type: Type of model to extract results for

    Returns:
        List of AUC scores
    """
    scores = []

    for split_dir in scaffold_dirs:
        # Try different result file names
        result_files = ['result.txt']
        result_path = None

        for result_file in result_files:
            candidate_path = os.path.join(split_dir, result_file)
            if os.path.exists(candidate_path):
                result_path = candidate_path
                break

        if result_path is None:
            print(f"Warning: No result file found in {split_dir}")
            continue

        auc = parse_auc_from_results(result_path, model_type)
        if auc is not None:
            scores.append(auc)

    return scores


def main():
    parser = argparse.ArgumentParser(description='Analyze scaffold split results')
    parser.add_argument('--model', type=str, required=True,
                        help='Multimodal model type (e.g., moltitox, smi_img)')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Baseline model type (e.g., graph, smiles, image)')
    parser.add_argument('--splits-dir', type=str,
                        default='/home/user3/toxic_research/JunWoo/실험기록들_table1', #유동적으로 바꾸기
                        help='Directory containing scaffold splits')
    parser.add_argument('--output-dir', type=str,
                        default='./checkpoints/statistical_analysis',
                        help='Directory to save results')

    args = parser.parse_args()

    # Define scaffold split directories
    split_names = ['10회차', '18회차', '21회차', '24회차', '25회차']
    scaffold_dirs = [os.path.join(args.splits_dir, name) for name in split_names]

    print(f"\nCollecting results for {len(scaffold_dirs)} scaffold splits...")
    print(f"Model: {args.model}")
    print(f"Baseline: {args.baseline}")

    # Collect scores
    model_scores = collect_scaffold_split_results(scaffold_dirs, args.model)
    baseline_scores = collect_scaffold_split_results(scaffold_dirs, args.baseline)

    if len(model_scores) == 0:
        print(f"\nError: No scores found for model '{args.model}'")
        return

    if len(baseline_scores) == 0:
        print(f"\nError: No scores found for baseline '{args.baseline}'")
        return

    if len(model_scores) != len(baseline_scores):
        print(f"\nWarning: Different number of scores found:")
        print(f"  Model: {len(model_scores)} splits")
        print(f"  Baseline: {len(baseline_scores)} splits")
        # Use minimum length
        min_len = min(len(model_scores), len(baseline_scores))
        model_scores = model_scores[:min_len]
        baseline_scores = baseline_scores[:min_len]

    print(f"\nCollected {len(model_scores)} scores for each method")
    print(f"\n{args.model} scores: {model_scores}")
    print(f"{args.baseline} scores: {baseline_scores}")

    # Perform statistical analysis
    print("\nPerforming statistical analysis...")
    results = analyze_scaffold_splits(
        multimodal_scores=model_scores,
        baseline_scores=baseline_scores,
        model_name=args.model,
        baseline_name=args.baseline,
        n_bootstrap=10000
    )

    # Print results
    print_statistical_results(results)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{args.model}_vs_{args.baseline}_statistical_analysis.json'
    )
    save_statistical_results(results, output_file)

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    p_val = results['paired_t_test']['p_value']
    ci_excludes_zero = results['bootstrap_difference']['CI_excludes_zero']
    mean_diff = results['summary_statistics']['mean_difference']

    if p_val < 0.05 and ci_excludes_zero:
        print(f"\n✓ The improvement of {args.model} over {args.baseline} is")
        print(f"  STATISTICALLY SIGNIFICANT (p={p_val:.4f}, α=0.05)")
        print(f"\n  Mean difference: {mean_diff:.4f}")
        print(f"  95% CI: [{results['bootstrap_difference']['95%_CI_lower']:.4f}, "
              f"{results['bootstrap_difference']['95%_CI_upper']:.4f}]")
        print(f"\n  The confidence interval excludes zero, confirming that")
        print(f"  the improvement is meaningful across scaffold splits.")
    elif p_val < 0.05:
        print(f"\n~ The improvement shows statistical significance in the t-test")
        print(f"  (p={p_val:.4f}), but the 95% CI includes zero.")
        print(f"  This suggests the effect size may be small or variable.")
    else:
        print(f"\n✗ The improvement is NOT statistically significant (p={p_val:.4f})")
        print(f"\n  While {args.model} shows a mean improvement of {mean_diff:.4f},")
        print(f"  this difference could be due to random variation.")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
