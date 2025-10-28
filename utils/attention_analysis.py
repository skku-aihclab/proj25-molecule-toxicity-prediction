"""
Utility functions for attention weight analysis and visualization.
Used to assess biological relevance of each modality in multimodal models.
"""
import torch
import numpy as np
import json
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def compute_modality_contributions(
    attn_weights: torch.Tensor,
    modality_names: List[str]
) -> Dict[str, float]:
    """
    Compute average contribution of each modality from attention weights.

    Args:
        attn_weights: Attention weights tensor of shape (batch_size, num_modalities, num_modalities)
        modality_names: List of modality names (e.g., ['Graph', 'SMILES', 'Image', 'Spectrum'])

    Returns:
        Dictionary mapping modality name to its average attention weight
    """
    # Average over batch dimension and source dimension (queries)
    # This gives us how much each modality (key) is attended to on average
    avg_attn = attn_weights.mean(dim=0).mean(dim=0)  # Shape: (num_modalities,)

    contributions = {}
    for i, name in enumerate(modality_names):
        contributions[name] = float(avg_attn[i].item())

    return contributions


def compute_sample_wise_contributions(
    attn_weights_list: List[torch.Tensor],
    modality_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sample-wise modality contributions and their statistics.

    Args:
        attn_weights_list: List of attention weight tensors from each batch
        modality_names: List of modality names

    Returns:
        Tuple of (mean_contributions, std_contributions) where each is a numpy array
        of shape (num_modalities,)
    """
    # Stack all attention weights
    all_attn = torch.cat(attn_weights_list, dim=0)  # (total_samples, num_modalities, num_modalities)

    # Average over source dimension (queries) to get how much each modality (key) is attended to
    # Shape: (total_samples, num_modalities)
    per_sample_attn = all_attn.mean(dim=1)

    # Convert to numpy
    per_sample_attn = per_sample_attn.numpy()

    # Compute statistics
    mean_contributions = per_sample_attn.mean(axis=0)
    std_contributions = per_sample_attn.std(axis=0)

    return mean_contributions, std_contributions


def save_attention_analysis(
    overall_contributions: Dict[str, float],
    contribution_stats: Dict[str, Dict[str, float]],
    save_dir: str,
    model_name: str
):
    """
    Save attention analysis results to JSON file.

    Args:
        overall_contributions: Overall modality contributions (mean)
        contribution_stats: Statistics of modality contributions (mean, std)
        save_dir: Directory to save results
        model_name: Name of the model (e.g., 'moltitox', 'smi_img')
    """
    os.makedirs(save_dir, exist_ok=True)

    results = {
        'model': model_name,
        'overall_modality_contributions': overall_contributions,
        'modality_contribution_statistics': contribution_stats
    }

    save_path = os.path.join(save_dir, f'{model_name}_attention_analysis.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAttention analysis saved to: {save_path}")


def print_attention_summary(
    overall_contributions: Dict[str, float],
    contribution_stats: Dict[str, Dict[str, float]],
    modality_names: List[str]
):
    """
    Print a summary of attention weights for interpretability.

    Args:
        overall_contributions: Overall modality contributions (mean)
        contribution_stats: Statistics of modality contributions (mean, std)
        modality_names: List of modality names
    """
    print("\n" + "=" * 60)
    print("ATTENTION WEIGHT ANALYSIS")
    print("=" * 60)
    print("\nNote: Attention weights are computed at the fusion stage")
    print("and are shared across all endpoints (tasks).")

    # Print overall contributions with statistics
    print("\nModality Contributions (Mean ± Std):")
    print("-" * 60)
    for name in modality_names:
        mean_weight = contribution_stats[name]['mean']
        std_weight = contribution_stats[name]['std']
        print(f"  {name:15s}: {mean_weight:.4f} ± {std_weight:.4f} ({mean_weight*100:.2f}%)")

    print("\n" + "=" * 60)


def compute_cross_modal_attention_matrix(
    attn_weights: torch.Tensor,
    modality_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-modal attention matrix showing how each modality attends to others.

    Args:
        attn_weights: Attention weights tensor of shape (batch_size, num_modalities, num_modalities)
                     where [i, j, k] represents sample i, query modality j attending to key modality k
        modality_names: List of modality names (e.g., ['Graph', 'SMILES', 'Image', 'Spectrum'])

    Returns:
        Tuple of (mean_matrix, std_matrix) where:
        - mean_matrix: Average attention from each modality (row) to each modality (column)
        - std_matrix: Standard deviation of attention weights across samples
        Both matrices have shape (num_modalities, num_modalities)
    """
    # Average over batch dimension to get mean attention matrix
    mean_matrix = attn_weights.mean(dim=0).cpu().numpy()  # Shape: (num_modalities, num_modalities)

    # Compute standard deviation across batch dimension
    std_matrix = attn_weights.std(dim=0).cpu().numpy()  # Shape: (num_modalities, num_modalities)

    return mean_matrix, std_matrix


def plot_cross_modal_attention_heatmap(
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray,
    modality_names: List[str],
    save_dir: str = None,
    model_name: str = None,
    title: str = "Cross-Modal Attention Matrix",
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Create a heatmap visualization of cross-modal attention weights.

    Args:
        mean_matrix: Mean attention matrix (num_modalities, num_modalities)
        std_matrix: Standard deviation matrix (num_modalities, num_modalities)
        modality_names: List of modality names
        save_dir: Directory to save the figure (will save to save_dir/image/)
        model_name: Name of the model (used for filename)
        title: Title for the plot
        figsize: Figure size (width, height)
    """
    num_modalities = len(modality_names)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap using seaborn
    sns.heatmap(
        mean_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=modality_names,
        yticklabels=modality_names,
        cbar_kws={'label': 'Attention Weight'},
        square=True,
        linewidths=0.5,
        linecolor='gray',
        vmin=0,
        vmax=1,
        ax=ax
    )

    # Add standard deviations as text annotations
    for i in range(num_modalities):
        for j in range(num_modalities):
            text = ax.text(
                j + 0.5, i + 0.7,
                f'±{std_matrix[i, j]:.3f}',
                ha='center', va='center',
                color='black', fontsize=8, style='italic'
            )

    ax.set_xlabel('Key Modality (Attended To)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Modality (Attending From)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save if directory and model name provided
    if save_dir and model_name:
        # Create image subfolder
        image_dir = os.path.join(save_dir, 'image')
        os.makedirs(image_dir, exist_ok=True)

        save_path = os.path.join(image_dir, f'{model_name}_cross_modal_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nCross-modal attention heatmap saved to: {save_path}")

    plt.close()

    return fig


def print_cross_modal_attention_matrix(
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray,
    modality_names: List[str]
):
    """
    Print cross-modal attention matrix in a readable format.

    Args:
        mean_matrix: Mean attention matrix (num_modalities, num_modalities)
        std_matrix: Standard deviation matrix (num_modalities, num_modalities)
        modality_names: List of modality names
    """
    num_modalities = len(modality_names)

    print("\n" + "=" * 80)
    print("CROSS-MODAL ATTENTION MATRIX (Mean ± Std)")
    print("=" * 80)
    print("\nRows: Query Modality (Attending From)")
    print("Columns: Key Modality (Attended To)")
    print("\nInterpretation: Higher values indicate stronger attention from row to column\n")

    # Print header
    header = "Query \\ Key".ljust(15)
    for name in modality_names:
        header += f"{name:>20s}"
    print(header)
    print("-" * (15 + 20 * num_modalities))

    # Print matrix rows
    for i, query_name in enumerate(modality_names):
        row = f"{query_name:15s}"
        for j in range(num_modalities):
            mean_val = mean_matrix[i, j]
            std_val = std_matrix[i, j]
            row += f"{mean_val:>10.4f} ± {std_val:<7.4f}"
        print(row)

    print("\n" + "=" * 80)


def save_cross_modal_attention_analysis(
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray,
    modality_names: List[str],
    save_dir: str,
    model_name: str
):
    """
    Save cross-modal attention matrix to JSON file.

    Args:
        mean_matrix: Mean attention matrix
        std_matrix: Standard deviation matrix
        modality_names: List of modality names
        save_dir: Directory to save results (will save to save_dir/json/)
        model_name: Name of the model
    """
    # Create json subfolder
    json_dir = os.path.join(save_dir, 'json')
    os.makedirs(json_dir, exist_ok=True)

    # Convert to nested dictionary format
    cross_modal_attention = {}
    for i, query_name in enumerate(modality_names):
        cross_modal_attention[query_name] = {}
        for j, key_name in enumerate(modality_names):
            cross_modal_attention[query_name][key_name] = {
                'mean': float(mean_matrix[i, j]),
                'std': float(std_matrix[i, j])
            }

    results = {
        'model': model_name,
        'cross_modal_attention_matrix': cross_modal_attention,
        'modality_names': modality_names
    }

    save_path = os.path.join(json_dir, f'{model_name}_cross_modal_attention.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nCross-modal attention analysis saved to: {save_path}")
