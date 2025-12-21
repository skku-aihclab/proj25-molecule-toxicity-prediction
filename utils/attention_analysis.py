import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def compute_modality_contributions(
    attn_weights: torch.Tensor,
    modality_names: List[str]
) -> Dict[str, float]:
    """
    Compute average contribution of each modality.
    Args:
        attn_weights: shape (batch_size, num_modalities, num_modalities)
        modality_names: list of modality names
    """
    # Average over batch dimension and source dimension (queries)
    # This gives us how much each modality (key) is attended to on average
    avg_attn = attn_weights.mean(dim=0).mean(dim=0)

    contributions = {}
    for i, name in enumerate(modality_names):
        contributions[name] = float(avg_attn[i].item())

    return contributions


def compute_sample_wise_contributions(
    attn_weights_list: List[torch.Tensor],
    modality_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sample-wise statistics of modality contributions.
    Args:
        attn_weights_list: list of attention tensors from each batch
        modality_names: list of modality names
    """
    # Stack all attention weights
    all_attn = torch.cat(attn_weights_list, dim=0)

    # Average over source dimension (queries) to get how much each modality (key) is attended to
    per_sample_attn = all_attn.mean(dim=1)

    # Convert to numpy
    per_sample_attn = per_sample_attn.numpy()

    # Compute statistics
    mean_contributions = per_sample_attn.mean(axis=0)
    std_contributions = per_sample_attn.std(axis=0)

    return mean_contributions, std_contributions


def compute_cross_modal_attention_matrix(
    attn_weights: torch.Tensor,
    modality_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-modal attention matrix (mean, std).
    Args:
        attn_weights: shape (batch_size, num_modalities, num_modalities)
        modality_names: list of modality names
    """
    # Average over batch dimension to get mean attention matrix
    mean_matrix = attn_weights.mean(dim=0).cpu().numpy()

    # Compute standard deviation across batch dimension
    std_matrix = attn_weights.std(dim=0).cpu().numpy()

    return mean_matrix, std_matrix


def plot_cross_modal_attention_heatmap(
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray,
    modality_names: List[str],
    title: str = "Cross-Modal Attention Matrix",
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Create heatmap visualization of cross-modal attention.
    Args:
        mean_matrix: mean attention matrix
        std_matrix: std attention matrix
        modality_names: list of modality names
        title: plot title
        figsize: figure size
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
            ax.text(
                j + 0.5, i + 0.7,
                f'±{std_matrix[i, j]:.3f}',
                ha='center', va='center',
                color='black', fontsize=8, style='italic'
            )

    ax.set_xlabel('Key Modality (Attended To)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Modality (Attending From)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    return fig


def print_cross_modal_attention_matrix(
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray,
    modality_names: List[str]
):
    """
    Print cross-modal attention matrix.
    Args:
        mean_matrix: mean attention matrix
        std_matrix: std attention matrix
        modality_names: list of modality names
    """
    num_modalities = len(modality_names)

    print("\n" + "=" * 80)
    print("CROSS-MODAL ATTENTION MATRIX (Mean ± Std)")
    print("=" * 80)
    print("\nRows: Query Modality (Attending From)")
    print("Columns: Key Modality (Attended To)\n")

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
