"""
eda.py — Exploratory Data Analysis visualizations.

Generates correlation heatmaps, class distribution charts,
and feature distribution plots for each disease dataset.
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

EDA_DIR = PROJECT_ROOT / "data" / "eda"

# Consistent styling
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def _ensure_eda_dir(dataset_name: str) -> Path:
    """Create EDA output directory for a dataset."""
    out_dir = EDA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_class_distribution(
    y: np.ndarray,
    target_names: List[str],
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot class distribution as a bar chart.

    Parameters
    ----------
    y : array-like
        Target values.
    target_names : list[str]
        Human-readable class names.
    title : str
        Plot title.
    save_path : Path, optional
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    unique, counts = np.unique(y, return_counts=True)
    labels = [target_names[int(u)] if int(u) < len(target_names) else str(u) for u in unique]
    colors = sns.color_palette("husl", len(unique))

    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f"{count}\n({count / sum(counts) * 100:.1f}%)",
            ha="center", va="bottom", fontweight="bold",
        )

    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {save_path}")
    plt.close(fig)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot Pearson correlation heatmap for numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with numeric features.
    title : str
        Plot title.
    save_path : Path, optional
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True if corr.shape[0] <= 15 else False,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )

    ax.set_title(title, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {save_path}")
    plt.close(fig)


def plot_feature_distributions(
    df: pd.DataFrame,
    target_col: str,
    target_names: List[str],
    title_prefix: str,
    save_path: Optional[Path] = None,
    max_features: int = 12,
) -> None:
    """
    Plot histograms/KDE of features split by target class.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    target_names : list[str]
    title_prefix : str
    save_path : Path, optional
    max_features : int
        Maximum number of features to plot.
    """
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    feature_cols = feature_cols[:max_features]

    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        for cls_val in sorted(df[target_col].unique()):
            label = target_names[int(cls_val)] if int(cls_val) < len(target_names) else str(cls_val)
            subset = df[df[target_col] == cls_val][col].dropna()
            ax.hist(subset, bins=25, alpha=0.6, label=label, edgecolor="white")
        ax.set_title(col, fontweight="bold")
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{title_prefix} — Feature Distributions", fontweight="bold", fontsize=16, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {save_path}")
    plt.close(fig)


def run_full_eda(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str,
    target_names: List[str],
) -> Path:
    """
    Run complete EDA and save all plots for a dataset.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    dataset_name : str
    target_names : list[str]

    Returns
    -------
    Path
        Directory where plots were saved.
    """
    out_dir = _ensure_eda_dir(dataset_name)
    display_name = dataset_name.replace("_", " ").title()

    logger.info(f"Running EDA for {display_name}...")

    # 1. Class distribution
    plot_class_distribution(
        y=df[target_col].values,
        target_names=target_names,
        title=f"{display_name} — Class Distribution",
        save_path=out_dir / "class_distribution.png",
    )

    # 2. Correlation heatmap
    plot_correlation_heatmap(
        df=df,
        title=f"{display_name} — Feature Correlation Heatmap",
        save_path=out_dir / "correlation_heatmap.png",
    )

    # 3. Feature distributions
    plot_feature_distributions(
        df=df,
        target_col=target_col,
        target_names=target_names,
        title_prefix=display_name,
        save_path=out_dir / "feature_distributions.png",
    )

    logger.info(f"EDA complete for {display_name}. Plots saved to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_all_datasets

    datasets = load_all_datasets()
    for name, (df, target_col, meta) in datasets.items():
        run_full_eda(df, target_col, name, meta["target_names"])
