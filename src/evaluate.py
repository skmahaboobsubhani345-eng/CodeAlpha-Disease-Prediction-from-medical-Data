"""
evaluate.py — Model evaluation, metrics computation, and explainability.

Computes classification metrics, generates ROC curves, confusion matrices,
comparison tables, and SHAP-based feature importance explanations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from src.data_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def _ensure_results_dir(dataset_name: str) -> Path:
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        {accuracy, precision, recall, f1, roc_auc}
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    target_names: List[str],
    save_path: Optional[Path] = None,
) -> None:
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved confusion matrix: {save_path}")
    plt.close(fig)


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot ROC curves for multiple models on the same axes.

    Parameters
    ----------
    roc_data : dict
        {model_name: (fpr, tpr, auc_score)}
    title : str
    save_path : Path, optional
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("husl", len(roc_data))

    for (name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved ROC curves: {save_path}")
    plt.close(fig)


def plot_comparison_table(
    results: Dict[str, Dict[str, float]],
    title: str,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create and save a model comparison table.

    Parameters
    ----------
    results : dict
        {model_name: {metric_name: value}}
    title : str
    save_path : Path, optional

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    df = pd.DataFrame(results).T
    df = df.round(4)
    df.index.name = "Model"

    # Save as CSV
    if save_path:
        csv_path = save_path.with_suffix(".csv")
        df.to_csv(csv_path)
        logger.info(f"  Saved comparison table: {csv_path}")

        # Also save as a styled image
        fig, ax = plt.subplots(figsize=(12, 3 + 0.5 * len(df)))
        ax.axis("off")

        table = ax.table(
            cellText=df.values,
            colLabels=[c.upper() for c in df.columns],
            rowLabels=df.index,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style header
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2C3E50")
                cell.set_text_props(color="white", fontweight="bold")
            elif col == -1:
                cell.set_facecolor("#34495E")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#ECF0F1" if row % 2 == 0 else "white")

        ax.set_title(title, fontweight="bold", fontsize=14, pad=20)
        plt.tight_layout()

        img_path = save_path.with_suffix(".png")
        fig.savefig(img_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved comparison image: {img_path}")
        plt.close(fig)

    return df


def _extract_shap_values(shap_obj: Any, is_single_sample: bool = False) -> np.ndarray:
    """Helper to extract positive class SHAP values across SHAP version array shapes."""
    if hasattr(shap_obj, "values"):
        shap_obj = shap_obj.values
    if isinstance(shap_obj, list):
        shap_obj = shap_obj[1]
    arr = np.asarray(shap_obj)
    if arr.ndim == 3:
        # Shape: (n_samples, n_features, n_classes)
        arr = arr[:, :, 1]
    if is_single_sample and arr.ndim == 2:
        return arr[0]
    return arr


def get_feature_importance(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    model_name: str = "",
    use_shap: bool = True,
    max_samples: int = 100,
) -> pd.DataFrame:
    """
    Get feature importance using SHAP or model-native methods.

    For tree-based models, uses SHAP TreeExplainer for precise attributions.
    For linear models, uses coefficient magnitudes.
    Falls back to permutation importance if SHAP fails.

    Parameters
    ----------
    model : fitted estimator
    X : np.ndarray
        Sample data for SHAP computation.
    feature_names : list[str]
    model_name : str
    use_shap : bool
        Whether to attempt SHAP-based explanation.
    max_samples : int
        Max samples for SHAP computation (performance).

    Returns
    -------
    pd.DataFrame
        Sorted feature importance with columns: feature, importance.
    """
    importance_values = None

    if use_shap:
        try:
            import shap

            # Subsample for performance
            if X.shape[0] > max_samples:
                idx = np.random.RandomState(42).choice(X.shape[0], max_samples, replace=False)
                X_sample = X[idx]
            else:
                X_sample = X

            # Choose appropriate explainer
            model_type = type(model).__name__
            if model_type in ("RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier"):
                explainer = shap.TreeExplainer(model)
                raw_shap = explainer.shap_values(X_sample)
                shap_matrix = _extract_shap_values(raw_shap)
                importance_values = np.abs(shap_matrix).mean(axis=0)
                logger.info(f"  SHAP TreeExplainer used for {model_name}")
            else:
                # Linear models: use coefficients directly (faster than KernelExplainer)
                if hasattr(model, "coef_"):
                    importance_values = np.abs(model.coef_).flatten()
                    logger.info(f"  Coefficient-based importance for {model_name}")
                else:
                    # Fallback to KernelExplainer with small background
                    background = shap.sample(X_sample, min(20, len(X_sample)))
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    raw_shap = explainer.shap_values(X_sample[:20], nsamples=100)
                    shap_matrix = _extract_shap_values(raw_shap)
                    importance_values = np.abs(shap_matrix).mean(axis=0)
                    logger.info(f"  SHAP KernelExplainer used for {model_name}")

        except Exception as e:
            logger.warning(f"  SHAP failed for {model_name}: {e}")

    # Fallback: model-native feature importances
    if importance_values is None:
        if hasattr(model, "feature_importances_"):
            importance_values = model.feature_importances_
            logger.info(f"  Using native feature_importances_ for {model_name}")
        elif hasattr(model, "coef_"):
            importance_values = np.abs(model.coef_).flatten()
            logger.info(f"  Using |coefficients| for {model_name}")
        else:
            importance_values = np.zeros(len(feature_names))
            logger.warning(f"  No feature importance available for {model_name}")

    # Ensure length matches
    if len(importance_values) != len(feature_names):
        importance_values = importance_values[:len(feature_names)]

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_values,
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    return df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str,
    top_n: int = 10,
    save_path: Optional[Path] = None,
) -> None:
    """Plot horizontal bar chart of top feature importances."""
    top = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("viridis", len(top))
    ax.barh(top["feature"], top["importance"], color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Importance (mean |SHAP value|)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved feature importance: {save_path}")
    plt.close(fig)


def explain_prediction(
    model: Any,
    sample: np.ndarray,
    feature_names: List[str],
    feature_descriptions: Dict[str, str],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Generate a plain-language explanation of a single prediction.

    Parameters
    ----------
    model : fitted estimator
    sample : np.ndarray
        Single sample (1D array).
    feature_names : list[str]
    feature_descriptions : dict
        Human-readable descriptions for each feature.
    top_n : int
        Number of top features to explain.

    Returns
    -------
    list[dict]
        Each dict has: feature, value, importance, direction, description.
    """
    sample_2d = sample.reshape(1, -1)

    try:
        import shap
        model_type = type(model).__name__

        if model_type in ("RandomForestClassifier", "XGBClassifier"):
            explainer = shap.TreeExplainer(model)
            raw_shap = explainer.shap_values(sample_2d)
            values = _extract_shap_values(raw_shap, is_single_sample=True)
        elif hasattr(model, "coef_"):
            values = (model.coef_.flatten() * sample).flatten()
        else:
            background = np.zeros((1, len(feature_names)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            raw_shap = explainer.shap_values(sample_2d, nsamples=100)
            values = _extract_shap_values(raw_shap, is_single_sample=True)

    except Exception:
        # Fallback: coefficient or importance * feature value
        if hasattr(model, "coef_"):
            values = (model.coef_.flatten() * sample).flatten()
        elif hasattr(model, "feature_importances_"):
            values = model.feature_importances_ * np.abs(sample)
        else:
            values = np.zeros(len(feature_names))

    values = np.asarray(values).flatten()

    # Sort by absolute contribution
    indices = np.argsort(np.abs(values))[::-1][:top_n]

    explanations = []
    for idx in indices:
        feat = feature_names[idx]
        desc = feature_descriptions.get(feat, feat)
        direction = "increases" if values[idx] > 0 else "decreases"

        explanations.append({
            "feature": feat,
            "value": float(sample[idx]),
            "importance": float(abs(values[idx])),
            "direction": direction,
            "description": desc,
            "explanation": (
                f"{desc} ({feat} = {sample[idx]:.2f}) "
                f"{direction} the risk by {abs(values[idx]):.3f}"
            ),
        })

    return explanations


def evaluate_all_models(
    trained_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
    target_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all trained models for a dataset and generate all plots.

    Parameters
    ----------
    trained_models : dict
        {model_name: fitted_estimator}
    X_test, y_test : np.ndarray
    feature_names : list[str]
    dataset_name : str
    target_names : list[str]

    Returns
    -------
    dict
        {model_name: {metric: value}}
    """
    out_dir = _ensure_results_dir(dataset_name)
    display_name = dataset_name.replace("_", " ").title()
    all_metrics = {}
    roc_data = {}

    for model_name, model in trained_models.items():
        logger.info(f"  Evaluating {model_name}...")

        y_pred = model.predict(X_test)

        # Get probabilities
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        all_metrics[model_name] = metrics

        # Confusion matrix
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plot_confusion_matrix(
            y_test, y_pred,
            title=f"{display_name} — {model_name} Confusion Matrix",
            target_names=target_names,
            save_path=out_dir / f"confusion_matrix_{safe_name}.png",
        )

        # ROC data
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[model_name] = (fpr, tpr, metrics["roc_auc"])

        # Feature importance (for best model per dataset, done later)
        logger.info(
            f"    {model_name}: Acc={metrics['accuracy']:.4f} "
            f"F1={metrics['f1']:.4f} AUC={metrics['roc_auc']:.4f}"
        )

    # ROC curves (all models)
    if roc_data:
        plot_roc_curves(
            roc_data,
            title=f"{display_name} — ROC Curves",
            save_path=out_dir / "roc_curves.png",
        )

    # Comparison table
    plot_comparison_table(
        all_metrics,
        title=f"{display_name} — Model Comparison",
        save_path=out_dir / "comparison_table",
    )

    return all_metrics
