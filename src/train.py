"""
train.py — Training orchestrator for all disease prediction pipelines.

Trains 4 classifiers per dataset, tunes RF and XGBoost via GridSearchCV,
evaluates on holdout set with 5-fold cross-validation, and saves the
best-performing model per dataset.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import joblib
from sklearn.model_selection import cross_val_score

from src.data_loader import load_dataset, PROJECT_ROOT, LOADERS
from src.preprocessing import preprocess_pipeline
from src.models import get_base_models, tune_random_forest, tune_xgboost
from src.evaluate import (
    evaluate_all_models,
    get_feature_importance,
    plot_feature_importance,
)
from src.eda import run_full_eda

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_single_dataset(
    dataset_name: str,
    tune: bool = True,
    run_eda: bool = True,
) -> Dict[str, Any]:
    """
    Full training pipeline for a single dataset.

    Steps:
        1. Load dataset
        2. Run EDA (optional)
        3. Preprocess
        4. Train base models + 5-fold CV
        5. Hyperparameter-tune RF and XGBoost
        6. Evaluate all models
        7. Save best model

    Parameters
    ----------
    dataset_name : str
        One of 'heart_disease', 'diabetes', 'breast_cancer'.
    tune : bool
        Whether to run hyperparameter tuning.
    run_eda : bool
        Whether to generate EDA plots.

    Returns
    -------
    dict
        Training results including metrics, best model name, and paths.
    """
    _ensure_dirs()
    start_time = time.time()

    display_name = dataset_name.replace("_", " ").title()
    logger.info(f"\n{'='*70}")
    logger.info(f"  TRAINING PIPELINE: {display_name}")
    logger.info(f"{'='*70}")

    # 1. Load
    logger.info(f"\n[1/7] Loading {display_name}...")
    df, target_col, metadata = load_dataset(dataset_name)
    logger.info(f"  Shape: {df.shape}, Target: '{target_col}'")

    # 2. EDA
    if run_eda:
        logger.info(f"\n[2/7] Running EDA...")
        run_full_eda(df, target_col, dataset_name, metadata["target_names"])

    # 3. Preprocess
    logger.info(f"\n[3/7] Preprocessing...")
    prep = preprocess_pipeline(df, target_col, dataset_name)
    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]
    feature_names = prep["feature_names"]
    scaler = prep["scaler"]

    logger.info(
        f"  After preprocessing: train={X_train.shape}, test={X_test.shape}"
    )

    # 4. Train base models with 5-fold CV
    logger.info(f"\n[4/7] Training base models with 5-fold CV...")
    base_models = get_base_models()
    trained_models = {}
    cv_scores = {}

    for model_name, model in base_models.items():
        logger.info(f"  Training {model_name}...")

        # 5-fold cross-validation
        scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="f1", n_jobs=-1
        )
        cv_scores[model_name] = {
            "mean_f1": float(scores.mean()),
            "std_f1": float(scores.std()),
        }
        logger.info(
            f"    CV F1: {scores.mean():.4f} ± {scores.std():.4f}"
        )

        # Fit on full training set
        model.fit(X_train, y_train)
        trained_models[model_name] = model

    # 5. Hyperparameter tuning
    if tune:
        logger.info(f"\n[5/7] Hyperparameter tuning...")

        # Tune Random Forest
        tuned_rf, rf_params = tune_random_forest(X_train, y_train)
        trained_models["Random Forest (Tuned)"] = tuned_rf

        # Tune XGBoost
        tuned_xgb, xgb_params = tune_xgboost(X_train, y_train)
        trained_models["XGBoost (Tuned)"] = tuned_xgb
    else:
        logger.info(f"\n[5/7] Skipping hyperparameter tuning")

    # 6. Evaluate
    logger.info(f"\n[6/7] Evaluating all models...")
    all_metrics = evaluate_all_models(
        trained_models, X_test, y_test, feature_names,
        dataset_name, metadata["target_names"],
    )

    # Print comparison
    logger.info(f"\n  {'Model':<30} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    logger.info(f"  {'-'*60}")
    for model_name, metrics in all_metrics.items():
        logger.info(
            f"  {model_name:<30} {metrics['accuracy']:>10.4f} "
            f"{metrics['f1']:>10.4f} {metrics['roc_auc']:>10.4f}"
        )

    # 7. Save best model
    logger.info(f"\n[7/7] Saving best model...")
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_model = trained_models[best_model_name]
    best_metrics = all_metrics[best_model_name]

    # Save model + metadata
    model_path = MODELS_DIR / f"{dataset_name}_best_model.pkl"
    model_bundle = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "metadata": metadata,
        "best_model_name": best_model_name,
        "metrics": best_metrics,
    }
    joblib.dump(model_bundle, model_path)
    logger.info(f"  Best model: {best_model_name}")
    logger.info(f"  Saved to: {model_path}")

    # Feature importance for best model
    results_dir = RESULTS_DIR / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    importance_df = get_feature_importance(
        best_model, X_test, feature_names, best_model_name
    )
    plot_feature_importance(
        importance_df,
        title=f"{display_name} — Feature Importance ({best_model_name})",
        save_path=results_dir / "feature_importance.png",
    )
    importance_df.to_csv(results_dir / "feature_importance.csv", index=False)

    elapsed = time.time() - start_time
    logger.info(f"\n  Pipeline completed in {elapsed:.1f}s")
    logger.info(f"  Best model: {best_model_name} (F1={best_metrics['f1']:.4f})")

    # Save complete results
    results = {
        "dataset": dataset_name,
        "display_name": display_name,
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "all_metrics": all_metrics,
        "cv_scores": cv_scores,
        "feature_names": feature_names,
        "model_path": str(model_path),
        "elapsed_seconds": elapsed,
    }

    results_path = RESULTS_DIR / dataset_name / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def train_all(tune: bool = True, run_eda: bool = True) -> Dict[str, Any]:
    """
    Train all disease prediction pipelines.

    Returns
    -------
    dict
        {dataset_name: results_dict}
    """
    all_results = {}

    for dataset_name in LOADERS:
        try:
            results = train_single_dataset(dataset_name, tune=tune, run_eda=run_eda)
            all_results[dataset_name] = results
        except Exception as e:
            logger.error(f"Failed to train {dataset_name}: {e}", exc_info=True)

    # Print final summary
    logger.info(f"\n{'='*70}")
    logger.info(f"  FINAL SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"  {'Dataset':<20} {'Best Model':<30} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    logger.info(f"  {'-'*80}")

    for name, res in all_results.items():
        m = res["best_metrics"]
        logger.info(
            f"  {res['display_name']:<20} {res['best_model_name']:<30} "
            f"{m['accuracy']:>10.4f} {m['f1']:>10.4f} {m['roc_auc']:>10.4f}"
        )

    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    summary = {
        name: {
            "best_model": res["best_model_name"],
            "metrics": res["best_metrics"],
        }
        for name, res in all_results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n  Summary saved to: {summary_path}")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    train_all()
