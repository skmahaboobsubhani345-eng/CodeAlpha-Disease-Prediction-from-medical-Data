"""
models.py — ML model definitions and hyperparameter tuning.

Provides base model constructors and GridSearchCV-based tuning
for Random Forest and XGBoost classifiers.
"""

import logging
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def get_base_models() -> Dict[str, Any]:
    """
    Return a dictionary of base (untuned) classifiers.

    All models are configured with probability=True for ROC-AUC computation,
    and use class_weight='balanced' where supported as a complementary
    approach to SMOTE.

    Returns
    -------
    dict
        {model_name: estimator_instance}
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    scoring: str = "f1",
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Tune Random Forest using GridSearchCV.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric for grid search.

    Returns
    -------
    best_model : RandomForestClassifier
        Best estimator from grid search.
    best_params : dict
        Best hyperparameters found.
    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    base = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    logger.info("  Tuning Random Forest (GridSearchCV)...")
    grid = GridSearchCV(
        base,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid.fit(X_train, y_train)

    logger.info(f"  Best RF params: {grid.best_params_}")
    logger.info(f"  Best RF CV {scoring}: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    scoring: str = "f1",
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Tune XGBoost using GridSearchCV.

    Parameters
    ----------
    X_train, y_train : np.ndarray
    cv : int
    scoring : str

    Returns
    -------
    best_model : XGBClassifier
    best_params : dict
    """
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    logger.info("  Tuning XGBoost (GridSearchCV)...")
    grid = GridSearchCV(
        base,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid.fit(X_train, y_train)

    logger.info(f"  Best XGB params: {grid.best_params_}")
    logger.info(f"  Best XGB CV {scoring}: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_
