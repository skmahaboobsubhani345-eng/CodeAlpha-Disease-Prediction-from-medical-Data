"""
preprocessing.py — Data cleaning, transformation, and preparation pipeline.

Handles missing values, feature scaling, categorical encoding,
and class imbalance for all disease prediction datasets.
"""

import logging
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def handle_missing_values(
    df: pd.DataFrame,
    dataset_name: str = "",
) -> pd.DataFrame:
    """
    Handle missing values with domain-aware imputation.

    For the Diabetes dataset, zeros in certain columns (Glucose, BloodPressure,
    SkinThickness, Insulin, BMI) are biologically impossible and treated as NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (will not be modified in-place).
    dataset_name : str
        Name of the dataset for domain-specific handling.

    Returns
    -------
    pd.DataFrame
        Dataframe with missing values imputed.
    """
    df = df.copy()

    # Domain-specific: Pima Diabetes has 0s that should be NaN
    if dataset_name == "diabetes":
        zero_invalid_cols = [
            "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
        ]
        for col in zero_invalid_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
                logger.info(
                    f"  {col}: replaced {(df[col].isna()).sum()} zeros with NaN"
                )

    # Impute numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        num_missing_before = df[numeric_cols].isnull().sum().sum()
        if num_missing_before > 0:
            imputer = SimpleImputer(strategy="median")
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            logger.info(f"  Imputed {num_missing_before} missing numeric values (median)")

    # Impute categorical columns with mode
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        num_missing_before = df[cat_cols].isnull().sum().sum()
        if num_missing_before > 0:
            imputer = SimpleImputer(strategy="most_frequent")
            df[cat_cols] = imputer.fit_transform(df[cat_cols])
            logger.info(f"  Imputed {num_missing_before} missing categorical values (mode)")

    return df


def encode_categoricals(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features using LabelEncoder.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Target column name (excluded from encoding).

    Returns
    -------
    df : pd.DataFrame
        Encoded dataframe.
    encoders : dict
        Mapping of column name → fitted LabelEncoder.
    """
    df = df.copy()
    encoders = {}

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info(f"  Encoded '{col}': {list(le.classes_)}")

    return df, encoders


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler (fit on train only).

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices.

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
    scaler : StandardScaler
        Fitted scaler (needed for inference).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("  Features scaled with StandardScaler")
    return X_train_scaled, X_test_scaled, scaler


def handle_class_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "smote",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle class imbalance using SMOTE oversampling.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    method : str
        'smote' for SMOTE oversampling, 'none' to skip.
    random_state : int

    Returns
    -------
    X_resampled, y_resampled : np.ndarray
    """
    if method == "none":
        return X, y

    original_dist = dict(zip(*np.unique(y, return_counts=True)))

    if method == "smote":
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)
        new_dist = dict(zip(*np.unique(y_res, return_counts=True)))
        logger.info(f"  SMOTE resampling: {original_dist} → {new_dist}")
        return X_res, y_res

    raise ValueError(f"Unknown imbalance method: {method}")


def preprocess_pipeline(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str = "",
    test_size: float = 0.2,
    apply_smote: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline: clean → encode → split → scale → balance.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset including target.
    target_col : str
        Name of the target column.
    dataset_name : str
        Dataset identifier for domain-specific handling.
    test_size : float
        Fraction of data for testing.
    apply_smote : bool
        Whether to apply SMOTE to the training set.
    random_state : int

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test : np.ndarray
        scaler : StandardScaler
        feature_names : list[str]
        encoders : dict
    """
    logger.info(f"Preprocessing '{dataset_name}' ({df.shape[0]} rows, {df.shape[1]} cols)")

    # 1. Handle missing values
    df = handle_missing_values(df, dataset_name)

    # 2. Encode categoricals
    df, encoders = encode_categoricals(df, target_col)

    # 3. Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].values.astype(int)
    feature_names = X.columns.tolist()

    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    # 4. Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        f"  Split: train={X_train.shape[0]}, test={X_test.shape[0]}"
    )

    # 5. Scale features
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # 6. Handle class imbalance (train set only)
    if apply_smote:
        X_train, y_train = handle_class_imbalance(X_train, y_train)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names,
        "encoders": encoders,
    }
