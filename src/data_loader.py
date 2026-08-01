"""
data_loader.py — Fetch, cache, and load disease datasets.

Provides a unified interface for loading Heart Disease, Diabetes,
and Breast Cancer datasets with automatic fallback:
    1. Primary source (ucimlrepo or sklearn.datasets)
    2. Bundled CSV in data/raw/

All downloaded data is cached locally so subsequent runs are offline-capable.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Project root is two levels up from this file (src/ -> disease-prediction/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def _ensure_dirs() -> None:
    """Create data directories if they don't exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_from_cache(dataset_name: str) -> pd.DataFrame | None:
    """Try to load a previously cached CSV."""
    cache_path = RAW_DATA_DIR / f"{dataset_name}.csv"
    if cache_path.exists():
        logger.info(f"Loading {dataset_name} from cache: {cache_path}")
        return pd.read_csv(cache_path)
    return None


def _save_to_cache(df: pd.DataFrame, dataset_name: str) -> None:
    """Save dataframe to CSV cache."""
    _ensure_dirs()
    cache_path = RAW_DATA_DIR / f"{dataset_name}.csv"
    df.to_csv(cache_path, index=False)
    logger.info(f"Cached {dataset_name} to {cache_path}")


# ---------------------------------------------------------------------------
# Heart Disease (Cleveland subset)
# ---------------------------------------------------------------------------

HEART_DISEASE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

HEART_DISEASE_FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment (0-2)",
    "ca": "Number of major vessels colored by fluoroscopy (0-3)",
    "thal": "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)",
}


def load_heart_disease() -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Load the UCI Heart Disease dataset (Cleveland subset).

    Returns
    -------
    df : pd.DataFrame
        The full dataset including features and target.
    target_col : str
        Name of the target column.
    metadata : dict
        Dataset metadata (description, feature info, etc.).
    """
    dataset_name = "heart_disease"
    target_col = "target"

    # Try cache first
    cached = _load_from_cache(dataset_name)
    if cached is not None:
        return cached, target_col, _heart_metadata()

    # Strategy 1: ucimlrepo
    try:
        from ucimlrepo import fetch_ucirepo
        logger.info("Fetching Heart Disease from UCI ML Repository...")
        dataset = fetch_ucirepo(id=45)
        features = dataset.data.features
        targets = dataset.data.targets

        df = pd.concat([features, targets], axis=1)

        # The UCI dataset may have multi-class target (0-4); binarize to 0/1
        if "num" in df.columns:
            df = df.rename(columns={"num": "target"})
        if target_col in df.columns:
            df[target_col] = (df[target_col] > 0).astype(int)

        _save_to_cache(df, dataset_name)
        logger.info(f"Heart Disease loaded: {df.shape}")
        return df, target_col, _heart_metadata()

    except Exception as e:
        logger.warning(f"ucimlrepo failed for Heart Disease: {e}")

    # Strategy 2: Direct CSV URL (UCI archive)
    try:
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        )
        logger.info("Fetching Heart Disease from UCI archive URL...")
        df = pd.read_csv(url, header=None, names=HEART_DISEASE_COLUMNS, na_values="?")
        df[target_col] = (df[target_col] > 0).astype(int)
        _save_to_cache(df, dataset_name)
        logger.info(f"Heart Disease loaded from URL: {df.shape}")
        return df, target_col, _heart_metadata()

    except Exception as e:
        logger.warning(f"URL fetch failed for Heart Disease: {e}")

    raise FileNotFoundError(
        f"Could not load Heart Disease dataset. Place a CSV at {RAW_DATA_DIR / 'heart_disease.csv'}"
    )


def _heart_metadata() -> Dict[str, Any]:
    return {
        "name": "Heart Disease",
        "source": "UCI ML Repository (Cleveland)",
        "n_classes": 2,
        "target_names": ["No Disease", "Disease"],
        "feature_descriptions": HEART_DISEASE_FEATURE_DESCRIPTIONS,
    }


# ---------------------------------------------------------------------------
# Diabetes (Pima Indians)
# ---------------------------------------------------------------------------

DIABETES_FEATURE_DESCRIPTIONS = {
    "Pregnancies": "Number of pregnancies",
    "Glucose": "Plasma glucose concentration (mg/dl)",
    "BloodPressure": "Diastolic blood pressure (mm Hg)",
    "SkinThickness": "Triceps skinfold thickness (mm)",
    "Insulin": "2-hour serum insulin (mu U/ml)",
    "BMI": "Body mass index (kg/m²)",
    "DiabetesPedigreeFunction": "Diabetes pedigree function (genetic risk score)",
    "Age": "Age in years",
}

DIABETES_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]


def load_diabetes() -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Load the Pima Indians Diabetes dataset.

    Returns
    -------
    df : pd.DataFrame
    target_col : str
    metadata : dict
    """
    dataset_name = "diabetes"
    target_col = "Outcome"

    cached = _load_from_cache(dataset_name)
    if cached is not None:
        return cached, target_col, _diabetes_metadata()

    # Strategy 1: ucimlrepo
    try:
        from ucimlrepo import fetch_ucirepo
        logger.info("Fetching Diabetes from UCI ML Repository...")
        dataset = fetch_ucirepo(id=34)
        features = dataset.data.features
        targets = dataset.data.targets

        df = pd.concat([features, targets], axis=1)

        # Standardize column names
        col_map = {}
        for expected in DIABETES_COLUMNS:
            for actual in df.columns:
                if actual.lower().replace("_", "") == expected.lower().replace("_", ""):
                    col_map[actual] = expected
                    break
        if col_map:
            df = df.rename(columns=col_map)

        # Ensure target column exists
        if target_col not in df.columns:
            # Try to find it
            for col in df.columns:
                if col.lower() in ("outcome", "class", "target"):
                    df = df.rename(columns={col: target_col})
                    break

        _save_to_cache(df, dataset_name)
        logger.info(f"Diabetes loaded: {df.shape}")
        return df, target_col, _diabetes_metadata()

    except Exception as e:
        logger.warning(f"ucimlrepo failed for Diabetes: {e}")

    # Strategy 2: Kaggle-hosted CSV URL
    try:
        url = (
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
            "pima-indians-diabetes.data.csv"
        )
        logger.info("Fetching Diabetes from GitHub mirror...")
        df = pd.read_csv(url, header=None, names=DIABETES_COLUMNS)
        _save_to_cache(df, dataset_name)
        logger.info(f"Diabetes loaded from URL: {df.shape}")
        return df, target_col, _diabetes_metadata()

    except Exception as e:
        logger.warning(f"URL fetch failed for Diabetes: {e}")

    raise FileNotFoundError(
        f"Could not load Diabetes dataset. Place a CSV at {RAW_DATA_DIR / 'diabetes.csv'}"
    )


def _diabetes_metadata() -> Dict[str, Any]:
    return {
        "name": "Diabetes",
        "source": "Pima Indians Diabetes (UCI/Kaggle)",
        "n_classes": 2,
        "target_names": ["No Diabetes", "Diabetes"],
        "feature_descriptions": DIABETES_FEATURE_DESCRIPTIONS,
    }


# ---------------------------------------------------------------------------
# Breast Cancer (Wisconsin Diagnostic)
# ---------------------------------------------------------------------------

BREAST_CANCER_FEATURE_DESCRIPTIONS = {
    "mean radius": "Mean of distances from center to points on the perimeter",
    "mean texture": "Standard deviation of gray-scale values",
    "mean perimeter": "Mean perimeter of the cell nucleus",
    "mean area": "Mean area of the cell nucleus",
    "mean smoothness": "Mean of local variation in radius lengths",
    "mean compactness": "Mean of (perimeter² / area - 1.0)",
    "mean concavity": "Mean severity of concave portions of the contour",
    "mean concave points": "Mean number of concave portions of the contour",
    "mean symmetry": "Mean symmetry of the cell nucleus",
    "mean fractal dimension": "Mean of 'coastline approximation' - 1",
    "radius error": "Standard error of radius",
    "texture error": "Standard error of texture",
    "perimeter error": "Standard error of perimeter",
    "area error": "Standard error of area",
    "smoothness error": "Standard error of smoothness",
    "compactness error": "Standard error of compactness",
    "concavity error": "Standard error of concavity",
    "concave points error": "Standard error of concave points",
    "symmetry error": "Standard error of symmetry",
    "fractal dimension error": "Standard error of fractal dimension",
    "worst radius": "Worst (largest) radius",
    "worst texture": "Worst (largest) texture",
    "worst perimeter": "Worst (largest) perimeter",
    "worst area": "Worst (largest) area",
    "worst smoothness": "Worst (largest) smoothness",
    "worst compactness": "Worst (largest) compactness",
    "worst concavity": "Worst (largest) concavity",
    "worst concave points": "Worst (largest) concave points",
    "worst symmetry": "Worst (largest) symmetry",
    "worst fractal dimension": "Worst (largest) fractal dimension",
}


def load_breast_cancer() -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Load the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

    Returns
    -------
    df : pd.DataFrame
    target_col : str
    metadata : dict
    """
    dataset_name = "breast_cancer"
    target_col = "target"

    cached = _load_from_cache(dataset_name)
    if cached is not None:
        return cached, target_col, _breast_cancer_metadata()

    # Strategy 1: sklearn (always available, no network needed)
    try:
        from sklearn.datasets import load_breast_cancer as _load_bc
        logger.info("Loading Breast Cancer from sklearn.datasets...")
        data = _load_bc()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df[target_col] = data.target
        _save_to_cache(df, dataset_name)
        logger.info(f"Breast Cancer loaded: {df.shape}")
        return df, target_col, _breast_cancer_metadata()

    except Exception as e:
        logger.warning(f"sklearn loading failed for Breast Cancer: {e}")

    raise FileNotFoundError(
        f"Could not load Breast Cancer dataset. Place a CSV at {RAW_DATA_DIR / 'breast_cancer.csv'}"
    )


def _breast_cancer_metadata() -> Dict[str, Any]:
    return {
        "name": "Breast Cancer",
        "source": "UCI ML Repository (Wisconsin Diagnostic) via sklearn",
        "n_classes": 2,
        "target_names": ["Malignant", "Benign"],
        "feature_descriptions": BREAST_CANCER_FEATURE_DESCRIPTIONS,
    }


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

LOADERS = {
    "heart_disease": load_heart_disease,
    "diabetes": load_diabetes,
    "breast_cancer": load_breast_cancer,
}


def load_dataset(name: str) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        One of 'heart_disease', 'diabetes', 'breast_cancer'.

    Returns
    -------
    df : pd.DataFrame
    target_col : str
    metadata : dict
    """
    if name not in LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(LOADERS.keys())}"
        )
    return LOADERS[name]()


def load_all_datasets() -> Dict[str, Tuple[pd.DataFrame, str, Dict[str, Any]]]:
    """Load all three datasets. Returns dict keyed by dataset name."""
    results = {}
    for name in LOADERS:
        try:
            results[name] = load_dataset(name)
            logger.info(f"✓ {name} loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load {name}: {e}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets = load_all_datasets()
    for name, (df, target, meta) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {meta['name']}")
        print(f"Shape: {df.shape}")
        print(f"Target column: {target}")
        print(f"Class distribution:\n{df[target].value_counts()}")
        print(f"Missing values: {df.isnull().sum().sum()}")
