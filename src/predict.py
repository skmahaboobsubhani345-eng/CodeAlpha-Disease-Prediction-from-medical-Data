"""
predict.py — Prediction interface for disease prediction models.

Provides a CLI and programmatic API for loading trained models
and making predictions on new patient data with explanations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import joblib

from src.data_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"

# Feature metadata for CLI input guidance
DATASET_FEATURES = {
    "heart_disease": {
        "names": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ],
        "descriptions": {
            "age": "Age in years",
            "sex": "Sex (1=male, 0=female)",
            "cp": "Chest pain type (0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic)",
            "trestbps": "Resting blood pressure (mm Hg)",
            "chol": "Serum cholesterol (mg/dl)",
            "fbs": "Fasting blood sugar > 120 mg/dl (1=true, 0=false)",
            "restecg": "Resting ECG (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)",
            "thalach": "Maximum heart rate achieved",
            "exang": "Exercise-induced angina (1=yes, 0=no)",
            "oldpeak": "ST depression induced by exercise relative to rest",
            "slope": "Slope of peak exercise ST segment (0=upsloping, 1=flat, 2=downsloping)",
            "ca": "Number of major vessels colored by fluoroscopy (0-3)",
            "thal": "Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)",
        },
        "example": "63,1,3,145,233,1,0,150,0,2.3,0,0,1",
    },
    "diabetes": {
        "names": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        ],
        "descriptions": {
            "Pregnancies": "Number of pregnancies",
            "Glucose": "Plasma glucose concentration (mg/dl)",
            "BloodPressure": "Diastolic blood pressure (mm Hg)",
            "SkinThickness": "Triceps skin fold thickness (mm)",
            "Insulin": "2-hour serum insulin (mu U/ml)",
            "BMI": "Body mass index (kg/m²)",
            "DiabetesPedigreeFunction": "Diabetes pedigree function score",
            "Age": "Age in years",
        },
        "example": "6,148,72,35,0,33.6,0.627,50",
    },
    "breast_cancer": {
        "names": [
            "mean radius", "mean texture", "mean perimeter", "mean area",
            "mean smoothness", "mean compactness", "mean concavity",
            "mean concave points", "mean symmetry", "mean fractal dimension",
            "radius error", "texture error", "perimeter error", "area error",
            "smoothness error", "compactness error", "concavity error",
            "concave points error", "symmetry error", "fractal dimension error",
            "worst radius", "worst texture", "worst perimeter", "worst area",
            "worst smoothness", "worst compactness", "worst concavity",
            "worst concave points", "worst symmetry", "worst fractal dimension",
        ],
        "descriptions": {
            "mean radius": "Mean of distances from center to perimeter points",
            "mean texture": "Standard deviation of gray-scale values",
            "mean perimeter": "Mean perimeter of the cell nucleus",
            "mean area": "Mean area of the cell nucleus",
            "mean smoothness": "Mean local variation in radius lengths",
            "mean compactness": "Mean of (perimeter²/area - 1.0)",
            "mean concavity": "Mean severity of concave portions",
            "mean concave points": "Mean number of concave contour portions",
            "mean symmetry": "Mean symmetry of the cell nucleus",
            "mean fractal dimension": "Mean 'coastline approximation' - 1",
        },
        "example": "17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189",
    },
}


def load_model(dataset_name: str) -> Dict[str, Any]:
    """
    Load a saved model bundle.

    Parameters
    ----------
    dataset_name : str
        One of 'heart_disease', 'diabetes', 'breast_cancer'.

    Returns
    -------
    dict
        Model bundle with keys: model, scaler, feature_names, metadata, etc.
    """
    model_path = MODELS_DIR / f"{dataset_name}_best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            f"Run 'python -m src.train' first."
        )
    bundle = joblib.load(model_path)
    logger.info(f"Loaded {bundle['best_model_name']} for {dataset_name}")
    return bundle


def predict_from_array(
    dataset_name: str,
    features: np.ndarray,
) -> Dict[str, Any]:
    """
    Make a prediction from a feature array.

    Parameters
    ----------
    dataset_name : str
    features : np.ndarray
        Raw feature values (unscaled).

    Returns
    -------
    dict
        prediction, probability, model_name, explanations.
    """
    bundle = load_model(dataset_name)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    metadata = bundle["metadata"]

    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict
    prediction = int(model.predict(features_scaled)[0])
    target_names = metadata["target_names"]
    prediction_label = target_names[prediction] if prediction < len(target_names) else str(prediction)

    # Probability
    probability = 0.5
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_scaled)[0]
        probability = float(proba[1])  # probability of positive class
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(features_scaled)[0]
        probability = float(1 / (1 + np.exp(-decision)))  # sigmoid

    # Explanations
    from src.evaluate import explain_prediction
    feature_descs = metadata.get("feature_descriptions", {})
    explanations = explain_prediction(
        model, features_scaled.flatten(), feature_names, feature_descs
    )

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "probability": probability,
        "model_name": bundle["best_model_name"],
        "explanations": explanations,
    }


def predict_from_dict(
    dataset_name: str,
    patient_data: Dict[str, float],
) -> Dict[str, Any]:
    """
    Make a prediction from a dictionary of feature values.

    Parameters
    ----------
    dataset_name : str
    patient_data : dict
        {feature_name: value}

    Returns
    -------
    dict
        Same as predict_from_array.
    """
    bundle = load_model(dataset_name)
    feature_names = bundle["feature_names"]

    # Build feature array in correct order
    features = np.array([
        patient_data.get(name, 0.0) for name in feature_names
    ], dtype=np.float64)

    return predict_from_array(dataset_name, features)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Disease Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Heart Disease prediction
  python -m src.predict --dataset heart_disease --features "63,1,3,145,233,1,0,150,0,2.3,0,0,1"

  # Diabetes prediction
  python -m src.predict --dataset diabetes --features "6,148,72,35,0,33.6,0.627,50"

  # List expected features
  python -m src.predict --dataset heart_disease --info
        """,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["heart_disease", "diabetes", "breast_cancer"],
        help="Dataset/disease type to predict",
    )
    parser.add_argument(
        "--features",
        type=str,
        help="Comma-separated feature values",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show feature information for the dataset",
    )
    return parser.parse_args()


def main():
    """CLI entry point for predictions."""
    logging.basicConfig(level=logging.WARNING)
    args = _parse_args()

    ds_info = DATASET_FEATURES.get(args.dataset)

    if args.info:
        print(f"\n  Dataset: {args.dataset}")
        print(f"  Expected features ({len(ds_info['names'])}):\n")
        for i, name in enumerate(ds_info["names"]):
            desc = ds_info["descriptions"].get(name, "")
            print(f"    {i+1:>2}. {name:<30} {desc}")
        print(f"\n  Example: --features \"{ds_info['example']}\"")
        return

    if not args.features:
        print("Error: --features is required. Use --info to see expected features.")
        sys.exit(1)

    # Parse features
    try:
        values = [float(x.strip()) for x in args.features.split(",")]
    except ValueError as e:
        print(f"Error parsing features: {e}")
        sys.exit(1)

    expected_n = len(ds_info["names"])
    if len(values) != expected_n:
        print(f"Error: Expected {expected_n} features, got {len(values)}")
        print(f"Use --info to see the expected features.")
        sys.exit(1)

    features = np.array(values)

    # Predict
    try:
        result = predict_from_array(args.dataset, features)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Display results
    print(f"\n{'='*60}")
    print(f"  Disease Prediction Result")
    print(f"{'='*60}")
    print(f"  Dataset:     {args.dataset.replace('_', ' ').title()}")
    print(f"  Model:       {result['model_name']}")
    print(f"  Prediction:  {result['prediction_label']}")
    print(f"  Probability: {result['probability']:.1%}")
    print(f"\n  Top Contributing Factors:")
    print(f"  {'-'*50}")

    for exp in result["explanations"]:
        icon = "↑" if exp["direction"] == "increases" else "↓"
        print(f"    {icon} {exp['explanation']}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
