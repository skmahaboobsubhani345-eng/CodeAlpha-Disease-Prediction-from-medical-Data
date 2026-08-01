"""
Disease Prediction from Medical Data
=====================================
A production-quality ML project for predicting Heart Disease,
Diabetes, and Breast Cancer from structured patient data.
"""

__version__ = "1.0.0"

DATASET_CONFIGS = {
    "heart_disease": {
        "display_name": "Heart Disease",
        "target_col": "target",
        "uci_id": 45,
        "description": "Predict presence of heart disease from clinical attributes (Cleveland subset).",
    },
    "diabetes": {
        "display_name": "Diabetes",
        "target_col": "Outcome",
        "uci_id": 34,
        "description": "Predict onset of diabetes from diagnostic measurements (Pima Indians).",
    },
    "breast_cancer": {
        "display_name": "Breast Cancer",
        "target_col": "target",
        "uci_id": None,  # loaded from sklearn
        "description": "Predict malignant vs benign tumors from cell nuclei features (Wisconsin Diagnostic).",
    },
}
