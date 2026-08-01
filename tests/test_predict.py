"""
test_predict.py — Unit tests for the prediction module.

Tests model loading, prediction output format, probability ranges,
and prediction consistency.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.predict import (
    predict_from_array,
    predict_from_dict,
    load_model,
    DATASET_FEATURES,
    MODELS_DIR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model_bundle():
    """Create a mock model bundle matching the saved format."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Create and fit a simple model
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    scaler = StandardScaler()
    scaler.fit(X)

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"],
        "metadata": {
            "name": "Test Disease",
            "target_names": ["Healthy", "Disease"],
            "feature_descriptions": {
                "feat_0": "Test feature 0",
                "feat_1": "Test feature 1",
                "feat_2": "Test feature 2",
                "feat_3": "Test feature 3",
                "feat_4": "Test feature 4",
            },
        },
        "best_model_name": "Random Forest (Test)",
        "metrics": {"accuracy": 0.9, "f1": 0.88, "roc_auc": 0.92},
    }


@pytest.fixture
def mock_load_model(mock_model_bundle):
    """Patch load_model to return the mock bundle."""
    with patch("src.predict.load_model", return_value=mock_model_bundle):
        yield mock_model_bundle


# ---------------------------------------------------------------------------
# Tests: DATASET_FEATURES configuration
# ---------------------------------------------------------------------------

class TestDatasetFeatures:
    def test_all_datasets_have_config(self):
        """All three datasets should have feature configurations."""
        expected = {"heart_disease", "diabetes", "breast_cancer"}
        assert set(DATASET_FEATURES.keys()) == expected

    def test_heart_disease_features_count(self):
        """Heart disease should have 13 features."""
        assert len(DATASET_FEATURES["heart_disease"]["names"]) == 13

    def test_diabetes_features_count(self):
        """Diabetes should have 8 features."""
        assert len(DATASET_FEATURES["diabetes"]["names"]) == 8

    def test_breast_cancer_features_count(self):
        """Breast cancer should have 30 features."""
        assert len(DATASET_FEATURES["breast_cancer"]["names"]) == 30

    def test_each_dataset_has_example(self):
        """Each dataset config should include an example input string."""
        for name, config in DATASET_FEATURES.items():
            assert "example" in config, f"{name} missing example"
            values = config["example"].split(",")
            assert len(values) == len(config["names"]), (
                f"{name}: example has {len(values)} values but {len(config['names'])} features expected"
            )


# ---------------------------------------------------------------------------
# Tests: predict_from_array
# ---------------------------------------------------------------------------

class TestPredictFromArray:
    def test_returns_expected_keys(self, mock_load_model):
        """Prediction result should contain all expected keys."""
        features = np.array([0.5, -0.3, 1.2, 0.1, -0.8])
        result = predict_from_array("test_dataset", features)

        expected_keys = {
            "prediction", "prediction_label", "probability",
            "model_name", "explanations",
        }
        assert set(result.keys()) == expected_keys

    def test_prediction_is_binary(self, mock_load_model):
        """Prediction should be 0 or 1."""
        features = np.array([0.5, -0.3, 1.2, 0.1, -0.8])
        result = predict_from_array("test_dataset", features)
        assert result["prediction"] in [0, 1]

    def test_probability_in_range(self, mock_load_model):
        """Probability should be between 0 and 1."""
        features = np.array([0.5, -0.3, 1.2, 0.1, -0.8])
        result = predict_from_array("test_dataset", features)
        assert 0.0 <= result["probability"] <= 1.0

    def test_prediction_label_is_string(self, mock_load_model):
        """Prediction label should be a human-readable string."""
        features = np.array([0.5, -0.3, 1.2, 0.1, -0.8])
        result = predict_from_array("test_dataset", features)
        assert isinstance(result["prediction_label"], str)
        assert result["prediction_label"] in ["Healthy", "Disease"]

    def test_explanations_returned(self, mock_load_model):
        """Explanations should be returned as a list."""
        features = np.array([0.5, -0.3, 1.2, 0.1, -0.8])
        result = predict_from_array("test_dataset", features)
        assert isinstance(result["explanations"], list)
        assert len(result["explanations"]) > 0

    def test_explanation_has_required_fields(self, mock_load_model):
        """Each explanation should have the required fields."""
        features = np.array([0.5, -0.3, 1.2, 0.1, -0.8])
        result = predict_from_array("test_dataset", features)

        required_fields = {"feature", "value", "importance", "direction", "description", "explanation"}
        for exp in result["explanations"]:
            assert required_fields.issubset(set(exp.keys()))

    def test_explanation_direction_valid(self, mock_load_model):
        """Direction should be 'increases' or 'decreases'."""
        features = np.array([0.5, -0.3, 1.2, 0.1, -0.8])
        result = predict_from_array("test_dataset", features)

        for exp in result["explanations"]:
            assert exp["direction"] in ["increases", "decreases"]


# ---------------------------------------------------------------------------
# Tests: predict_from_dict
# ---------------------------------------------------------------------------

class TestPredictFromDict:
    def test_dict_prediction_matches_array(self, mock_load_model):
        """Dict-based prediction should match array-based prediction."""
        bundle = mock_load_model
        feature_names = bundle["feature_names"]
        values = [0.5, -0.3, 1.2, 0.1, -0.8]

        patient_data = dict(zip(feature_names, values))
        result_dict = predict_from_dict("test_dataset", patient_data)
        result_array = predict_from_array("test_dataset", np.array(values))

        assert result_dict["prediction"] == result_array["prediction"]
        assert abs(result_dict["probability"] - result_array["probability"]) < 1e-10

    def test_missing_features_default_to_zero(self, mock_load_model):
        """Missing features in dict should default to 0.0."""
        partial_data = {"feat_0": 1.5, "feat_1": -0.5}
        result = predict_from_dict("test_dataset", partial_data)
        assert result["prediction"] in [0, 1]


# ---------------------------------------------------------------------------
# Tests: load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_raises_on_missing_model(self):
        """Should raise FileNotFoundError if model doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_dataset")

    def test_models_dir_is_correct(self):
        """MODELS_DIR should point to the models/ directory."""
        assert MODELS_DIR.name == "models"
