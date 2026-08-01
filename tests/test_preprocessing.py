"""
test_preprocessing.py — Unit tests for the preprocessing pipeline.

Tests missing value handling, feature scaling, categorical encoding,
and SMOTE class balancing.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    handle_missing_values,
    encode_categoricals,
    scale_features,
    handle_class_imbalance,
    preprocess_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Create a sample dataframe with known properties."""
    np.random.seed(42)
    return pd.DataFrame({
        "age": [25, 30, np.nan, 45, 50, 60, 35, 40, 55, 28],
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        "bp": [120, 130, 140, np.nan, 160, 110, 125, 135, 145, 155],
        "cholesterol": [200, 250, 220, 180, np.nan, 300, 210, 240, 260, 190],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def diabetes_df():
    """Create a sample diabetes-like dataframe with zero values."""
    return pd.DataFrame({
        "Pregnancies": [6, 1, 8, 1, 0],
        "Glucose": [148, 85, 0, 89, 137],       # 0 is invalid
        "BloodPressure": [72, 66, 0, 66, 40],    # 0 is invalid
        "SkinThickness": [35, 29, 0, 23, 35],    # 0 is invalid
        "Insulin": [0, 0, 0, 94, 168],           # 0 is invalid
        "BMI": [33.6, 26.6, 0, 28.1, 43.1],     # 0 is invalid
        "DiabetesPedigreeFunction": [0.627, 0.351, 0.158, 0.167, 2.288],
        "Age": [50, 31, 32, 21, 33],
        "Outcome": [1, 0, 1, 0, 1],
    })


@pytest.fixture
def numeric_arrays():
    """Create sample numeric arrays for scaling tests."""
    np.random.seed(42)
    X_train = np.random.randn(80, 5)
    X_test = np.random.randn(20, 5)
    return X_train, X_test


# ---------------------------------------------------------------------------
# Tests: handle_missing_values
# ---------------------------------------------------------------------------

class TestHandleMissingValues:
    def test_removes_all_nans(self, sample_df):
        """Missing values should be imputed (no NaNs remaining)."""
        result = handle_missing_values(sample_df)
        assert result.isnull().sum().sum() == 0

    def test_does_not_modify_original(self, sample_df):
        """Original dataframe should not be modified."""
        original_nans = sample_df.isnull().sum().sum()
        handle_missing_values(sample_df)
        assert sample_df.isnull().sum().sum() == original_nans

    def test_preserves_shape(self, sample_df):
        """Shape should be preserved after imputation."""
        result = handle_missing_values(sample_df)
        assert result.shape == sample_df.shape

    def test_diabetes_zeros_replaced(self, diabetes_df):
        """Diabetes-specific: zeros in certain columns should be treated as NaN and imputed."""
        result = handle_missing_values(diabetes_df, dataset_name="diabetes")

        # These columns had zeros that should have been replaced
        for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
            assert (result[col] == 0).sum() == 0, f"Column {col} still has zeros"

    def test_numeric_median_imputation(self, sample_df):
        """Numeric columns should be imputed with median values."""
        result = handle_missing_values(sample_df)
        # age had one NaN, median of others is ~40
        assert not np.isnan(result["age"].iloc[2])


# ---------------------------------------------------------------------------
# Tests: encode_categoricals
# ---------------------------------------------------------------------------

class TestEncodeCategoricals:
    def test_encodes_string_columns(self, sample_df):
        """String columns should be encoded to integers."""
        result, encoders = encode_categoricals(sample_df, "target")
        assert result["sex"].dtype in [np.int32, np.int64, np.intp]
        assert "sex" in encoders

    def test_does_not_encode_target(self, sample_df):
        """Target column should not be encoded."""
        _, encoders = encode_categoricals(sample_df, "target")
        assert "target" not in encoders

    def test_encoder_can_inverse_transform(self, sample_df):
        """Encoders should support inverse transformation."""
        result, encoders = encode_categoricals(sample_df, "target")
        original_values = encoders["sex"].inverse_transform(result["sex"])
        assert set(original_values) == {"F", "M"}


# ---------------------------------------------------------------------------
# Tests: scale_features
# ---------------------------------------------------------------------------

class TestScaleFeatures:
    def test_output_shapes(self, numeric_arrays):
        """Output arrays should have same shape as input."""
        X_train, X_test = numeric_arrays
        X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
        assert X_train_s.shape == X_train.shape
        assert X_test_s.shape == X_test.shape

    def test_train_mean_zero(self, numeric_arrays):
        """Scaled training data should have approximately zero mean."""
        X_train, X_test = numeric_arrays
        X_train_s, _, _ = scale_features(X_train, X_test)
        assert np.abs(X_train_s.mean(axis=0)).max() < 1e-10

    def test_train_unit_variance(self, numeric_arrays):
        """Scaled training data should have approximately unit variance."""
        X_train, X_test = numeric_arrays
        X_train_s, _, _ = scale_features(X_train, X_test)
        assert np.abs(X_train_s.std(axis=0) - 1.0).max() < 0.1

    def test_scaler_returned(self, numeric_arrays):
        """A fitted scaler should be returned."""
        X_train, X_test = numeric_arrays
        _, _, scaler = scale_features(X_train, X_test)
        assert hasattr(scaler, "mean_")
        assert hasattr(scaler, "scale_")


# ---------------------------------------------------------------------------
# Tests: handle_class_imbalance
# ---------------------------------------------------------------------------

class TestHandleClassImbalance:
    def test_smote_balances_classes(self):
        """SMOTE should produce balanced classes."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.array([0] * 80 + [1] * 20)

        X_res, y_res = handle_class_imbalance(X, y, method="smote")

        unique, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1], "Classes should be balanced after SMOTE"

    def test_smote_increases_samples(self):
        """SMOTE should increase the number of samples."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.array([0] * 80 + [1] * 20)

        X_res, y_res = handle_class_imbalance(X, y, method="smote")
        assert len(y_res) > len(y)

    def test_none_method_passthrough(self):
        """method='none' should return data unchanged."""
        X = np.random.randn(50, 3)
        y = np.array([0] * 40 + [1] * 10)

        X_res, y_res = handle_class_imbalance(X, y, method="none")
        np.testing.assert_array_equal(X_res, X)
        np.testing.assert_array_equal(y_res, y)


# ---------------------------------------------------------------------------
# Tests: preprocess_pipeline
# ---------------------------------------------------------------------------

class TestPreprocessPipeline:
    def test_returns_all_keys(self, sample_df):
        """Pipeline should return all expected keys."""
        result = preprocess_pipeline(sample_df, "target")
        expected_keys = {
            "X_train", "X_test", "y_train", "y_test",
            "scaler", "feature_names", "encoders",
        }
        assert set(result.keys()) == expected_keys

    def test_no_nans_in_output(self, sample_df):
        """Output arrays should contain no NaN values."""
        result = preprocess_pipeline(sample_df, "target")
        assert not np.any(np.isnan(result["X_train"]))
        assert not np.any(np.isnan(result["X_test"]))

    def test_target_is_integer(self, sample_df):
        """Target values should be integers."""
        result = preprocess_pipeline(sample_df, "target")
        assert result["y_train"].dtype in [np.int32, np.int64, np.intp]
        assert result["y_test"].dtype in [np.int32, np.int64, np.intp]

    def test_feature_names_match_dimensions(self, sample_df):
        """Feature names count should match X dimensions."""
        result = preprocess_pipeline(sample_df, "target", apply_smote=False)
        assert len(result["feature_names"]) == result["X_train"].shape[1]
