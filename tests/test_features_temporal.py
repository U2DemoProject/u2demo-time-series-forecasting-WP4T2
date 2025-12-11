"""Tests for u2.features.temporal module."""

import numpy as np
import pandas as pd
import pytest

from u2.defaults import DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME
from u2.features.temporal import add_encoded_time_features, cos_transformer, sin_transformer


class TestSinTransformer:
    """Tests for sin_transformer function."""

    def test_sin_transformer_creation(self) -> None:
        """Test that sin_transformer creates a valid transformer."""
        transformer = sin_transformer(24)
        assert transformer is not None
        assert hasattr(transformer, "fit_transform")

    def test_sin_transformer_period_24(self) -> None:
        """Test sin transformation with period 24 (hourly)."""
        transformer = sin_transformer(24)
        data = np.array([[0], [6], [12], [18]]).astype(float)
        result = transformer.fit_transform(data)

        # At 0 hours, sin(0) = 0
        assert np.isclose(result[0, 0], 0, atol=0.01)
        # At 12 hours (half period), sin(pi) = 0
        assert np.isclose(result[2, 0], 0, atol=0.01)

    def test_sin_transformer_period_365(self) -> None:
        """Test sin transformation with period 365 (daily)."""
        transformer = sin_transformer(365)
        data = np.array([[0], [365 // 4], [365 // 2]]).astype(float)
        result = transformer.fit_transform(data)

        # Values should be between -1 and 1
        assert np.all(result >= -1.01)
        assert np.all(result <= 1.01)


class TestCosTransformer:
    """Tests for cos_transformer function."""

    def test_cos_transformer_creation(self) -> None:
        """Test that cos_transformer creates a valid transformer."""
        transformer = cos_transformer(24)
        assert transformer is not None
        assert hasattr(transformer, "fit_transform")

    def test_cos_transformer_period_24(self) -> None:
        """Test cos transformation with period 24 (hourly)."""
        transformer = cos_transformer(24)
        data = np.array([[0], [6], [12], [18]]).astype(float)
        result = transformer.fit_transform(data)

        # At 0 hours, cos(0) = 1
        assert np.isclose(result[0, 0], 1, atol=0.01)
        # At 12 hours (half period), cos(pi) = -1
        assert np.isclose(result[2, 0], -1, atol=0.01)

    def test_cos_transformer_range(self) -> None:
        """Test that cos transformation output is in [-1, 1]."""
        transformer = cos_transformer(12)
        data = np.array([[i] for i in range(100)]).astype(float)
        result = transformer.fit_transform(data)

        assert np.all(result >= -1.01)
        assert np.all(result <= 1.01)


@pytest.mark.parametrize(
    ("csv_fixture"),
    [("csv_multiple_groups"), ("csv_single_group"), ("csv_daily_freq"), ("csv_same_end_date_different_length")],
)
class TestAddEncodedTimeFeatures:
    """Tests for add_encoded_time_features function."""

    def test_add_temporal_features_hourly(self, csv_fixture: str, request: pytest.FixtureRequest) -> None:
        """Test adding temporal features to hourly data."""
        csv_path = request.getfixturevalue(csv_fixture)
        df = pd.read_csv(csv_path)
        df[DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME] = pd.to_datetime(df[DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME])

        result_df, _ = add_encoded_time_features(df, DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME)

        assert len(result_df.columns) > len(df.columns)
        assert DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME in result_df.columns
        assert DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME in result_df.columns

    def test_temporal_features_names(self, csv_fixture: str, request: pytest.FixtureRequest) -> None:
        """Test that expected temporal feature names are created."""
        csv_path = request.getfixturevalue(csv_fixture)
        df = pd.read_csv(csv_path)
        df[DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME] = pd.to_datetime(df[DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME])

        result_df, _ = add_encoded_time_features(df, DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME)

        assert any("day_of_week" in col for col in result_df.columns)
        assert any("day_of_year" in col for col in result_df.columns)
        assert "month_cos" in result_df.columns
        assert "month_sin" in result_df.columns

    def test_temporal_features_no_nulls(self, csv_fixture: str, request: pytest.FixtureRequest) -> None:
        """Test that temporal features don't introduce NaN values."""
        csv_path = request.getfixturevalue(csv_fixture)
        df = pd.read_csv(csv_path)
        df[DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME] = pd.to_datetime(df[DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME])

        result_df, _ = add_encoded_time_features(df, DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME)

        new_cols = [col for col in result_df.columns if col not in df.columns]
        for col in new_cols:
            assert result_df[col].notna().all()
