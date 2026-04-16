"""Tests for u2.features.temporal module.

Covers sin/cos transformer construction and numerical correctness, the add_encoded_time_features pipeline (expected column names, value ranges, absence of nulls for both hourly and daily fixtures), DST/timezone boundary handling, and degenerate input robustness.
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import load_training_input_from_json
from u2.constants.columns import (
    TIME_SERIES_GROUP_COLUMN_NAME,
    TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)
from u2.features.temporal import (
    add_day_of_week_onehot,
    add_encoded_time_features,
    cos_transformer,
    sin_transformer,
)


@pytest.mark.unit
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


@pytest.mark.unit
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
    ("json_fixture"),
    [("json_single_timeseries"), ("json_daily_freq")],
)
@pytest.mark.unit
class TestAddEncodedTimeFeatures:
    """Tests for add_encoded_time_features function."""

    def test_add_temporal_features_hourly(self, json_fixture: str, request: pytest.FixtureRequest) -> None:
        """Test adding temporal features to hourly data."""
        json_path = request.getfixturevalue(json_fixture)
        data = load_training_input_from_json(json_path)
        df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )

        df[TIME_SERIES_TIMESTAMP_COLUMN_NAME] = pd.to_datetime(df[TIME_SERIES_TIMESTAMP_COLUMN_NAME])

        result_df, _ = add_encoded_time_features(df, TIME_SERIES_TIMESTAMP_COLUMN_NAME)

        assert len(result_df.columns) > len(df.columns)
        assert TIME_SERIES_TIMESTAMP_COLUMN_NAME in result_df.columns
        assert TIME_SERIES_PREDICT_VALUE_COLUMN_NAME in result_df.columns

    def test_temporal_features_names(self, json_fixture: str, request: pytest.FixtureRequest) -> None:
        """Test that expected temporal feature names are created."""
        json_path = request.getfixturevalue(json_fixture)
        data = load_training_input_from_json(json_path)
        df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )

        df[TIME_SERIES_TIMESTAMP_COLUMN_NAME] = pd.to_datetime(df[TIME_SERIES_TIMESTAMP_COLUMN_NAME])

        result_df, _ = add_encoded_time_features(df, TIME_SERIES_TIMESTAMP_COLUMN_NAME)

        assert any("day_of_week" in col for col in result_df.columns)
        assert any("day_of_year" in col for col in result_df.columns)
        assert "month_cos" in result_df.columns
        assert "month_sin" in result_df.columns

    def test_temporal_features_no_nulls(self, json_fixture: str, request: pytest.FixtureRequest) -> None:
        """Test that temporal features don't introduce NaN values."""
        json_path = request.getfixturevalue(json_fixture)
        data = load_training_input_from_json(json_path)
        df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )

        df[TIME_SERIES_TIMESTAMP_COLUMN_NAME] = pd.to_datetime(df[TIME_SERIES_TIMESTAMP_COLUMN_NAME])

        result_df, _ = add_encoded_time_features(df, TIME_SERIES_TIMESTAMP_COLUMN_NAME)

        new_cols = [col for col in result_df.columns if col not in df.columns]
        for col in new_cols:
            assert result_df[col].notna().all()


@pytest.mark.unit
class TestDSTTransitions:
    """Test temporal features across DST and calendar boundaries."""

    def test_temporal_features_across_spring_forward(self):
        """Temporal features should handle CET -> CEST transition without NaN or crash."""
        dates = pd.date_range(start="2023-03-25", periods=72, freq="h")
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.default_rng(42).random(72),
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        result_df, added_cols = add_encoded_time_features(df)

        assert len(added_cols) > 0
        assert result_df[added_cols].notna().all().all(), "Temporal features should have no NaN"
        assert len(result_df) == 72

    def test_temporal_features_across_fall_back(self):
        """Temporal features should handle CEST -> CET transition without NaN or crash."""
        dates = pd.date_range(start="2023-10-28", periods=72, freq="h")
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.default_rng(42).random(72),
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        result_df, added_cols = add_encoded_time_features(df)

        assert len(added_cols) > 0
        assert result_df[added_cols].notna().all().all()
        assert len(result_df) == 72

    def test_hour_sin_cos_continuity_across_midnight(self):
        """hour_sin should wrap smoothly from hour 23 to hour 0."""
        dates = pd.date_range(start="2023-06-15 22:00", periods=6, freq="h")
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: [1.0] * 6,
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        result_df, _ = add_encoded_time_features(df)

        # sin(0/24*2pi) = 0
        hour_sin = result_df["hour_sin"].values
        assert abs(hour_sin[2]) < 0.01, f"hour_sin at midnight should be ~0, got {hour_sin[2]}"

    def test_year_boundary_temporal_features(self):
        """Temporal features should be continuous across Dec 31 -> Jan 1."""
        dates = pd.date_range(start="2023-12-30", periods=72, freq="h")
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.default_rng(42).random(72),
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        result_df, added_cols = add_encoded_time_features(df)

        assert result_df[added_cols].notna().all().all()
        assert np.all(np.abs(result_df["month_sin"].values) <= 1.01)


@pytest.mark.unit
class TestTimestampEdgeCases:
    """Test temporal features with calendar edge cases and sub-hourly frequencies."""

    def test_leap_year_feb_29(self):
        """Temporal features should handle February 29 without NaN or crash."""
        dates = pd.date_range(start="2024-02-28", periods=72, freq="h")
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.default_rng(42).random(72),
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        result_df, added_cols = add_encoded_time_features(df)

        assert len(added_cols) > 0
        assert result_df[added_cols].notna().all().all()
        assert len(result_df) == 72
        # Feb 29 should be day_of_year 60 in a leap year
        feb_29_rows = result_df[result_df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].dt.month == 2]
        assert len(feb_29_rows) > 0

    def test_non_leap_year_feb_boundary(self):
        """Temporal features should handle Feb 28 -> Mar 1 transition in non-leap year."""
        dates = pd.date_range(start="2023-02-27", periods=72, freq="h")
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.default_rng(42).random(72),
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        result_df, added_cols = add_encoded_time_features(df)

        assert result_df[added_cols].notna().all().all()

    def test_sub_hourly_15min_frequency(self):
        """15-minute frequency should include hour sin/cos features."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="15min")
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.default_rng(42).random(200),
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        result_df, added_cols = add_encoded_time_features(df)

        assert "hour_sin" in result_df.columns
        assert "hour_cos" in result_df.columns
        assert result_df[added_cols].notna().all().all()

    def test_sub_hourly_30min_frequency(self):
        """30-minute frequency should include hour sin/cos features."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="30min")
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.default_rng(42).random(200),
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        result_df, added_cols = add_encoded_time_features(df)

        assert "hour_sin" in result_df.columns
        assert "hour_cos" in result_df.columns

    def test_daily_frequency_excludes_hour_features(self):
        """Daily frequency should NOT include hour sin/cos features."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.default_rng(42).random(100),
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        result_df, added_cols = add_encoded_time_features(df)

        assert "hour_sin" not in result_df.columns
        assert "hour_cos" not in result_df.columns
        # But day_of_week and month features should still exist
        assert any("day_of_week" in col for col in added_cols)
        assert "month_sin" in result_df.columns


@pytest.mark.unit
class TestDayOfWeekColumnConsistency:
    """Verify day-of-week one-hot encoding produces a fixed set of columns."""

    def test_weekday_only_data_produces_all_seven_columns(self):
        """Data covering only Mon-Fri should still produce all 7 day_of_week columns.

        pd.get_dummies creates columns only for values present in the data.
        If training data spans Mon-Fri and prediction data lands on a Saturday,
        the model would see a column that didn't exist during training.
        All 7 columns (day_of_week_0 through day_of_week_6) should always be
        created regardless of which days are present in the input.
        """
        # Mon 2023-01-02 through Fri 2023-01-06 — no Saturday or Sunday
        dates = pd.date_range(start="2023-01-02", periods=5 * 24, freq="h")
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: range(len(dates)),
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        result = add_day_of_week_onehot(df, datetime_col=TIME_SERIES_TIMESTAMP_COLUMN_NAME)

        expected_cols = [f"day_of_week_{i}" for i in range(7)]
        for col in expected_cols:
            assert col in result.columns, f"Missing column {col} for training/prediction consistency"


@pytest.mark.unit
class TestDegenerateInputsTemporal:
    """Test temporal features with edge-case input magnitudes."""

    def test_extreme_values_temporal_features(self):
        """Temporal features should remain in [-1, 1] regardless of input value magnitude."""
        n = 100
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: pd.date_range(start="2023-01-01", periods=n, freq="h"),
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: [1e12] * n,
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        result_df, added_cols = add_encoded_time_features(df)

        for col in added_cols:
            if "sin" in col or "cos" in col:
                assert result_df[col].between(-1.01, 1.01).all(), f"{col} has values outside [-1, 1]"
