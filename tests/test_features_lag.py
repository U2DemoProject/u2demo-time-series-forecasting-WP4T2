"""Tests for u2demo_time_series_forecasting.features.lag module.

Covers lag column creation, rolling-statistic columns, correct numerical values, copy semantics (input DataFrame is not mutated), trimming by max shift, edge cases with empty lag/window lists, time gaps, and degenerate inputs.
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import load_training_input_from_json
from u2demo_time_series_forecasting.constants.columns import (
    TIME_SERIES_GROUP_COLUMN_NAME,
    TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)
from u2demo_time_series_forecasting.features.lag import LagFeatureGenerator


@pytest.mark.unit
class TestLagFeatureGenerator:
    """Tests for LagFeatureGenerator class."""

    def test_initialization(self, json_single_timeseries: str) -> None:
        """Test LagFeatureGenerator initialization."""
        data = load_training_input_from_json(json_single_timeseries)
        df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )
        df[TIME_SERIES_GROUP_COLUMN_NAME] = 0
        generator = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)

        assert generator.df is not None
        assert generator.group_name == TIME_SERIES_GROUP_COLUMN_NAME
        assert generator.added_columns == []

    def test_creates_copy(self, json_single_timeseries: str) -> None:
        """Test that generator creates a copy of the DataFrame."""
        data = load_training_input_from_json(json_single_timeseries)
        df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )
        df[TIME_SERIES_GROUP_COLUMN_NAME] = 0
        original_shape = df.shape

        generator = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)

        assert df.shape == original_shape
        assert generator.df is not df

    def test_create_lag_features_multiple_lags(self, json_single_timeseries: str) -> None:
        """Test creating multiple lag features."""
        data = load_training_input_from_json(json_single_timeseries)
        df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )
        df[TIME_SERIES_GROUP_COLUMN_NAME] = 0
        generator = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)

        lags = [1, 7, 24]
        generator.create_lag_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, lags)

        assert len(generator.added_columns) == len(lags)

    def test_lag_values_correct(self, json_single_timeseries: str) -> None:
        """Test that lag values are computed correctly."""
        data = load_training_input_from_json(json_single_timeseries)
        df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )
        df[TIME_SERIES_GROUP_COLUMN_NAME] = 0

        generator = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)

        lags = [1, 7, 24]
        generator.create_lag_features([TIME_SERIES_PREDICT_VALUE_COLUMN_NAME], lags)
        result_df = generator.df

        for lag in lags:
            assert pd.isna(result_df.iloc[:lag][f"lag_{lag}"]).all()

        for (_, group_result_df), (_, group_df) in zip(
            result_df.groupby(TIME_SERIES_GROUP_COLUMN_NAME),
            df.groupby(TIME_SERIES_GROUP_COLUMN_NAME),
            strict=True,
        ):
            result_g = group_result_df.sort_values(TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)
            g = group_df.sort_values(TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)
            for lag in lags:
                col = f"lag_{lag}"
                expected_lag = g[TIME_SERIES_PREDICT_VALUE_COLUMN_NAME].iloc[:-lag].values
                actual_lag = result_g[col].iloc[lag:].values

                assert (expected_lag == actual_lag).all()

    def test_create_rolling_features(self, json_single_timeseries: str) -> None:
        """Test creating rolling statistical features."""
        data = load_training_input_from_json(json_single_timeseries)
        df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )
        df[TIME_SERIES_GROUP_COLUMN_NAME] = 0
        generator = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)

        stats = ["mean", "std", "max", "min"]
        windows = [7, 24, 48]
        generator.create_rolling_stat_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, windows)

        assert len(generator.added_columns) == len(windows) * len(stats)

    def test_rolling_values_correct(self, json_single_timeseries: str) -> None:
        """Test that rolling statistical values (mean, std, max, min) are computed correctly."""
        data = load_training_input_from_json(json_single_timeseries)
        df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )
        df[TIME_SERIES_GROUP_COLUMN_NAME] = 0

        generator = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)

        windows = [7, 24, 48]
        stats = ["mean", "std", "max", "min"]

        generator.create_rolling_stat_features(
            TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
            windows,
            statistics=stats,
        )
        result_df = generator.df

        for w in windows:
            for stat in stats:
                col = f"roll_{stat}_{w}"
                assert pd.isna(result_df.iloc[: w - 1][col]).all()

        for (_, group_result_df), (_, group_orig_df) in zip(
            result_df.groupby(TIME_SERIES_GROUP_COLUMN_NAME),
            df.groupby(TIME_SERIES_GROUP_COLUMN_NAME),
            strict=True,
        ):
            result_g = group_result_df.sort_values(TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)
            orig_g = group_orig_df.sort_values(TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)

            values = orig_g[TIME_SERIES_PREDICT_VALUE_COLUMN_NAME]

            for w in windows:
                for stat in stats:
                    col = f"roll_{stat}_{w}"
                    expected = values.shift(1).rolling(w).agg(stat)
                    actual = result_g[col].values

                    assert np.allclose(expected, actual, atol=1e-9, equal_nan=True)

    def test_trim_by_max_shift(self, json_single_timeseries: str) -> None:
        """Test trimming by max shift."""
        data = load_training_input_from_json(json_single_timeseries)
        df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )
        df[TIME_SERIES_GROUP_COLUMN_NAME] = 0
        original_len = len(df)

        generator = LagFeatureGenerator(df, "item_id")

        lags = [1, 7, 24]
        generator.create_lag_features([TIME_SERIES_PREDICT_VALUE_COLUMN_NAME], lags)

        result_df, _ = generator.finalize()

        assert len(result_df) == original_len - (max(lags) * len(df[TIME_SERIES_GROUP_COLUMN_NAME].unique()))
        assert result_df.notna().all().all()

    def test_empty_lags_list(self, json_single_timeseries: str) -> None:
        """Test behavior with empty lags list."""
        data = load_training_input_from_json(json_single_timeseries)
        df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )
        df[TIME_SERIES_GROUP_COLUMN_NAME] = 0
        generator = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)

        generator.create_lag_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [])

        result_df = generator.df

        assert result_df is not None

    def test_empty_windows_list(self, json_single_timeseries: str) -> None:
        """Test behavior with empty lags list."""
        data = load_training_input_from_json(json_single_timeseries)
        df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )
        df[TIME_SERIES_GROUP_COLUMN_NAME] = 0
        generator = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)

        generator.create_rolling_stat_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [])

        result_df = generator.df

        assert result_df is not None

    def test_unsupported_statistics_raises(self, json_single_timeseries: str) -> None:
        """Requesting unsupported statistics should raise ValueError."""
        data = load_training_input_from_json(json_single_timeseries)
        df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )
        df[TIME_SERIES_GROUP_COLUMN_NAME] = 0
        generator = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)

        with pytest.raises(ValueError, match="Unsupported statistics"):
            generator.create_rolling_stat_features(
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, windows=[7], statistics=["median"]
            )

    def test_no_cross_group_leakage(self, multi_group_df: pd.DataFrame) -> None:
        """Lag features must not bleed values across groups.

        The first `lag` rows of every group must be NaN regardless of whether
        another group ends with non-NaN values at overlapping timestamps.
        """
        lags = [1, 5, 24]
        generator = LagFeatureGenerator(multi_group_df, TIME_SERIES_GROUP_COLUMN_NAME)
        generator.create_lag_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, lags)
        result_df = generator.df

        for group_id, group_df in result_df.groupby(TIME_SERIES_GROUP_COLUMN_NAME):
            g = group_df.sort_values(TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)
            for lag in lags:
                col = f"lag_{lag}"
                initial_nan = g[col].iloc[:lag]
                assert pd.isna(initial_nan).all(), (
                    f"Group '{group_id}': {col} first {lag} rows must be NaN but got {initial_nan.values}"
                )

    def test_mid_series_nan_propagates_through_lag(self) -> None:
        """A NaN in the value column mid-series must propagate into the lag column.

        lag_1 at position nan_pos+1 must be NaN (it is shifted from the NaN row).
        The row after that must be non-NaN (the NaN does not spread further).
        """
        n = 50
        values = np.ones(n, dtype=float)
        nan_pos = 20
        values[nan_pos] = np.nan

        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: pd.date_range(start="2023-01-01", periods=n, freq="h"),
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: values,
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        generator = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)
        generator.create_lag_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [1])
        result = generator.df.sort_values(TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)

        assert pd.isna(result["lag_1"].iloc[nan_pos + 1]), (
            f"lag_1 at position {nan_pos + 1} should be NaN (shifted from NaN at {nan_pos})"
        )
        assert not pd.isna(result["lag_1"].iloc[nan_pos + 2]), (
            f"lag_1 at position {nan_pos + 2} should not be NaN (position {nan_pos + 1} has value 1.0)"
        )

    def test_mid_series_nan_propagates_through_rolling(self) -> None:
        """A NaN mid-series must contaminate every rolling window that spans it.

        With shift(1) + rolling(window=7): the NaN at nan_pos propagates into the
        shifted series at nan_pos+1, making roll_mean_7 NaN for exactly `window`
        consecutive rows before clearing.
        """
        n = 100
        window = 7
        nan_pos = 30
        values = np.ones(n, dtype=float)
        values[nan_pos] = np.nan

        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: pd.date_range(start="2023-01-01", periods=n, freq="h"),
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: values,
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        generator = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)
        generator.create_rolling_stat_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [window])
        result = generator.df.sort_values(TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)

        col = f"roll_mean_{window}"
        # shift(1) places the NaN at index nan_pos+1 in the shifted series.
        # rolling(window) at position p covers shifted[p-window+1 : p+1].
        # The NaN at nan_pos+1 contaminates positions nan_pos+1 through nan_pos+window.
        for pos in range(nan_pos + 1, nan_pos + window + 1):
            assert pd.isna(result[col].iloc[pos]), f"{col}[{pos}] should be NaN (NaN window)"
        assert not pd.isna(result[col].iloc[nan_pos + window + 1]), (
            f"{col}[{nan_pos + window + 1}] should not be NaN (window has cleared the NaN)"
        )

    def test_lag_features_across_gap_use_positional_shift(self) -> None:
        """With a time gap, lag_1 shifts by position, not by time -- expected behavior."""
        n = 50
        dates = pd.date_range(start="2023-01-01", periods=n, freq="h")
        gap_indices = [20, 21, 22, 23, 24]
        dates_with_gap = dates.delete(gap_indices)
        values = np.arange(len(dates_with_gap), dtype=float)

        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates_with_gap,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: values,
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        gen = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)
        gen.create_lag_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [1])
        result = gen.df.sort_values(TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)

        assert result["lag_1"].iloc[20] == result[TIME_SERIES_PREDICT_VALUE_COLUMN_NAME].iloc[19]

    def test_rolling_stats_across_gap(self) -> None:
        """Rolling stats span the gap positionally -- the window includes pre-gap values."""
        n = 50
        dates = pd.date_range(start="2023-01-01", periods=n, freq="h")
        gap_indices = [20, 21, 22, 23, 24]
        dates_with_gap = dates.delete(gap_indices)
        values = np.ones(len(dates_with_gap), dtype=float)

        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates_with_gap,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: values,
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        gen = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)
        gen.create_rolling_stat_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [7])
        result = gen.df.sort_values(TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)

        post_warmup = result["roll_mean_7"].iloc[10:]
        valid = post_warmup.dropna()
        assert np.allclose(valid, 1.0)

    def test_constant_values_lag_features(self) -> None:
        """Constant values should produce valid lag and rolling features (std=0)."""
        n = 100
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: pd.date_range(start="2023-01-01", periods=n, freq="h"),
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: [42.0] * n,
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        gen = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)
        gen.create_lag_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [1, 24])
        gen.create_rolling_stat_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [7, 24])
        result, _ = gen.finalize()

        assert (result["lag_1"] == 42.0).all()
        assert (result["lag_24"] == 42.0).all()
        assert np.allclose(result["roll_mean_7"], 42.0)
        assert np.allclose(result["roll_std_7"], 0.0)

    def test_all_zero_values(self) -> None:
        """All-zero input should not cause division-by-zero or NaN in features."""
        n = 100
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: pd.date_range(start="2023-01-01", periods=n, freq="h"),
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: [0.0] * n,
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        gen = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)
        gen.create_lag_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [1])
        gen.create_rolling_stat_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [7])
        result, _ = gen.finalize()

        assert result["lag_1"].notna().all()
        assert (result["lag_1"] == 0.0).all()
        assert np.allclose(result["roll_mean_7"], 0.0)
        assert np.allclose(result["roll_std_7"], 0.0)

    def test_extreme_values_lag_features(self) -> None:
        """Lag features with extreme values should produce finite results."""
        n = 100
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: pd.date_range(start="2023-01-01", periods=n, freq="h"),
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: [1e12] * n,
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        gen = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)
        gen.create_lag_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [1, 24])
        gen.create_rolling_stat_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [7])
        result, _ = gen.finalize()

        assert np.isfinite(result["lag_1"].values).all()
        assert np.isfinite(result["roll_mean_7"].values).all()
        assert np.allclose(result["roll_std_7"], 0.0)

    def test_negative_values(self) -> None:
        """Negative values should be handled correctly by lag features."""
        n = 100
        values = np.linspace(-100, -1, n)
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: pd.date_range(start="2023-01-01", periods=n, freq="h"),
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: values,
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        gen = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)
        gen.create_lag_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [1])
        gen.create_rolling_stat_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [7])
        result, _ = gen.finalize()

        assert result["lag_1"].notna().all()
        assert (result["roll_mean_7"] < 0).all(), "Mean of negative values should be negative"

    def test_multi_group_no_leakage_after_finalize(self, multi_group_df: pd.DataFrame) -> None:
        """After finalize(), no NaN should remain and groups should be independent."""
        lags = [1, 24]
        windows = [12]

        gen = LagFeatureGenerator(multi_group_df, TIME_SERIES_GROUP_COLUMN_NAME)
        gen.create_lag_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, lags)
        gen.create_rolling_stat_features(TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, windows)
        result, _ = gen.finalize()

        assert result.notna().all().all(), "Finalized multi-group lag features should have no NaN"
        assert set(result[TIME_SERIES_GROUP_COLUMN_NAME].unique()) == {"group_a", "group_b"}

    def test_empty_statistics_list_creates_no_features(self) -> None:
        """Passing statistics=[] should produce zero rolling features.

        An explicit empty list signals "no statistics wanted" and should be
        distinguished from None (which means "use all defaults").
        """
        n = 100
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: pd.date_range(start="2023-01-01", periods=n, freq="h"),
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: range(n),
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )

        gen = LagFeatureGenerator(df, TIME_SERIES_GROUP_COLUMN_NAME)
        gen.create_rolling_stat_features(
            TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
            windows=[24],
            statistics=[],
        )

        assert len(gen.added_columns) == 0
