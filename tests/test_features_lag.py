"""Tests for u2.features.lag module."""

import numpy as np
import pandas as pd

from u2.defaults import (
    DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME,
    DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)
from u2.features.lag import LagFeatureGenerator


class TestLagFeatureGenerator:
    """Tests for LagFeatureGenerator class."""

    def test_initialization(self, csv_multiple_groups: str) -> None:
        """Test LagFeatureGenerator initialization."""
        df = pd.read_csv(csv_multiple_groups)
        generator = LagFeatureGenerator(df, DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME)

        assert generator.df is not None
        assert generator.group_name == DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME
        assert generator.added_columns == []

    def test_creates_copy(self, csv_multiple_groups: str) -> None:
        """Test that generator creates a copy of the DataFrame."""
        df = pd.read_csv(csv_multiple_groups)
        original_shape = df.shape

        generator = LagFeatureGenerator(df, DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME)

        assert df.shape == original_shape
        assert generator.df is not df

    def test_create_lag_features_multiple_lags(self, csv_multiple_groups: str) -> None:
        """Test creating multiple lag features."""
        df = pd.read_csv(csv_multiple_groups)
        generator = LagFeatureGenerator(df, DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME)

        lags = [1, 7, 24]
        generator.create_lag_features(DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, lags)

        assert len(generator.added_columns) == len(lags)

    def test_lag_values_correct(self, csv_multiple_groups: str) -> None:
        """Test that lag values are computed correctly."""
        df = pd.read_csv(csv_multiple_groups)

        generator = LagFeatureGenerator(df, DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME)

        lags = [1, 7, 24]
        generator.create_lag_features([DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME], lags)
        result_df = generator.df

        for lag in lags:
            assert pd.isna(result_df.iloc[:lag][f"lag_{lag}"]).all()

        for (_, group_result_df), (_, group_df) in zip(
            result_df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME),
            df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME),
            strict=True,
        ):
            result_g = group_result_df.sort_values(DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)
            g = group_df.sort_values(DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)
            for lag in lags:
                col = f"lag_{lag}"
                expected_lag = g[DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME].iloc[:-lag].values
                actual_lag = result_g[col].iloc[lag:].values

                assert (expected_lag == actual_lag).all()

    def test_create_rolling_features(self, csv_multiple_groups: str) -> None:
        """Test creating rolling statistical features."""
        df = pd.read_csv(csv_multiple_groups)
        generator = LagFeatureGenerator(df, DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME)

        stats = ["mean", "std", "max", "min"]
        windows = [7, 24, 48]
        generator.create_rolling_stat_features(DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, windows)

        assert len(generator.added_columns) == len(windows) * len(stats)

    def test_rolling_values_correct(self, csv_multiple_groups: str) -> None:
        """Test that rolling statistical values (mean, std, max, min) are computed correctly."""
        df = pd.read_csv(csv_multiple_groups)

        generator = LagFeatureGenerator(df, DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME)

        windows = [7, 24, 48]
        stats = ["mean", "std", "max", "min"]

        generator.create_rolling_stat_features(
            DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
            windows,
            statistics=stats,
        )
        result_df = generator.df

        for w in windows:
            for stat in stats:
                col = f"roll_{stat}_{w}"
                assert pd.isna(result_df.iloc[: w - 1][col]).all()

        for (_, group_result_df), (_, group_orig_df) in zip(
            result_df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME),
            df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME),
            strict=True,
        ):
            result_g = group_result_df.sort_values(DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)
            orig_g = group_orig_df.sort_values(DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)

            values = orig_g[DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME]

            for w in windows:
                for stat in stats:
                    col = f"roll_{stat}_{w}"
                    expected = values.shift(1).rolling(w).agg(stat)
                    actual = result_g[col].values

                    assert np.allclose(expected, actual, atol=1e-9, equal_nan=True)

    def test_trim_by_max_shift(self, csv_multiple_groups: str) -> None:
        """Test trimming by max shift."""
        df = pd.read_csv(csv_multiple_groups)
        original_len = len(df)

        generator = LagFeatureGenerator(df, "item_id")

        lags = [1, 7, 24]
        generator.create_lag_features([DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME], lags)

        result_df, _ = generator.finalize()

        assert len(result_df) == original_len - (max(lags) * len(df[DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME].unique()))
        assert result_df.notna().all().all()

    def test_empty_lags_list(self, csv_multiple_groups: str) -> None:
        """Test behavior with empty lags list."""
        df = pd.read_csv(csv_multiple_groups)
        generator = LagFeatureGenerator(df, DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME)

        generator.create_lag_features(DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [])

        result_df = generator.df

        assert result_df is not None

    def test_empty_windows_list(self, csv_multiple_groups: str) -> None:
        """Test behavior with empty lags list."""
        df = pd.read_csv(csv_multiple_groups)
        generator = LagFeatureGenerator(df, DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME)

        generator.create_rolling_stat_features(DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME, [])

        result_df = generator.df

        assert result_df is not None
