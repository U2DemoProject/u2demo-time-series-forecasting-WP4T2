"""Tests for u2.dataset module."""

import contextlib
import datetime as dt
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from u2.dataset import U2TimeSeriesPredictionDataset
from u2.defaults import (
    DEFAULT_LOCATION_COLUMN_NAME,
    DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME,
    DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)


class TestU2TimeSeriesPredictionDataset:
    """Tests for U2TimeSeriesPredictionDataset class."""

    @pytest.mark.parametrize(
        ("csv_fixture", "expectation"),
        [
            (
                "csv_multiple_groups_different_end_dates",
                pytest.raises(ValueError, match="Not all groups have the same final timestamp."),
            ),
            ("csv_multiple_groups", does_not_raise()),
            ("csv_single_group", does_not_raise()),
            ("csv_same_end_date_different_length", does_not_raise()),
            ("csv_daily_freq", does_not_raise()),
            ("csv_with_nan_values", pytest.raises(ValueError, match="The input data contains NaN values.")),
            (
                "csv_short_periods",
                pytest.raises(ValueError, match="No group in the input CSV has enough rows to predict on."),
            ),
            ("csv_empty", pytest.raises(ValueError, match="No group in the input CSV has enough rows to predict on.")),
            (
                "csv_other_columns",
                pytest.raises(
                    AttributeError,
                    match=r"One of `.*` and/or `.*` and/or `.*` columns are missing.",
                ),
            ),
        ],
    )
    def test_from_csv_all_fixtures(
        self,
        csv_fixture: str,
        expectation: AbstractContextManager,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test loading dataset from all CSV fixtures with expected outcomes."""
        # Get the actual CSV path from the fixture
        csv_path = request.getfixturevalue(csv_fixture)

        with expectation:
            dataset = U2TimeSeriesPredictionDataset.from_csv(
                csv_path,
                min_num_rows_per_group=10,
            )

            assert dataset.df is not None
            assert DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME in dataset.df.columns
            assert DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME in dataset.df.columns
            assert DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME in dataset.df.columns
            assert DEFAULT_LOCATION_COLUMN_NAME in dataset.df.columns

    def test_from_csv_custom_columns(self, csv_other_columns: str) -> None:
        """Test loading dataset from CSV with custom column names."""
        dataset = U2TimeSeriesPredictionDataset.from_csv(
            csv_other_columns,
            timestamp_col_name="datetime",
            predict_value_col_name="energy",
            group_col_name="asset_name",
            location_col_name="site",
            min_num_rows_per_group=10,
        )

        assert dataset.df is not None
        assert DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME in dataset.df.columns
        assert DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME in dataset.df.columns
        assert DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME in dataset.df.columns
        assert DEFAULT_LOCATION_COLUMN_NAME in dataset.df.columns

    @pytest.mark.parametrize(("horizon"), [1, 24, 72])
    def test_extend_by_horizon(self, csv_multiple_groups: str, horizon: int) -> None:
        """Test extending dataset by horizon."""
        dataset = U2TimeSeriesPredictionDataset.from_csv(
            csv_multiple_groups,
            min_num_rows_per_group=10,
        )

        pre_len = dataset.df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME).size()

        dataset.extend_by_horizon(horizon)

        post_len = dataset.df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME).size()

        assert ((post_len - pre_len) == horizon).all()

    @pytest.mark.parametrize(("horizon"), [1, 24, 72])
    def test_split_past_and_future(self, csv_multiple_groups: str, horizon: int) -> None:
        """Test splitting past and future dataframes."""
        dataset = U2TimeSeriesPredictionDataset.from_csv(
            csv_multiple_groups,
            min_num_rows_per_group=10,
        )

        pre_len = dataset.df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME).size()

        dataset.extend_by_horizon(horizon)

        dataset.split_past_and_future_dfs(horizon)

        post_len = dataset.df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME).size()

        assert dataset.future_df is not None

        future_len = dataset.future_df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME).size()

        assert (pre_len == post_len).all()
        assert (future_len == horizon).all()

    def test_invalid_csv_path(self) -> None:
        """Test error handling for invalid CSV path."""
        with contextlib.suppress(FileNotFoundError, Exception):
            U2TimeSeriesPredictionDataset.from_csv(
                "/invalid/path/to/file.csv",
                min_num_rows_per_group=10,
            )

    def test_add_temporal(self, csv_multiple_groups: str) -> None:
        """Test that temporal features are added and recorded."""
        dataset = U2TimeSeriesPredictionDataset.from_csv(
            csv_multiple_groups,
            min_num_rows_per_group=10,
        )

        dataset.add_temporal()

        assert "temporal" in dataset.added_columns
        assert len(dataset.added_columns["temporal"]) > 0
        assert any(col.startswith(("month_", "day_of_week_")) for col in dataset.df.columns)

    @pytest.mark.parametrize(("lags", "windows"), [([24, 48, 72], [12, 24]), ([48, 72], [24, 72])])
    def test_add_lags(self, csv_multiple_groups: str, lags: list[int], windows: list[int]) -> None:
        """Test that lag and rolling features are added and rows trimmed accordingly."""
        dataset = U2TimeSeriesPredictionDataset.from_csv(
            csv_multiple_groups,
            min_num_rows_per_group=10,
        )
        stats = ["mean", "std", "max", "min"]

        max_shift = max(lags + windows)

        pre_len = dataset.df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME).size()

        dataset.add_lags(lags=lags, windows=windows)

        post_len = dataset.df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME).size()

        assert "lag" in dataset.added_columns

        assert all(f"lag_{lag}" in dataset.added_columns["lag"] for lag in lags)
        assert all(f"lag_{lag}" in dataset.df.columns for lag in lags)

        assert all(f"roll_{stat}_{window}" in dataset.added_columns["lag"] for window in windows for stat in stats)
        assert all(f"roll_{stat}_{window}" in dataset.df.columns for window in windows for stat in stats)

        assert ((pre_len - post_len) == max_shift).all()

        for _, group_df in dataset.df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME):
            g = group_df.sort_values(DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)
            for lag in lags:
                col = f"lag_{lag}"
                expected_lag = g[DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME].shift(lag).iloc[lag:]
                actual_lag = g[col].iloc[lag:]

                assert (expected_lag == actual_lag).all()

            for window in windows:
                for stat in stats:
                    col = f"roll_{stat}_{window}"
                    expected_roll_stat = (
                        g[DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME]
                        .shift(1)
                        .rolling(window=window)
                        .agg(stat)
                        .iloc[window:]
                    )
                    actual_roll_stat = g[col].iloc[window:]
                    assert np.allclose(expected_roll_stat, actual_roll_stat, atol=1e-9)

    @pytest.mark.parametrize(
        ("lags", "windows", "expectation"),
        [
            ([24, 72], [12, 48], does_not_raise()),
            (
                [144],
                [12, 24],
                pytest.raises(ValueError, match="No group in the input CSV has enough rows to predict on."),
            ),
            (
                [48],
                [24, 168],
                pytest.raises(ValueError, match="No group in the input CSV has enough rows to predict on."),
            ),
        ],
    )
    def test_long_lags(
        self, csv_multiple_groups: str, lags: list[int], windows: list[int], expectation: AbstractContextManager
    ) -> None:
        """Test that lag and rolling features are added and rows trimmed accordingly."""
        dataset = U2TimeSeriesPredictionDataset.from_csv(
            csv_multiple_groups,
            min_num_rows_per_group=10,
        )
        with expectation:
            # TODO: set min_num_rows to max(min_num_rows, max(lags+windows))
            dataset.add_lags(lags=lags, windows=windows)

    def test_add_holidays_monkeypatched(self, csv_multiple_groups: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test add_holidays using monkeypatched holidays merger to avoid external calls."""
        dataset = U2TimeSeriesPredictionDataset.from_csv(
            csv_multiple_groups,
            min_num_rows_per_group=10,
        )

        def fake_merge_holidays(df: pd.DataFrame, location_col: str, datetime_col: str):  # noqa: ANN202, ARG001
            df2 = df.copy()
            df2["holiday_public"] = 0
            df2["holiday_bank"] = 0
            return df2, ["holiday_public", "holiday_bank"]

        monkeypatch.setattr("u2.dataset.merge_holidays_for_multiple_places", fake_merge_holidays)

        dataset.add_holidays()

        assert "holiday" in dataset.added_columns
        assert {"holiday_public", "holiday_bank"}.issubset(set(dataset.df.columns))

    def test_add_weather_monkeypatched(self, csv_multiple_groups: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test add_weather using monkeypatched weather enricher to avoid external calls."""
        dataset = U2TimeSeriesPredictionDataset.from_csv(
            csv_multiple_groups,
            min_num_rows_per_group=10,
        )

        def fake_enrich_weather(  # noqa: ANN202
            df: pd.DataFrame,
            time_col: str,  # noqa: ARG001
            location_col: str,  # noqa: ARG001
            present_time: dt.datetime,  # noqa: ARG001
            freq: str,  # noqa: ARG001
        ):
            df2 = df.copy()
            df2["temperature_2m"] = 1.23
            return df2, ["temperature_2m"]

        monkeypatch.setattr("u2.dataset.enrich_weather", fake_enrich_weather)

        dataset.final_timestamp = dataset.df[DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME].max()
        dataset.freq = dataset.freq or "h"
        dataset.freq_timedelta = dataset.freq_timedelta or dt.timedelta(hours=1)

        dataset.add_weather()

        assert "weather" in dataset.added_columns
        assert "temperature_2m" in dataset.df.columns
