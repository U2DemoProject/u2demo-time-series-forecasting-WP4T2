"""Tests for u2.dataset module."""

import contextlib
import datetime as dt
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tests.conftest import load_training_input_from_json
from u2.dataset import U2TimeSeriesPredictionDataset
from u2.defaults import (
    TIME_SERIES_GROUP_COLUMN_NAME,
    TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)


class TestU2TimeSeriesPredictionDataset:
    """Tests for U2TimeSeriesPredictionDataset class."""

    @pytest.mark.parametrize(
        ("json_fixture", "expectation"),
        [
            ("json_single_timeseries", does_not_raise()),
            ("json_daily_freq", does_not_raise()),
            ("json_with_nan_values", pytest.raises(ValidationError)),
            ("json_short_periods", does_not_raise()),
            ("json_empty", pytest.raises(ValidationError)),
        ],
    )
    def test_from_json_all_fixtures(
        self,
        json_fixture: str,
        expectation: AbstractContextManager,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test loading dataset from all JSON fixtures with expected outcomes."""
        json_path = request.getfixturevalue(json_fixture)

        with expectation:
            data = load_training_input_from_json(json_path)
            dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

            assert dataset.df is not None
            assert TIME_SERIES_TIMESTAMP_COLUMN_NAME in dataset.df.columns
            assert TIME_SERIES_PREDICT_VALUE_COLUMN_NAME in dataset.df.columns
            assert TIME_SERIES_GROUP_COLUMN_NAME in dataset.df.columns

    @pytest.mark.parametrize(
        ("json_fixture", "min_timeseries_length", "expectation"),
        [
            ("json_single_timeseries", 98, does_not_raise()),
            (
                "json_single_timeseries",
                170,
                pytest.raises(ValueError, match=r"No timeseries in the input has enough rows to predict/train on*"),
            ),
            (
                "json_short_periods",
                4,
                does_not_raise(),
            ),
            (
                "json_short_periods",
                6,
                pytest.raises(ValueError, match=r"No timeseries in the input has enough rows to predict/train on*"),
            ),
        ],
    )
    def test_drop_short_timeseries(
        self,
        json_fixture: str,
        min_timeseries_length: int,
        expectation: AbstractContextManager,
        request: pytest.FixtureRequest,
    ):
        json_path = request.getfixturevalue(json_fixture)

        with expectation:
            data = load_training_input_from_json(json_path)
            dataset = U2TimeSeriesPredictionDataset.from_training_input(data)
            dataset.drop_short_timeseries(min_timeseries_length)

    @pytest.mark.parametrize(("horizon"), [1, 24, 72])
    def test_extend_by_horizon(self, json_single_timeseries: str, horizon: int) -> None:
        """Test extending dataset by horizon."""
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

        pre_len = dataset.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()

        dataset.extend_by_horizon(horizon)

        post_len = dataset.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()

        assert ((post_len - pre_len) == horizon).all()

    @pytest.mark.parametrize(("horizon"), [1, 24, 72])
    def test_split_past_and_future(self, json_single_timeseries: str, horizon: int) -> None:
        """Test splitting past and future dataframes."""
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

        pre_len = dataset.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()

        dataset.extend_by_horizon(horizon)

        dataset.split_past_and_future_dfs(horizon)

        post_len = dataset.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()

        assert dataset.future_df is not None

        future_len = dataset.future_df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()

        assert (pre_len == post_len).all()
        assert (future_len == horizon).all()

    def test_invalid_json_path(self) -> None:
        """Test error handling for invalid JSON path."""
        with contextlib.suppress(FileNotFoundError, Exception):
            data = load_training_input_from_json("/invalid/path/to/file.json")
            U2TimeSeriesPredictionDataset.from_training_input(data)

    def test_add_temporal(self, json_single_timeseries: str) -> None:
        """Test that temporal features are added and recorded."""
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

        dataset.add_temporal()

        assert "temporal" in dataset.added_columns
        assert len(dataset.added_columns["temporal"]) > 0
        assert any(col.startswith(("month_", "day_of_week_")) for col in dataset.df.columns)

    @pytest.mark.parametrize(("lags", "windows"), [([24, 48, 72], [12, 24]), ([48, 72], [24, 72])])
    def test_add_lags(self, json_single_timeseries: str, lags: list[int], windows: list[int]) -> None:
        """Test that lag and rolling features are added and rows trimmed accordingly."""
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)
        stats = ["mean", "std", "max", "min"]

        max_shift = max(lags + windows)

        pre_len = dataset.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()

        dataset.add_lags(lags=lags, windows=windows)

        post_len = dataset.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()

        assert "lag" in dataset.added_columns

        assert all(f"lag_{lag}" in dataset.added_columns["lag"] for lag in lags)
        assert all(f"lag_{lag}" in dataset.df.columns for lag in lags)

        assert all(f"roll_{stat}_{window}" in dataset.added_columns["lag"] for window in windows for stat in stats)
        assert all(f"roll_{stat}_{window}" in dataset.df.columns for window in windows for stat in stats)

        assert ((pre_len - post_len) == max_shift).all()

        for _, group_df in dataset.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME):
            g = group_df.sort_values(TIME_SERIES_TIMESTAMP_COLUMN_NAME).reset_index(drop=True)
            for lag in lags:
                col = f"lag_{lag}"
                expected_lag = g[TIME_SERIES_PREDICT_VALUE_COLUMN_NAME].shift(lag).iloc[lag:]
                actual_lag = g[col].iloc[lag:]

                assert (expected_lag == actual_lag).all()

            for window in windows:
                for stat in stats:
                    col = f"roll_{stat}_{window}"
                    expected_roll_stat = (
                        g[TIME_SERIES_PREDICT_VALUE_COLUMN_NAME].shift(1).rolling(window=window).agg(stat).iloc[window:]
                    )
                    actual_roll_stat = g[col].iloc[window:]
                    assert np.allclose(expected_roll_stat, actual_roll_stat, atol=1e-9)

    @pytest.mark.parametrize(
        ("num_weekly_lags", "min_lookback", "expectation"),
        [
            (0, 24, does_not_raise()),
            (
                1,
                48,
                pytest.raises(ValueError, match=r"No timeseries in the input has enough rows to predict/train on*"),
            ),
            (
                1,
                72,
                pytest.raises(ValueError, match=r"No timeseries in the input has enough rows to predict/train on*"),
            ),
        ],
    )
    def test_long_lags(
        self,
        json_single_timeseries: str,
        num_weekly_lags: int,
        min_lookback: int,
        expectation: AbstractContextManager,
    ) -> None:
        """Test that lag and rolling features are added and rows trimmed accordingly."""
        data = load_training_input_from_json(json_single_timeseries)
        data.features = ["lag"]
        data.num_weekly_lags = num_weekly_lags
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)
        with expectation:
            dataset.drop_short_timeseries(num_weekly_lags * 7 * 24 + min_lookback)

    def test_add_holidays_monkeypatched(self, json_single_timeseries: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test add_holidays using monkeypatched holidays merger to avoid external calls."""
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

        def fake_merge_holidays(df: pd.DataFrame, location_col: str, datetime_col: str):  # noqa: ANN202, ARG001
            df2 = df.copy()
            df2["holiday_public"] = 0
            df2["holiday_bank"] = 0
            return df2, ["holiday_public", "holiday_bank"]

        monkeypatch.setattr("u2.dataset.merge_holidays_with_dataframe", fake_merge_holidays)

        dataset.add_holidays()

        assert "holiday" in dataset.added_columns
        assert {"holiday_public", "holiday_bank"}.issubset(set(dataset.df.columns))

    def test_add_weather_monkeypatched(self, json_single_timeseries: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test add_weather using monkeypatched weather enricher to avoid external calls."""
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

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

        dataset.final_timestamp = dataset.df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].max()
        dataset.freq = dataset.freq or "h"
        dataset.freq_timedelta = dataset.freq_timedelta or dt.timedelta(hours=1)

        dataset.add_weather()

        assert "weather" in dataset.added_columns
        assert "temperature_2m" in dataset.df.columns

    def test_location_dependent_features_require_location(self, json_without_location: str):
        data = load_training_input_from_json(json_without_location)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

        with pytest.raises(
            AttributeError,
            match="Location information is required to add `holiday` features but is missing in the dataset.",
        ):
            dataset.add_holidays()

        with pytest.raises(
            AttributeError,
            match="Location information is required to add `weather` features but is missing in the dataset.",
        ):
            dataset.add_weather()
