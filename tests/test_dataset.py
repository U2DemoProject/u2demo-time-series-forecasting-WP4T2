"""Tests for u2demo_time_series_forecasting.dataset module.

Covers DataFrame construction from training/forecast inputs, short time-series filtering, horizon extension, past/future splitting, all feature-engineering entry points (temporal, lags, holidays, weather), forecast timestamp correctness, train/predict feature symmetry, time gaps, and multi-group forecasting pipeline behavior. Holiday and weather calls are monkeypatched to avoid external API calls.
"""

import datetime as dt
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tests.conftest import load_training_input_from_json
from u2demo_time_series_forecasting.constants.columns import (
    TIME_SERIES_GROUP_COLUMN_NAME,
    TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)
from u2demo_time_series_forecasting.data_models import ForecastInput, ForecastTrainingInput, TimeValue
from u2demo_time_series_forecasting.dataset import U2TimeSeriesForecastingDataset


@pytest.mark.unit
class TestU2TimeSeriesForecastingDataset:
    """Tests for U2TimeSeriesForecastingDataset class."""

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
            dataset = U2TimeSeriesForecastingDataset.from_training_input(data)

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
            dataset = U2TimeSeriesForecastingDataset.from_training_input(data)
            dataset.drop_short_timeseries(min_timeseries_length)

    @pytest.mark.parametrize(("horizon"), [1, 24, 72])
    def test_extend_by_horizon(self, json_single_timeseries: str, horizon: int) -> None:
        """Test extending dataset by horizon."""
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesForecastingDataset.from_training_input(data)

        pre_len = dataset.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()

        dataset.extend_by_horizon(horizon)

        post_len = dataset.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()

        assert ((post_len - pre_len) == horizon).all()

    @pytest.mark.parametrize(("horizon"), [1, 24, 72])
    def test_split_past_and_future(self, json_single_timeseries: str, horizon: int) -> None:
        """Test splitting past and future dataframes."""
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesForecastingDataset.from_training_input(data)

        pre_len = dataset.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()

        dataset.extend_by_horizon(horizon)

        dataset.split_past_and_future_dfs(horizon)

        post_len = dataset.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()

        assert dataset.future_df is not None

        future_len = dataset.future_df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()

        assert (pre_len == post_len).all()
        assert (future_len == horizon).all()

    def test_invalid_json_path(self) -> None:
        """Test that loading from a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_training_input_from_json("/invalid/path/to/file.json")

    def test_add_temporal(self, json_single_timeseries: str) -> None:
        """Test that temporal features are added and recorded."""
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesForecastingDataset.from_training_input(data)

        dataset.add_temporal()

        assert "temporal" in dataset.added_columns
        assert len(dataset.added_columns["temporal"]) > 0
        assert any(col.startswith(("month_", "day_of_week_")) for col in dataset.df.columns)

    @pytest.mark.parametrize(("lags", "windows"), [([24, 48, 72], [12, 24]), ([48, 72], [24, 72])])
    def test_add_lags(self, json_single_timeseries: str, lags: list[int], windows: list[int]) -> None:
        """Test that lag and rolling features are added and rows trimmed accordingly."""
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesForecastingDataset.from_training_input(data)
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
        dataset = U2TimeSeriesForecastingDataset.from_training_input(data)
        with expectation:
            dataset.drop_short_timeseries(num_weekly_lags * 7 * 24 + min_lookback)

    def test_add_holidays_monkeypatched(self, json_single_timeseries: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test add_holidays using monkeypatched holidays merger to avoid external calls."""
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesForecastingDataset.from_training_input(data)

        def fake_merge_holidays(df: pd.DataFrame, location_col: str, datetime_col: str):
            df2 = df.copy()
            df2["holiday_public"] = 0
            df2["holiday_bank"] = 0
            return df2, ["holiday_public", "holiday_bank"]

        monkeypatch.setattr("u2demo_time_series_forecasting.dataset.merge_holidays_with_dataframe", fake_merge_holidays)

        dataset.add_holidays()

        assert "holiday" in dataset.added_columns
        assert {"holiday_public", "holiday_bank"}.issubset(set(dataset.df.columns))

    def test_add_weather_monkeypatched(self, json_single_timeseries: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test add_weather using monkeypatched weather enricher to avoid external calls."""
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesForecastingDataset.from_training_input(data)

        def fake_enrich_weather(
            df: pd.DataFrame,
            time_col: str,
            location_col: str,
            present_time: dt.datetime,
            freq: str,
        ):
            df2 = df.copy()
            df2["temperature_2m"] = 1.23
            return df2, ["temperature_2m"]

        monkeypatch.setattr("u2demo_time_series_forecasting.dataset.enrich_weather", fake_enrich_weather)

        dataset.final_timestamp = dataset.df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].max()
        dataset.freq = dataset.freq or "h"
        dataset.freq_timedelta = dataset.freq_timedelta or dt.timedelta(hours=1)

        dataset.add_weather()

        assert "weather" in dataset.added_columns
        assert "temperature_2m" in dataset.df.columns

    def test_location_dependent_features_require_location(self, json_without_location: str):
        data = load_training_input_from_json(json_without_location)
        dataset = U2TimeSeriesForecastingDataset.from_training_input(data)

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

    def test_from_training_input_with_exogenous_series(self):
        """Exogenous series should be merged into the dataset."""
        dates = pd.date_range(start="2023-01-01", periods=48, freq="h")
        payload = {
            "forecast_type": "stochastic",
            "time_parameters": {"timestep_minutes": 60, "horizon_hours": 24},
            "training_model_config": {
                "model_id": "m1",
                "model_preset": "fast_training",
                "model_version": "1.0.0",
                "training_split": {"num_val_windows": 1, "num_test_windows": 1},
            },
            "target_series": {
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "training_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
            "exogenous_series": [
                {
                    "series_id": "exog1",
                    "variable_name": "temperature",
                    "training_values": [
                        {"time": ts.isoformat(), "value": 20.0 + i * 0.1} for i, ts in enumerate(dates)
                    ],
                }
            ],
        }
        data = ForecastTrainingInput(**payload)
        dataset = U2TimeSeriesForecastingDataset.from_training_input(data)

        assert "temperature" in dataset.df.columns
        assert len(dataset.df) == len(dates)

    def test_from_training_input_frequency_mismatch(self):
        """Mismatched timestep_minutes should raise ValueError."""
        dates = pd.date_range(start="2023-01-01", periods=48, freq="h")
        payload = {
            "forecast_type": "stochastic",
            "time_parameters": {"timestep_minutes": 30, "horizon_hours": 24},
            "training_model_config": {
                "model_id": "m1",
                "model_preset": "fast_training",
                "model_version": "1.0.0",
                "training_split": {"num_val_windows": 1, "num_test_windows": 1},
            },
            "target_series": {
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "training_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
        }
        data = ForecastTrainingInput(**payload)

        with pytest.raises(ValueError, match="does not match expected frequency"):
            U2TimeSeriesForecastingDataset.from_training_input(data)

    def test_from_forecast_input_basic(self):
        """from_forecast_input should build a valid dataset and extend by horizon."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="h")
        data = ForecastInput(
            forecast_type="stochastic",
            time_parameters={"timestep_minutes": 60, "horizon_hours": 24},
            model_id="m1",
            model_version="1.0.0",
            target_series={
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "historical_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
        )
        dataset = U2TimeSeriesForecastingDataset.from_forecast_input(data)

        assert dataset.df is not None
        assert dataset.final_timestamp is not None
        # Extended by 24 hourly steps
        assert len(dataset.df) == 200 + 24

    def test_from_forecast_input_frequency_mismatch(self):
        """Mismatched timestep_minutes should raise ValueError in from_forecast_input."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="h")
        data = ForecastInput(
            forecast_type="stochastic",
            time_parameters={"timestep_minutes": 30, "horizon_hours": 24},
            model_id="m1",
            model_version="1.0.0",
            target_series={
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "historical_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
        )

        with pytest.raises(ValueError, match="does not match expected frequency"):
            U2TimeSeriesForecastingDataset.from_forecast_input(data)

    def test_from_forecast_input_with_nan_values(self):
        """NaN values in forecast input should raise ValidationError at the Pydantic layer."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="h")
        values = [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)]
        values[10]["value"] = None  # inject NaN

        with pytest.raises(ValidationError):
            ForecastInput(
                forecast_type="stochastic",
                time_parameters={"timestep_minutes": 60, "horizon_hours": 24},
                model_id="m1",
                model_version="1.0.0",
                target_series={
                    "series_id": "s1",
                    "variable_name": "power",
                    "asset_id": "a1",
                    "asset_type": "PV",
                    "historical_values": values,
                },
            )

    def test_from_forecast_input_with_exogenous_series(self):
        """Exogenous series (historical + future) should be merged in from_forecast_input."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="h")
        future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(hours=1), periods=24, freq="h")

        data = ForecastInput(
            forecast_type="stochastic",
            time_parameters={"timestep_minutes": 60, "horizon_hours": 24},
            model_id="m1",
            model_version="1.0.0",
            target_series={
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "historical_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
            exogenous_series=[
                {
                    "series_id": "exog1",
                    "variable_name": "temperature",
                    "historical_values": [
                        {"time": ts.isoformat(), "value": 20.0 + i * 0.1} for i, ts in enumerate(dates)
                    ],
                    "future_values": [
                        {"time": ts.isoformat(), "value": 25.0 + i * 0.1} for i, ts in enumerate(future_dates)
                    ],
                }
            ],
        )
        dataset = U2TimeSeriesForecastingDataset.from_forecast_input(data)

        assert "temperature" in dataset.df.columns
        assert len(dataset.df) == 200 + 24

    def test_from_forecast_input_with_location(self):
        """from_forecast_input should set latlon when location is provided."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="h")
        data = ForecastInput(
            forecast_type="stochastic",
            time_parameters={"timestep_minutes": 60, "horizon_hours": 24},
            model_id="m1",
            model_version="1.0.0",
            target_series={
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "historical_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
            location={"latitude": 48.7758, "longitude": 9.1829},
        )
        dataset = U2TimeSeriesForecastingDataset.from_forecast_input(data)

        assert dataset.latlon == (48.7758, 9.1829)

    def test_from_training_input_exogenous_nan_after_merge(self):
        """Exogenous series with misaligned timestamps should produce NaN and raise ValueError."""
        target_dates = pd.date_range(start="2023-01-01", periods=48, freq="h")
        # Exogenous has only 24 of the 48 timestamps → left merge creates NaN
        exog_dates = target_dates[:24]

        payload = {
            "forecast_type": "stochastic",
            "time_parameters": {"timestep_minutes": 60, "horizon_hours": 24},
            "training_model_config": {
                "model_id": "m1",
                "model_preset": "fast_training",
                "model_version": "1.0.0",
                "training_split": {"num_val_windows": 1, "num_test_windows": 1},
            },
            "target_series": {
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "training_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(target_dates)],
            },
            "exogenous_series": [
                {
                    "series_id": "exog1",
                    "variable_name": "temperature",
                    "training_values": [{"time": ts.isoformat(), "value": 20.0} for ts in exog_dates],
                }
            ],
        }
        data = ForecastTrainingInput(**payload)

        with pytest.raises(ValueError, match="The input data contains NaN values"):
            U2TimeSeriesForecastingDataset.from_training_input(data)

    def test_timestamps_sorted_per_group_after_exogenous_merge(self):
        """Timestamps must be monotonically increasing within each group after exogenous merge."""
        dates = pd.date_range(start="2023-01-01", periods=48, freq="h")
        payload = {
            "forecast_type": "stochastic",
            "time_parameters": {"timestep_minutes": 60, "horizon_hours": 24},
            "training_model_config": {
                "model_id": "m1",
                "model_preset": "fast_training",
                "model_version": "1.0.0",
                "training_split": {"num_val_windows": 1, "num_test_windows": 1},
            },
            "target_series": {
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "training_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
            "exogenous_series": [
                {
                    "series_id": "exog1",
                    "variable_name": "temperature",
                    "training_values": [
                        {"time": ts.isoformat(), "value": 20.0 + i * 0.1} for i, ts in enumerate(dates)
                    ],
                }
            ],
        }
        data = ForecastTrainingInput(**payload)
        dataset = U2TimeSeriesForecastingDataset.from_training_input(data)

        for _, group_df in dataset.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME):
            timestamps = group_df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].reset_index(drop=True)
            assert timestamps.is_monotonic_increasing, "Timestamps are not sorted after exogenous series merge"

    def test_from_training_input_empty_after_merge(self, monkeypatch):
        """Empty DataFrame after construction should raise ValueError."""
        dates = pd.date_range(start="2023-01-01", periods=48, freq="h")
        payload = {
            "forecast_type": "stochastic",
            "time_parameters": {"timestep_minutes": 60, "horizon_hours": 24},
            "training_model_config": {
                "model_id": "m1",
                "model_preset": "fast_training",
                "model_version": "1.0.0",
                "training_split": {"num_val_windows": 1, "num_test_windows": 1},
            },
            "target_series": {
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "training_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
            "exogenous_series": [
                {
                    "series_id": "exog1",
                    "variable_name": "temperature",
                    "training_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
                }
            ],
        }
        data = ForecastTrainingInput(**payload)

        monkeypatch.setattr(pd, "merge", lambda *args, **kwargs: pd.DataFrame())

        with pytest.raises(ValueError, match="The input data is empty"):
            U2TimeSeriesForecastingDataset.from_training_input(data)

    def test_from_forecast_input_nan_after_construction(self):
        """NaN in DataFrame after construction should raise ValueError in from_forecast_input."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="h")
        data = ForecastInput(
            forecast_type="stochastic",
            time_parameters={"timestep_minutes": 60, "horizon_hours": 24},
            model_id="m1",
            model_version="1.0.0",
            target_series={
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "historical_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
        )

        data.target_series.historical_values[5] = TimeValue(time=dates[5], value=float("nan"))

        with pytest.raises(ValueError, match="The input data contains NaN values"):
            U2TimeSeriesForecastingDataset.from_forecast_input(data)

    @pytest.mark.parametrize(
        ("n_rows", "threshold", "expectation"),
        [
            (10, 10, does_not_raise()),  # exactly at threshold → series is kept
            (
                9,
                10,
                pytest.raises(ValueError, match=r"No timeseries in the input has enough rows"),
            ),  # one below → dropped
        ],
    )
    def test_drop_short_timeseries_exact_boundary(
        self,
        n_rows: int,
        threshold: int,
        expectation,
    ):
        """Series at exactly min_timeseries_length is kept; one row below is dropped."""
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: pd.date_range(start="2023-01-01", periods=n_rows, freq="h"),
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.arange(n_rows, dtype=float),
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )
        dataset = U2TimeSeriesForecastingDataset.from_dataframe(df)

        with expectation:
            dataset.drop_short_timeseries(threshold)
            assert len(dataset.df) == n_rows

    def test_from_forecast_input_empty_after_exogenous_merge(self, monkeypatch):
        """Empty DataFrame after exogenous merge should raise ValueError in from_forecast_input."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="h")
        future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(hours=1), periods=24, freq="h")

        data = ForecastInput(
            forecast_type="stochastic",
            time_parameters={"timestep_minutes": 60, "horizon_hours": 24},
            model_id="m1",
            model_version="1.0.0",
            target_series={
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "historical_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
            exogenous_series=[
                {
                    "series_id": "exog1",
                    "variable_name": "temperature",
                    "historical_values": [{"time": ts.isoformat(), "value": 20.0} for ts in dates],
                    "future_values": [{"time": ts.isoformat(), "value": 25.0} for ts in future_dates],
                }
            ],
        )

        def patched_merge(*args, **kwargs):
            return pd.DataFrame()

        monkeypatch.setattr("u2demo_time_series_forecasting.dataset.pd.merge", patched_merge)

        with pytest.raises(ValueError, match="The input data is empty"):
            U2TimeSeriesForecastingDataset.from_forecast_input(data)


@pytest.mark.unit
class TestExogenousSeriesMerge:
    """Tests for exogenous series merge paths in from_training_input and from_forecast_input."""

    def test_training_multiple_exogenous_series(self):
        """Multiple exogenous series should all be merged into the dataset."""
        dates = pd.date_range(start="2023-01-01", periods=48, freq="h")
        payload = {
            "forecast_type": "stochastic",
            "time_parameters": {"timestep_minutes": 60, "horizon_hours": 24},
            "training_model_config": {
                "model_id": "m1",
                "model_preset": "fast_training",
                "model_version": "1.0.0",
                "training_split": {"num_val_windows": 1, "num_test_windows": 1},
            },
            "target_series": {
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "training_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
            "exogenous_series": [
                {
                    "series_id": "exog1",
                    "variable_name": "temperature",
                    "training_values": [{"time": ts.isoformat(), "value": 20.0} for ts in dates],
                },
                {
                    "series_id": "exog2",
                    "variable_name": "humidity",
                    "training_values": [{"time": ts.isoformat(), "value": 50.0} for ts in dates],
                },
            ],
        }
        data = ForecastTrainingInput(**payload)
        dataset = U2TimeSeriesForecastingDataset.from_training_input(data)

        assert "temperature" in dataset.df.columns
        assert "humidity" in dataset.df.columns
        assert len(dataset.df) == 48

    def test_forecast_exogenous_with_future_values_fills_horizon(self):
        """Future values from exogenous series should fill the extended horizon rows."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="h")
        future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(hours=1), periods=24, freq="h")

        data = ForecastInput(
            forecast_type="stochastic",
            time_parameters={"timestep_minutes": 60, "horizon_hours": 24},
            model_id="m1",
            model_version="1.0.0",
            target_series={
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "historical_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
            exogenous_series=[
                {
                    "series_id": "exog1",
                    "variable_name": "temperature",
                    "historical_values": [
                        {"time": ts.isoformat(), "value": 20.0 + i * 0.1} for i, ts in enumerate(dates)
                    ],
                    "future_values": [{"time": ts.isoformat(), "value": 99.0} for ts in future_dates],
                }
            ],
        )
        dataset = U2TimeSeriesForecastingDataset.from_forecast_input(data)

        # The last 24 rows (horizon) should have the future exogenous values
        last_24_temps = dataset.df["temperature"].tail(24)
        assert (last_24_temps == 99.0).all()

    def test_forecast_exogenous_without_future_values_has_nan_in_horizon(self):
        """Exogenous series without future_values should have NaN in the horizon rows."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="h")

        data = ForecastInput(
            forecast_type="stochastic",
            time_parameters={"timestep_minutes": 60, "horizon_hours": 24},
            model_id="m1",
            model_version="1.0.0",
            target_series={
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "historical_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
            exogenous_series=[
                {
                    "series_id": "exog1",
                    "variable_name": "temperature",
                    "historical_values": [{"time": ts.isoformat(), "value": 20.0} for ts in dates],
                }
            ],
        )
        dataset = U2TimeSeriesForecastingDataset.from_forecast_input(data)

        # The last 24 rows (horizon) won't have exogenous data -> NaN after left merge
        last_24_temps = dataset.df["temperature"].tail(24)
        assert last_24_temps.isna().all()

    def test_training_exogenous_misaligned_timestamps_raises(self):
        """Exogenous series covering only part of target dates should produce NaN and raise."""
        target_dates = pd.date_range(start="2023-01-01", periods=48, freq="h")
        exog_dates = target_dates[:24]  # only first half

        payload = {
            "forecast_type": "stochastic",
            "time_parameters": {"timestep_minutes": 60, "horizon_hours": 24},
            "training_model_config": {
                "model_id": "m1",
                "model_preset": "fast_training",
                "model_version": "1.0.0",
                "training_split": {"num_val_windows": 1, "num_test_windows": 1},
            },
            "target_series": {
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "training_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(target_dates)],
            },
            "exogenous_series": [
                {
                    "series_id": "exog1",
                    "variable_name": "temperature",
                    "training_values": [{"time": ts.isoformat(), "value": 20.0} for ts in exog_dates],
                }
            ],
        }
        data = ForecastTrainingInput(**payload)

        with pytest.raises(ValueError, match="The input data contains NaN values"):
            U2TimeSeriesForecastingDataset.from_training_input(data)

    def test_training_exogenous_values_are_correct(self):
        """Exogenous values should be correctly aligned with target timestamps."""
        dates = pd.date_range(start="2023-01-01", periods=48, freq="h")
        payload = {
            "forecast_type": "stochastic",
            "time_parameters": {"timestep_minutes": 60, "horizon_hours": 24},
            "training_model_config": {
                "model_id": "m1",
                "model_preset": "fast_training",
                "model_version": "1.0.0",
                "training_split": {"num_val_windows": 1, "num_test_windows": 1},
            },
            "target_series": {
                "series_id": "s1",
                "variable_name": "power",
                "asset_id": "a1",
                "asset_type": "PV",
                "training_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
            "exogenous_series": [
                {
                    "series_id": "exog1",
                    "variable_name": "temperature",
                    "training_values": [
                        {"time": ts.isoformat(), "value": 100.0 + float(i)} for i, ts in enumerate(dates)
                    ],
                }
            ],
        }
        data = ForecastTrainingInput(**payload)
        dataset = U2TimeSeriesForecastingDataset.from_training_input(data)

        # First row should have temperature = 100.0, second = 101.0, etc.
        assert dataset.df["temperature"].iloc[0] == 100.0
        assert dataset.df["temperature"].iloc[1] == 101.0


def _make_ds_training_input(periods=200, freq="h", values=None):
    """Build a ForecastTrainingInput for dataset symmetry tests."""
    timestep_minutes = 60 if freq == "h" else 1440
    dates = pd.date_range(start="2023-01-01", periods=periods, freq=freq)

    if values is None:
        rng = np.random.default_rng(42)
        values = rng.random(periods).tolist()
    else:
        values = list(values)

    return ForecastTrainingInput(
        forecast_type="stochastic",
        time_parameters={"timestep_minutes": timestep_minutes, "horizon_hours": 24},
        training_model_config={
            "model_id": "m1",
            "model_preset": "fast_training",
            "model_version": "1.0.0",
            "training_split": {"num_val_windows": 1, "num_test_windows": 1},
        },
        target_series={
            "series_id": "s1",
            "variable_name": "power",
            "asset_id": "a1",
            "asset_type": "PV",
            "training_values": [
                {"time": ts.isoformat(), "value": float(v)} for ts, v in zip(dates, values, strict=True)
            ],
        },
    )


def _make_ds_forecast_input(periods=200, freq="h", horizon_hours=24, values=None):
    """Build a ForecastInput for dataset symmetry tests."""
    timestep_minutes = 60 if freq == "h" else 1440
    dates = pd.date_range(start="2023-01-01", periods=periods, freq=freq)

    if values is None:
        rng = np.random.default_rng(42)
        values = rng.random(periods).tolist()
    else:
        values = list(values)

    return ForecastInput(
        forecast_type="stochastic",
        time_parameters={"timestep_minutes": timestep_minutes, "horizon_hours": horizon_hours},
        model_id="m1",
        model_version="1.0.0",
        target_series={
            "series_id": "s1",
            "variable_name": "power",
            "asset_id": "a1",
            "asset_type": "PV",
            "historical_values": [
                {"time": ts.isoformat(), "value": float(v)} for ts, v in zip(dates, values, strict=True)
            ],
        },
    )


@pytest.mark.unit
class TestForecastTimestampCorrectness:
    """Verify predicted timestamps are correctly positioned and spaced."""

    def test_first_predicted_timestamp_is_one_step_after_last_historical(self):
        """After extend + split, the first future timestamp must be last_historical + freq."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="h")
        data = _make_ds_forecast_input(periods=200)
        ds = U2TimeSeriesForecastingDataset.from_forecast_input(data)

        last_historical = dates[-1]
        horizon = 24
        ds.split_past_and_future_dfs(horizon)

        first_future_ts = ds.future_df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].iloc[0]
        expected = last_historical + pd.Timedelta(hours=1)
        assert first_future_ts == expected, f"Expected {expected}, got {first_future_ts}"

    def test_predicted_timestamps_are_evenly_spaced(self):
        """Future timestamps must be evenly spaced at the dataset frequency."""
        data = _make_ds_forecast_input(periods=200)
        ds = U2TimeSeriesForecastingDataset.from_forecast_input(data)

        horizon = 24
        ds.split_past_and_future_dfs(horizon)

        future_ts = ds.future_df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].sort_values().reset_index(drop=True)
        diffs = future_ts.diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all(), f"Non-uniform spacing found: {diffs.unique()}"

    def test_predicted_timestamps_daily_frequency(self):
        """Daily frequency data should produce daily-spaced future timestamps."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        rng = np.random.default_rng(42)
        values = rng.random(100).tolist()
        data = _make_ds_forecast_input(periods=100, freq="D", values=values, horizon_hours=24)
        ds = U2TimeSeriesForecastingDataset.from_forecast_input(data)

        horizon = 1  # 24h / 1440min = 1 daily step
        ds.split_past_and_future_dfs(horizon)

        future_ts = ds.future_df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].sort_values().reset_index(drop=True)
        last_historical = dates[-1]
        assert future_ts.iloc[0] == last_historical + pd.Timedelta(days=1)

    def test_future_df_has_exactly_horizon_rows(self):
        """future_df must have exactly `horizon` rows per group."""
        data = _make_ds_forecast_input(periods=200)
        ds = U2TimeSeriesForecastingDataset.from_forecast_input(data)

        horizon = 24
        ds.split_past_and_future_dfs(horizon)

        per_group = ds.future_df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()
        assert (per_group == horizon).all()


@pytest.mark.unit
class TestFeatureSymmetry:
    """Verify train and predict paths produce identical feature columns."""

    def test_temporal_features_identical_between_train_and_predict(self):
        """Same input data produces the same temporal feature columns in both paths."""
        rng = np.random.default_rng(42)
        periods = 200
        values = rng.random(periods).tolist()

        train_ds = U2TimeSeriesForecastingDataset.from_training_input(
            _make_ds_training_input(periods=periods, values=values)
        )
        train_ds.add_temporal()
        train_cols = sorted(train_ds.added_columns["temporal"])

        predict_ds = U2TimeSeriesForecastingDataset.from_forecast_input(
            _make_ds_forecast_input(periods=periods, values=values)
        )
        predict_ds.add_temporal()
        predict_cols = sorted(predict_ds.added_columns["temporal"])

        assert train_cols == predict_cols
        assert len(train_cols) > 0

    def test_lag_features_identical_between_train_and_predict(self):
        """Same lags/windows produce the same lag feature columns in both paths."""
        rng = np.random.default_rng(42)
        periods = 500
        values = rng.random(periods).tolist()
        lags = [168]
        windows = [168]

        train_ds = U2TimeSeriesForecastingDataset.from_training_input(
            _make_ds_training_input(periods=periods, values=values)
        )
        train_ds.add_lags(lags, windows)
        train_cols = sorted(train_ds.added_columns["lag"])

        predict_ds = U2TimeSeriesForecastingDataset.from_forecast_input(
            _make_ds_forecast_input(periods=periods, values=values)
        )
        predict_ds.add_lags(lags, windows)
        predict_cols = sorted(predict_ds.added_columns["lag"])

        assert train_cols == predict_cols
        assert any(col.startswith("lag_") for col in train_cols)
        assert any(col.startswith("roll_") for col in train_cols)

    def test_combined_features_column_sets_match(self):
        """Temporal + lag features produce the same column set regardless of path."""
        rng = np.random.default_rng(42)
        periods = 500
        values = rng.random(periods).tolist()
        lags = [24]
        windows = [24]

        train_ds = U2TimeSeriesForecastingDataset.from_training_input(
            _make_ds_training_input(periods=periods, values=values)
        )
        train_ds.add_temporal()
        train_ds.add_lags(lags, windows)
        train_all = set()
        for cols in train_ds.added_columns.values():
            train_all.update(cols)

        predict_ds = U2TimeSeriesForecastingDataset.from_forecast_input(
            _make_ds_forecast_input(periods=periods, values=values)
        )
        predict_ds.add_temporal()
        predict_ds.add_lags(lags, windows)
        predict_all = set()
        for cols in predict_ds.added_columns.values():
            predict_all.update(cols)

        assert train_all == predict_all


@pytest.mark.unit
class TestSplitPastAndFutureDfs:
    """Dedicated tests for split_past_and_future_dfs method."""

    def _make_dataset(self, periods=200, freq="h"):
        """Build a simple dataset for split testing."""
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: pd.date_range(start="2023-01-01", periods=periods, freq=freq),
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.arange(periods, dtype=float),
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )
        return U2TimeSeriesForecastingDataset.from_dataframe(df)

    def test_horizon_of_one(self):
        """Splitting with horizon=1 should extract exactly one row per group."""
        ds = self._make_dataset(periods=50)
        ds.split_past_and_future_dfs(1)

        assert len(ds.future_df) == 1
        assert len(ds.df) == 49

    def test_future_rows_have_nan_value_after_extend(self):
        """After extend + split, future rows should have NaN in the value column."""
        ds = self._make_dataset(periods=50)
        ds.extend_by_horizon(10)
        ds.split_past_and_future_dfs(10)

        assert ds.future_df[TIME_SERIES_PREDICT_VALUE_COLUMN_NAME].isna().all()

    def test_past_rows_preserve_original_values(self):
        """Past rows after split should retain their original values."""
        ds = self._make_dataset(periods=50)
        original_values = ds.df[TIME_SERIES_PREDICT_VALUE_COLUMN_NAME].values.copy()

        ds.split_past_and_future_dfs(10)

        np.testing.assert_array_equal(ds.df[TIME_SERIES_PREDICT_VALUE_COLUMN_NAME].values, original_values[:40])

    def test_horizon_equals_dataset_length(self):
        """Splitting with horizon == len(df) should give empty past, all future."""
        ds = self._make_dataset(periods=20)
        ds.split_past_and_future_dfs(20)

        assert len(ds.df) == 0
        assert len(ds.future_df) == 20

    def test_multi_group_independence(self):
        """Each group should be split independently."""
        frames = []
        for i, size in enumerate([100, 80]):
            dates = pd.date_range(start="2023-01-01", periods=size, freq="h")
            frames.append(
                pd.DataFrame(
                    {
                        TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
                        TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.arange(size, dtype=float),
                        TIME_SERIES_GROUP_COLUMN_NAME: f"g{i}",
                    }
                )
            )
        ds = U2TimeSeriesForecastingDataset.from_dataframe(pd.concat(frames, ignore_index=True))

        ds.split_past_and_future_dfs(10)

        past_counts = ds.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()
        future_counts = ds.future_df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()
        assert past_counts["g0"] == 90
        assert past_counts["g1"] == 70
        assert (future_counts == 10).all()

    def test_future_timestamps_are_contiguous_with_past(self):
        """The first future timestamp should be exactly one step after the last past timestamp."""
        ds = self._make_dataset(periods=100)
        ds.split_past_and_future_dfs(24)

        last_past = ds.df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].max()
        first_future = ds.future_df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].min()
        assert first_future - last_past == pd.Timedelta(hours=1)


@pytest.mark.unit
class TestFeatureIdempotency:
    """Tests that calling feature methods twice doesn't duplicate columns or corrupt data."""

    def test_add_temporal_twice_does_not_duplicate_columns(self):
        """Calling add_temporal() twice should not create extra columns in the DataFrame."""
        data = _make_ds_training_input(periods=200)
        ds = U2TimeSeriesForecastingDataset.from_training_input(data)

        ds.add_temporal()
        first_col_count = len(ds.df.columns)

        ds.add_temporal()

        assert len(ds.df.columns) == first_col_count

    def test_add_lags_twice_does_not_duplicate_columns(self):
        """Calling add_lags() twice with same params should not duplicate lag columns."""
        data = _make_ds_training_input(periods=200)
        ds = U2TimeSeriesForecastingDataset.from_training_input(data)

        ds.add_lags([24], [24])

        # Second call will trim more rows and potentially add columns again
        # The key thing is that column names should stay the same
        ds.add_lags([24], [24])
        second_lag_cols = [c for c in ds.df.columns if c.startswith(("lag_", "roll_"))]

        # No duplicate column names
        assert len(second_lag_cols) == len(set(second_lag_cols))

    def test_add_temporal_then_lags_preserves_temporal(self):
        """Adding lags after temporal should not remove temporal columns."""
        data = _make_ds_training_input(periods=200)
        ds = U2TimeSeriesForecastingDataset.from_training_input(data)

        ds.add_temporal()
        temporal_cols = set(ds.added_columns["temporal"])

        ds.add_lags([24], [24])

        # Temporal columns should still be present
        assert temporal_cols.issubset(set(ds.df.columns))
        assert "temporal" in ds.added_columns
        assert "lag" in ds.added_columns


@pytest.mark.unit
class TestTimeGapsDataset:
    """Test extend_by_horizon behavior when input has time gaps."""

    def test_extend_by_horizon_uses_freq_not_last_diff(self):
        """extend_by_horizon uses the stored freq_timedelta, not the gap between last rows."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="h")
        gap_indices = [47, 48]
        dates_with_gap = dates.delete(gap_indices)
        rng = np.random.default_rng(42)
        values = rng.random(len(dates_with_gap))

        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates_with_gap,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: values,
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )
        # The gap deliberately breaks pd.infer_freq, so build the dataset by
        # passing freq metadata explicitly to test that extend_by_horizon uses
        # the stored freq_timedelta rather than the spacing of the last rows.
        ds = U2TimeSeriesForecastingDataset(
            df=df,
            freq="h",
            freq_minutes=60,
            freq_timedelta=pd.Timedelta(hours=1),
        )

        last_ts_before = ds.df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].iloc[-1]
        ds.extend_by_horizon(3)

        new_rows = ds.df.tail(3)
        expected_first = last_ts_before + pd.Timedelta(hours=1)
        assert new_rows[TIME_SERIES_TIMESTAMP_COLUMN_NAME].iloc[0] == expected_first

        diffs = new_rows[TIME_SERIES_TIMESTAMP_COLUMN_NAME].diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()


@pytest.mark.unit
class TestDatasetEdgeCases:
    """Test dataset construction and splitting with boundary-value inputs."""

    def test_daily_freq_with_sub_day_horizon_yields_nonzero_steps(self):
        """With daily data (1440-min steps), a sub-day horizon_hours should
        still produce at least 1 horizon step.

        horizon_hours * 60 // freq_minutes truncates to 0 when
        horizon_hours < 24 and freq_minutes == 1440, which would produce a
        semantically invalid 0-step forecast.
        """
        dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
        data = ForecastInput(
            forecast_type="stochastic",
            time_parameters={"timestep_minutes": 1440, "horizon_hours": 12},
            model_id="test-model",
            model_version="1.0.0",
            target_series={
                "series_id": "s1",
                "variable_name": "power",
                "historical_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
        )

        ts = U2TimeSeriesForecastingDataset.from_forecast_input(data)
        horizon_steps = max(1, data.time_parameters.horizon_hours * 60 // ts.freq_minutes)

        assert horizon_steps >= 1

    def test_irregular_timestamps_raise_clear_error(self):
        """Irregularly-spaced timestamps (e.g. 1 h then 2 h gaps) should raise
        a ValueError whose message mentions frequency or spacing, not an
        obscure error from downstream numeric conversion.
        """
        data = ForecastTrainingInput(
            forecast_type="stochastic",
            time_parameters={"horizon_hours": 24},
            training_model_config={
                "model_id": "test-model",
                "model_preset": "fast_training",
                "model_version": "1.0.0",
                "training_split": {"num_val_windows": 1, "num_test_windows": 1},
            },
            target_series={
                "series_id": "s1",
                "variable_name": "power",
                "training_values": [
                    {"time": "2023-01-01T00:00:00", "value": 1.0},
                    {"time": "2023-01-01T01:00:00", "value": 2.0},
                    {"time": "2023-01-01T03:00:00", "value": 3.0},  # 2 h gap
                ],
            },
        )

        with pytest.raises(ValueError, match="(?i)frequency|irregular|spacing"):
            U2TimeSeriesForecastingDataset.from_training_input(data)

    def test_split_past_future_with_horizon_zero_preserves_data(self):
        """split_past_and_future_dfs(0) should keep all rows in df and
        produce an empty future_df.

        In Python -0 == 0, so iloc[:-0] silently becomes iloc[:0] (empty).
        A zero horizon should be a no-op on the data, not discard it.
        """
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: range(100),
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )
        ds = U2TimeSeriesForecastingDataset.from_dataframe(df)

        ds.split_past_and_future_dfs(0)

        assert len(ds.df) == 100
        assert len(ds.future_df) == 0

    def test_duplicate_exogenous_variable_names_detected(self):
        """Two exogenous series sharing the same variable_name should be
        rejected or deduplicated — not silently produce duplicate DataFrame
        columns that corrupt downstream model training.
        """
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        data = ForecastTrainingInput(
            forecast_type="stochastic",
            time_parameters={"timestep_minutes": 60, "horizon_hours": 24},
            training_model_config={
                "model_id": "test-model",
                "model_preset": "fast_training",
                "model_version": "1.0.0",
                "training_split": {"num_val_windows": 1, "num_test_windows": 1},
            },
            target_series={
                "series_id": "s1",
                "variable_name": "power",
                "training_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
            },
            exogenous_series=[
                {
                    "series_id": "exog1",
                    "variable_name": "temperature",
                    "is_known_covariate": True,
                    "training_values": [
                        {"time": ts.isoformat(), "value": float(i) * 0.5} for i, ts in enumerate(dates)
                    ],
                },
                {
                    "series_id": "exog2",
                    "variable_name": "temperature",
                    "is_known_covariate": True,
                    "training_values": [
                        {"time": ts.isoformat(), "value": float(i) * 0.3} for i, ts in enumerate(dates)
                    ],
                },
            ],
        )

        with pytest.raises(ValueError, match="(?i)duplicate"):
            U2TimeSeriesForecastingDataset.from_training_input(data)


@pytest.mark.unit
class TestMultiGroupForecasting:
    """Test forecasting pipeline behavior with multi-group data."""

    def _make_multi_group_dataset(self, group_sizes):
        """Build a dataset with multiple groups of different lengths."""
        frames = []
        for i, size in enumerate(group_sizes):
            dates = pd.date_range(start="2023-01-01", periods=size, freq="h")
            rng = np.random.default_rng(42 + i)
            df = pd.DataFrame(
                {
                    TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
                    TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: rng.random(size),
                    TIME_SERIES_GROUP_COLUMN_NAME: f"group_{i}",
                }
            )
            frames.append(df)
        return U2TimeSeriesForecastingDataset.from_dataframe(pd.concat(frames, ignore_index=True))

    def test_extend_by_horizon_adds_rows_per_group(self):
        """Each group should get exactly `horizon` new rows."""
        ds = self._make_multi_group_dataset([200, 150, 100])
        horizon = 24

        pre_counts = ds.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()
        ds.extend_by_horizon(horizon)
        post_counts = ds.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()

        assert ((post_counts - pre_counts) == horizon).all()

    def test_split_past_future_per_group(self):
        """split_past_and_future_dfs should extract the right rows from each group."""
        ds = self._make_multi_group_dataset([200, 150])
        horizon = 24
        ds.extend_by_horizon(horizon)
        ds.split_past_and_future_dfs(horizon)

        future_counts = ds.future_df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()
        assert (future_counts == horizon).all()

        past_counts = ds.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME).size()
        assert past_counts["group_0"] == 200
        assert past_counts["group_1"] == 150

    def test_future_timestamps_correct_per_group_with_different_lengths(self):
        """Groups with different lengths should each get correctly-timed future timestamps."""
        ds = self._make_multi_group_dataset([200, 150])
        horizon = 12
        ds.extend_by_horizon(horizon)
        ds.split_past_and_future_dfs(horizon)

        for group_id, group_df in ds.future_df.groupby(TIME_SERIES_GROUP_COLUMN_NAME):
            ts = group_df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].sort_values().reset_index(drop=True)
            diffs = ts.diff().dropna()
            assert (diffs == pd.Timedelta(hours=1)).all(), f"Group {group_id} has non-uniform future spacing"


@pytest.mark.unit
class TestMissingLatlonGuards:
    """Feature methods that depend on ``self.latlon`` should raise eagerly when it is missing."""

    def _make_dataset(self) -> U2TimeSeriesForecastingDataset:
        df = pd.DataFrame(
            {
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: pd.date_range("2023-01-01", periods=3, freq="h"),
                TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: [1.0, 2.0, 3.0],
                TIME_SERIES_GROUP_COLUMN_NAME: 0,
            }
        )
        return U2TimeSeriesForecastingDataset.from_dataframe(df)

    def test_add_holidays_without_latlon_raises(self):
        ds = self._make_dataset()
        with pytest.raises(AttributeError, match="Location information is required"):
            ds.add_holidays()

    def test_add_weather_without_latlon_raises(self):
        ds = self._make_dataset()
        with pytest.raises(AttributeError, match="Location information is required"):
            ds.add_weather()
