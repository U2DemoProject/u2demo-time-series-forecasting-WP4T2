"""Tests for u2demo_time_series_forecasting.features.pipeline.apply_features.

Covers the shared feature-engineering dispatcher used by both train and predict:
enum coercion, per-feature delegation, and the lag-requires-lags-and-windows guard.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from u2demo_time_series_forecasting.data_models import ForecastInput, ForecastTrainingInput
from u2demo_time_series_forecasting.dataset import U2TimeSeriesForecastingDataset
from u2demo_time_series_forecasting.features.names import FeatureName
from u2demo_time_series_forecasting.features.pipeline import apply_features


@pytest.mark.unit
class TestApplyFeatures:
    """Tests for apply_features dispatcher."""

    def test_empty_features_is_noop(self):
        """An empty feature list should not call any dataset method."""
        ts = MagicMock()

        apply_features(ts, [])

        ts.add_temporal.assert_not_called()
        ts.add_lags.assert_not_called()
        ts.add_holidays.assert_not_called()
        ts.add_weather.assert_not_called()

    def test_temporal_delegates_to_add_temporal(self):
        """FeatureName.TEMPORAL should call add_temporal only."""
        ts = MagicMock()

        apply_features(ts, [FeatureName.TEMPORAL])

        ts.add_temporal.assert_called_once_with()
        ts.add_lags.assert_not_called()
        ts.add_holidays.assert_not_called()
        ts.add_weather.assert_not_called()

    def test_lag_delegates_to_add_lags(self):
        """FeatureName.LAG should forward lags and windows to add_lags."""
        ts = MagicMock()

        apply_features(ts, [FeatureName.LAG], lags=[1, 2], windows=[3, 4])

        ts.add_lags.assert_called_once_with([1, 2], [3, 4])

    def test_lag_without_lags_raises(self):
        """LAG without lags should raise ValueError."""
        ts = MagicMock()

        with pytest.raises(ValueError, match="lag feature requires lags and windows"):
            apply_features(ts, [FeatureName.LAG], windows=[1])

        ts.add_lags.assert_not_called()

    def test_lag_without_windows_raises(self):
        """LAG without windows should raise ValueError."""
        ts = MagicMock()

        with pytest.raises(ValueError, match="lag feature requires lags and windows"):
            apply_features(ts, [FeatureName.LAG], lags=[1])

        ts.add_lags.assert_not_called()

    def test_holiday_delegates_to_add_holidays(self):
        """FeatureName.HOLIDAY should call add_holidays."""
        ts = MagicMock()

        apply_features(ts, [FeatureName.HOLIDAY])

        ts.add_holidays.assert_called_once_with()

    def test_weather_delegates_to_add_weather(self):
        """FeatureName.WEATHER should call add_weather."""
        ts = MagicMock()

        apply_features(ts, [FeatureName.WEATHER])

        ts.add_weather.assert_called_once_with()

    def test_string_inputs_are_coerced_to_enum(self):
        """Plain strings matching a FeatureName value should be accepted."""
        ts = MagicMock()

        apply_features(ts, ["temporal", "holiday"])

        ts.add_temporal.assert_called_once_with()
        ts.add_holidays.assert_called_once_with()

    def test_unknown_string_raises(self):
        """An unknown feature string should be rejected by the enum coercion."""
        ts = MagicMock()

        with pytest.raises(ValueError, match="nope"):
            apply_features(ts, ["nope"])

    def test_all_features_called_in_order(self):
        """When every feature is requested, all four methods are called."""
        ts = MagicMock()

        apply_features(
            ts,
            [FeatureName.TEMPORAL, FeatureName.LAG, FeatureName.HOLIDAY, FeatureName.WEATHER],
            lags=[1],
            windows=[1],
        )

        ts.add_temporal.assert_called_once_with()
        ts.add_lags.assert_called_once_with([1], [1])
        ts.add_holidays.assert_called_once_with()
        ts.add_weather.assert_called_once_with()


def _training_input_with_lag(periods: int) -> ForecastTrainingInput:
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="h")
    return ForecastTrainingInput(
        forecast_type="stochastic",
        time_parameters={"timestep_minutes": 60, "horizon_hours": 24},
        training_model_config={
            "model_id": "lag-symmetry",
            "model_preset": "fast_training",
            "model_version": "1.0.0",
            "training_split": {"num_val_windows": 1, "num_test_windows": 1},
        },
        target_series={
            "series_id": "s1",
            "variable_name": "active_power",
            "asset_id": "pv1",
            "asset_type": "PV",
            "training_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
        },
        features=["lag"],
        num_weekly_lags=2,
    )


def _forecast_input_with_lag(periods: int) -> ForecastInput:
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="h")
    return ForecastInput(
        forecast_type="stochastic",
        time_parameters={"timestep_minutes": 60, "horizon_hours": 24},
        model_id="lag-symmetry",
        model_version="1.0.0",
        target_series={
            "series_id": "s1",
            "variable_name": "active_power",
            "asset_id": "pv1",
            "asset_type": "PV",
            "unit": "kW",
            "historical_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
        },
    )


@pytest.mark.unit
class TestTrainPredictLagSymmetry:
    """Guard against drift between train.py and predict.py lag handling.

    Training derives lag offsets dynamically from ``num_weekly_lags`` and the
    inferred frequency. It then persists the resulting feature columns and the
    raw lag values in ``ModelMetadata``. Forecasting must be able to read those
    persisted lags back and replay ``apply_features`` to produce the *exact*
    same column names — otherwise the predictor will see covariates it was
    never trained on (or be missing some it expects). This test asserts that
    round-trip identity end-to-end without going through AutoGluon.
    """

    def test_lag_columns_round_trip_through_metadata(self):
        # Step 1: build a training dataset and apply the lag pipeline using
        # the same lag derivation that u2demo_time_series_forecasting.train.train() uses for hourly data.
        train_ds = U2TimeSeriesForecastingDataset.from_training_input(_training_input_with_lag(periods=2000))
        hourly_steps_per_week = 168
        num_weekly_lags = 2
        computed_lags = [hourly_steps_per_week * (1 + i) for i in range(num_weekly_lags)]

        apply_features(train_ds, [FeatureName.LAG], lags=computed_lags, windows=computed_lags)
        train_lag_columns = sorted(train_ds.added_columns[FeatureName.LAG])

        # Step 2: project added columns back into the (lags, windows) tuples
        # that u2demo_time_series_forecasting.model._save_metadata() persists into ModelMetadata.
        persisted_lags = sorted(int(c.split("_")[1]) for c in train_lag_columns if c.startswith("lag_"))
        persisted_windows = sorted(int(c.split("_")[2]) for c in train_lag_columns if c.startswith("roll_mean_"))
        assert persisted_lags == sorted(computed_lags), (
            "lag columns added by training do not round-trip into the metadata.lags list"
        )
        assert persisted_windows == sorted(computed_lags), (
            "rolling windows added by training do not round-trip into the metadata.windows list"
        )

        # Step 3: build a fresh forecast dataset and replay apply_features
        # using only the persisted metadata, exactly like u2demo_time_series_forecasting.predict.predict().
        forecast_ds = U2TimeSeriesForecastingDataset.from_forecast_input(_forecast_input_with_lag(periods=2000))
        apply_features(forecast_ds, [FeatureName.LAG], lags=persisted_lags, windows=persisted_windows)
        forecast_lag_columns = sorted(forecast_ds.added_columns[FeatureName.LAG])

        # Step 4: the column sets must match exactly — any drift means
        # AutoGluon will see different covariates between train and predict.
        assert forecast_lag_columns == train_lag_columns
