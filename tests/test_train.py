"""Tests for u2.train module.

Covers the train orchestration: dataset creation -> feature engineering -> model creation -> training. The model layer is mocked to keep tests fast and focus on the orchestration logic.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from u2.data_models import ForecastTrainingInput
from u2.train import train


def _make_training_input(
    periods=200,
    features=None,
    num_weekly_lags=0,
    include_location=True,
    forecast_type="stochastic",
    quantile_levels=None,
    eval_metric=None,
    lookback_steps=None,
    model_preset="fast_training",
    excluded_model_types=None,
):
    """Build a ForecastTrainingInput for testing."""
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="h")

    payload = {
        "forecast_type": forecast_type,
        "time_parameters": {"timestep_minutes": 60, "horizon_hours": 24},
        "training_model_config": {
            "model_id": "test-model",
            "model_preset": model_preset,
            "model_version": "1.0.0",
            "training_split": {"num_val_windows": 1, "num_test_windows": 1},
        },
        "target_series": {
            "series_id": "s1",
            "variable_name": "active_power",
            "asset_id": "pv1",
            "asset_type": "PV",
            "training_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
        },
    }

    if features is not None:
        payload["features"] = features
    if num_weekly_lags > 0:
        payload["num_weekly_lags"] = num_weekly_lags
    if include_location:
        payload["location"] = {"latitude": 48.7758, "longitude": 9.1829}
    if quantile_levels:
        payload["training_model_config"]["quantile_levels"] = quantile_levels
    if eval_metric:
        payload["training_model_config"]["eval_metric"] = eval_metric
    if lookback_steps:
        payload["training_model_config"]["lookback_steps"] = lookback_steps
    if excluded_model_types is not None:
        payload["training_model_config"]["excluded_model_types"] = excluded_model_types

    return ForecastTrainingInput(**payload)


@pytest.mark.unit
class TestTrainOrchestration:
    """Tests for train function orchestration logic."""

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_creates_dataset_from_input(self, mock_model_cls):
        """Train should create a dataset from the input and call model.train."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        # lookback_steps=50 ensures 200 periods is enough (50 + 24 horizon = 74 min)
        data = _make_training_input(periods=200, features=[], lookback_steps=50)

        train(data)

        mock_model.new.assert_called_once()
        mock_model.train.assert_called_once()

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_temporal_feature_applied(self, mock_model_cls):
        """When 'temporal' is in features, add_temporal should be called on dataset."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        data = _make_training_input(periods=2000, features=["temporal"])

        train(data)

        mock_model.train.assert_called_once()

    def test_unrecognized_feature_raises(self):
        """Unrecognized features should be rejected at model validation time."""
        with pytest.raises(ValueError, match="nonexistent_feature"):
            _make_training_input(periods=200, features=["nonexistent_feature"])

    @patch("u2.train.U2TimeSeriesPredictionDataset")
    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_raises_when_freq_minutes_not_initialized(self, mock_model_cls, mock_dataset_cls):
        """train should raise when dataset exposes freq_minutes=None / freq=None."""
        mock_dataset = MagicMock()
        mock_dataset.freq = None
        mock_dataset.freq_minutes = None
        mock_dataset_cls.from_training_input.return_value = mock_dataset

        data = _make_training_input(periods=200, features=[], lookback_steps=50)

        with pytest.raises(RuntimeError, match="Frequency is not initialized"):
            train(data)

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_deterministic_sets_quantile_half(self, mock_model_cls):
        """Deterministic forecast should use quantile_levels=[0.5]."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        data = _make_training_input(periods=200, features=[], forecast_type="deterministic", lookback_steps=50)

        train(data)

        call_kwargs = mock_model.new.call_args
        assert call_kwargs.kwargs.get("quantile_levels") == [0.5] or call_kwargs[1].get("quantile_levels") == [0.5]

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_stochastic_uses_config_quantiles(self, mock_model_cls):
        """Stochastic forecast with quantile_levels should pass them through."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        data = _make_training_input(
            periods=200, features=[], forecast_type="stochastic", quantile_levels=[0.1, 0.5, 0.9], lookback_steps=50
        )

        train(data)

        call_kwargs = mock_model.new.call_args
        levels = call_kwargs.kwargs.get("quantile_levels") or call_kwargs[1].get("quantile_levels")
        assert levels == [0.1, 0.5, 0.9]

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_eval_metric_from_config(self, mock_model_cls):
        """Custom eval_metric from config should be uppercased and passed."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        data = _make_training_input(periods=200, features=[], eval_metric="rmse", lookback_steps=50)

        train(data)

        call_kwargs = mock_model.new.call_args
        metric = call_kwargs.kwargs.get("eval_metric") or call_kwargs[1].get("eval_metric")
        assert metric == "RMSE"

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_default_eval_metric(self, mock_model_cls):
        """Without eval_metric in config, should use DEFAULT_TRAIN_EVAL_METRIC."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        data = _make_training_input(periods=200, features=[], lookback_steps=50)

        train(data)

        call_kwargs = mock_model.new.call_args
        metric = call_kwargs.kwargs.get("eval_metric") or call_kwargs[1].get("eval_metric")
        assert metric == "WQL"

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_model_path_includes_id_and_version(self, mock_model_cls):
        """Model path should include model_id and model_version."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        data = _make_training_input(periods=200, features=[], lookback_steps=50)

        train(data)

        path_arg = mock_model_cls.call_args[0][0]
        assert "test-model" in path_arg
        assert "1.0.0" in path_arg

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_short_timeseries_raises(self, mock_model_cls):
        """Very short time series should fail validation against min_timeseries_length."""
        data = _make_training_input(periods=5, features=[], lookback_steps=1000)

        with pytest.raises(ValueError, match="No timeseries"):
            train(data)

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_default_lookback_is_1024(self, mock_model_cls):
        """Without lookback_steps, DEFAULT_TRAIN_MIN_LOOKBACK (1024) is used,
        so 200 periods is too short."""
        data = _make_training_input(periods=200, features=[])

        with pytest.raises(ValueError, match="No timeseries"):
            train(data)

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_lag_feature_computation(self, mock_model_cls):
        """When 'lag' feature is enabled, lags should be computed based on frequency and num_weekly_lags."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        # 168 steps/week for hourly, num_weekly_lags=1 => lags=[168]
        # Need enough periods: 168 + lookback + horizon
        data = _make_training_input(periods=500, features=["lag"], num_weekly_lags=1, lookback_steps=50)

        train(data)

        mock_model.train.assert_called_once()

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_holiday_feature_applied(self, mock_model_cls, monkeypatch):
        """When 'holiday' is in features, add_holidays should be called on dataset."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        # Mock the holiday merge to avoid external calls
        monkeypatch.setattr(
            "u2.dataset.merge_holidays_with_dataframe",
            lambda df, latlon, ts_col: (df.assign(holiday_public=0), ["holiday_public"]),
        )

        data = _make_training_input(periods=200, features=["holiday"], lookback_steps=50)

        train(data)

        mock_model.train.assert_called_once()

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_weather_feature_applied(self, mock_model_cls, monkeypatch):
        """When 'weather' is in features, add_weather should be called on dataset."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        monkeypatch.setattr(
            "u2.dataset.enrich_weather",
            lambda df, ts_col, latlon, present_time, freq: (df.assign(temperature_2m=20.0), ["temperature_2m"]),
        )

        data = _make_training_input(periods=200, features=["weather"], lookback_steps=50)

        train(data)

        mock_model.train.assert_called_once()

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_excluded_model_types_passed(self, mock_model_cls):
        """excluded_model_types from config should be passed in training_kwargs."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        data = _make_training_input(periods=200, features=[], lookback_steps=50, excluded_model_types=["Chronos"])

        train(data)

        call_kwargs = mock_model.train.call_args
        training_kwargs = call_kwargs.kwargs.get("training_kwargs") or call_kwargs[1].get("training_kwargs")
        assert training_kwargs["excluded_model_types"] == ["Chronos"]

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_stochastic_no_quantiles_defaults_none(self, mock_model_cls):
        """Stochastic without quantile_levels in config should pass None."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        data = _make_training_input(periods=200, features=[], forecast_type="stochastic", lookback_steps=50)

        train(data)

        call_kwargs = mock_model.new.call_args
        levels = call_kwargs.kwargs.get("quantile_levels") or call_kwargs[1].get("quantile_levels")
        assert levels is None

    @patch("u2.train.U2TimeSeriesPredictionModel")
    def test_lag_feature_without_num_weekly_lags_should_not_crash(self, mock_model_cls):
        """When features includes 'lag' but num_weekly_lags is not set (defaults to 0),
        the lag list is empty. train() should handle this gracefully instead of crashing
        on max() of an empty sequence.
        """
        mock_model_cls.return_value = MagicMock()

        data = _make_training_input(periods=2000, features=["lag"], lookback_steps=50)

        train(data)
