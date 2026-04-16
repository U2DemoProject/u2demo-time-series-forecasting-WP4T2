"""Tests for u2demo_time_series_forecasting.predict module.

Covers the predict orchestration: model loading -> dataset creation -> feature replay -> split -> prediction -> Forecast formatting. The model layer is mocked to keep tests fast and focus on the orchestration logic.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from u2demo_time_series_forecasting.data_models import Forecast, ForecastInput, Location
from u2demo_time_series_forecasting.metadata import ModelMetadata, TrainingMetadata
from u2demo_time_series_forecasting.predict import predict


def _make_forecast_input(periods=200, horizon_hours=24, model_id="test-model", model_version="1.0.0"):
    """Build a ForecastInput for testing."""
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="h")

    return ForecastInput(
        forecast_type="stochastic",
        time_parameters={"timestep_minutes": 60, "horizon_hours": horizon_hours},
        model_id=model_id,
        model_version=model_version,
        target_series={
            "series_id": "s1",
            "variable_name": "active_power",
            "asset_id": "pv1",
            "asset_type": "PV",
            "unit": "kW",
            "historical_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
        },
    )


def _make_metadata(freq="h", prediction_length=24, added_columns=None, lags=None, windows=None):
    """Build a ModelMetadata instance matching a trained model."""
    return ModelMetadata(
        version="0.1.0",
        frequency=freq,
        horizon=prediction_length,
        min_timeseries_length=50,
        added_columns=added_columns or {},
        lags=lags or [],
        windows=windows or [],
        training=TrainingMetadata(
            start_utc="2023-01-01 00:00:00",
            end_utc="2023-01-01 00:01:00",
            min_lookback=50,
            num_val_windows=1,
            num_test_windows=1,
        ),
    )


def _make_mock_model(freq="h", prediction_length=24, metadata=None):
    """Build a mock model that looks like a loaded U2TimeSeriesPredictionModel."""
    mock_model = MagicMock()
    mock_model.metadata = metadata or _make_metadata(freq=freq, prediction_length=prediction_length)
    mock_model.freq = freq
    mock_model.prediction_length = prediction_length
    mock_model.quantile_levels = [0.1, 0.5, 0.9]

    # Build realistic prediction output
    future_dates = pd.date_range("2023-01-09T08:00:00", periods=prediction_length, freq=freq)
    index = pd.MultiIndex.from_arrays(
        [[0] * prediction_length, future_dates],
        names=["item_id", "timestamp"],
    )
    pred_df = pd.DataFrame(
        {
            "mean": [1.0] * prediction_length,
            "0.1": [0.5] * prediction_length,
            "0.5": [1.0] * prediction_length,
            "0.9": [1.5] * prediction_length,
        },
        index=index,
    )
    mock_model.predict.return_value = pred_df
    return mock_model


@pytest.mark.unit
class TestPredictOrchestration:
    """Tests for predict function orchestration logic."""

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_returns_forecast_object(self, mock_model_cls):
        """predict should return a Forecast data model."""
        mock_model = _make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        assert isinstance(result, Forecast)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_forecast_has_correct_series_metadata(self, mock_model_cls):
        """Forecast series should carry through target_series metadata."""
        mock_model = _make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        assert result.forecast_series.series_id == "s1"
        assert result.forecast_series.variable_name == "active_power"
        assert result.forecast_series.asset_id == "pv1"
        assert result.forecast_series.asset_type == "PV"

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_forecast_has_correct_length(self, mock_model_cls):
        """Forecast should have horizon_hours / freq_hours values."""
        mock_model = _make_mock_model(prediction_length=24)
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input(horizon_hours=24)
        result = predict(data)

        assert len(result.forecast_series.values) == 24

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_forecast_includes_quantiles(self, mock_model_cls):
        """Each forecasted value should include quantile predictions."""
        mock_model = _make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        first_value = result.forecast_series.values[0]
        assert first_value.quantiles is not None
        assert "0.1" in first_value.quantiles
        assert "0.5" in first_value.quantiles
        assert "0.9" in first_value.quantiles

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_model_loaded_from_correct_path(self, mock_model_cls):
        """Model should be loaded from MODEL_PATH/model_id_model_version."""
        mock_model = _make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input(model_id="my-model", model_version="2.0.0")
        predict(data)

        path_arg = mock_model_cls.call_args[0][0]
        assert "my-model" in path_arg
        assert "2.0.0" in path_arg
        mock_model.load.assert_called_once()

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_raises_when_metadata_not_loaded(self, mock_model_cls):
        """predict should raise if model.load() leaves metadata None."""
        mock_model = MagicMock()
        mock_model.metadata = None
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()

        with pytest.raises(RuntimeError, match="Model metadata is not loaded"):
            predict(data)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionDataset")
    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_raises_when_freq_minutes_not_initialized(self, mock_model_cls, mock_dataset_cls):
        """predict should raise when the dataset exposes freq_minutes=None."""
        mock_model = _make_mock_model()
        mock_model_cls.return_value = mock_model

        mock_dataset = MagicMock()
        mock_dataset.freq = "h"
        mock_dataset.freq_minutes = None
        mock_dataset_cls.from_forecast_input.return_value = mock_dataset

        data = _make_forecast_input()

        with pytest.raises(RuntimeError, match="Frequency minutes is not initialized"):
            predict(data)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_frequency_mismatch_raises(self, mock_model_cls):
        """Frequency mismatch between model and data should raise ValueError."""
        mock_model = _make_mock_model(freq="D", prediction_length=24)
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()

        with pytest.raises(ValueError, match="frequency"):
            predict(data)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_horizon_mismatch_raises(self, mock_model_cls):
        """Horizon mismatch between request and model should raise ValueError."""
        mock_model = _make_mock_model(prediction_length=48)
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input(horizon_hours=24)

        with pytest.raises(ValueError, match="horizon"):
            predict(data)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_temporal_features_applied_when_in_metadata(self, mock_model_cls):
        """When temporal is in model metadata, add_temporal should be called."""
        mock_model = _make_mock_model()
        mock_model.metadata.added_columns = {"temporal": ["month_sin"]}
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        assert isinstance(result, Forecast)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_forecast_type_preserved(self, mock_model_cls):
        """Forecast type from input should be preserved in output."""
        mock_model = _make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        assert result.forecast_type == "stochastic"

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_lag_features_applied_when_in_metadata(self, mock_model_cls):
        """When lag is in model metadata, add_lags should be called."""
        mock_model = _make_mock_model()
        mock_model.metadata.added_columns = {"lag": ["lag_168", "roll_mean_168"]}
        mock_model.metadata.lags = [168]
        mock_model.metadata.windows = [168]
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input(periods=500)
        result = predict(data)

        assert isinstance(result, Forecast)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_holiday_features_applied_when_in_metadata(self, mock_model_cls, monkeypatch):
        """When holiday is in model metadata, add_holidays should be called."""
        mock_model = _make_mock_model()
        mock_model.metadata.added_columns = {"holiday": ["holiday_public"]}
        mock_model_cls.return_value = mock_model

        monkeypatch.setattr(
            "u2demo_time_series_forecasting.dataset.merge_holidays_with_dataframe",
            lambda df, latlon, ts_col: (df.assign(holiday_public=0), ["holiday_public"]),
        )

        data = _make_forecast_input()
        data.location = Location(latitude=48.7758, longitude=9.1829)
        result = predict(data)

        assert isinstance(result, Forecast)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_weather_features_applied_when_in_metadata(self, mock_model_cls, monkeypatch):
        """When weather is in model metadata, add_weather should be called."""
        mock_model = _make_mock_model()
        mock_model.metadata.added_columns = {"weather": ["temperature_2m"]}
        mock_model_cls.return_value = mock_model

        monkeypatch.setattr(
            "u2demo_time_series_forecasting.dataset.enrich_weather",
            lambda df, ts_col, latlon, present_time, freq: (df.assign(temperature_2m=20.0), ["temperature_2m"]),
        )

        data = _make_forecast_input()
        data.location = Location(latitude=48.7758, longitude=9.1829)
        result = predict(data)

        assert isinstance(result, Forecast)


@pytest.mark.unit
class TestPredictTransformation:
    """Tests for the predict output DataFrame transformation (predict.py lines 88-105).

    This is the glue code that reshapes AutoGluon output into our Forecast schema:
    column rename (mean -> value), timestamp formatting, quantile dict assembly.
    """

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_mean_column_renamed_to_value(self, mock_model_cls):
        """The 'mean' column from AG output should become 'value' in forecast."""
        mock_model = _make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        for v in result.forecast_series.values:
            assert v.value == 1.0  # mock returns mean=1.0

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_timestamps_formatted_as_iso_strings(self, mock_model_cls):
        """Timestamps in forecast should be ISO-formatted strings."""
        mock_model = _make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        for v in result.forecast_series.values:
            assert isinstance(v.time, datetime)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_quantile_dict_keys_are_strings(self, mock_model_cls):
        """Quantile keys should be string representations of quantile levels."""
        mock_model = _make_mock_model()
        mock_model.quantile_levels = [0.1, 0.5, 0.9]
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        for v in result.forecast_series.values:
            assert set(v.quantiles.keys()) == {"0.1", "0.5", "0.9"}

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_quantile_values_match_mock_output(self, mock_model_cls):
        """Quantile values should be correctly mapped from the prediction DataFrame."""
        mock_model = _make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        first = result.forecast_series.values[0]
        assert first.quantiles["0.1"] == 0.5
        assert first.quantiles["0.5"] == 1.0
        assert first.quantiles["0.9"] == 1.5

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_series_metadata_carried_through(self, mock_model_cls):
        """Target series metadata (unit, unitIri, etc.) should propagate to output."""
        mock_model = _make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        assert result.forecast_series.unit == "kW"
        assert result.forecast_series.asset_type == "PV"

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_time_parameters_carried_through(self, mock_model_cls):
        """Time parameters from input should be preserved in output Forecast."""
        mock_model = _make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input(horizon_hours=24)
        result = predict(data)

        assert result.time_parameters is not None
        assert result.time_parameters.horizon_hours == 24

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_forecast_date_is_recent(self, mock_model_cls):
        """forecast_date should be a recent ISO datetime (not None or empty)."""
        mock_model = _make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        assert result.forecast_date is not None


@pytest.mark.unit
class TestFeatureReplaySymmetry:
    """Tests that metadata-driven feature replay in predict matches training.

    When predict loads metadata with specific features, it must apply the exact
    same feature engineering that training did.
    """

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_empty_metadata_skips_all_features(self, mock_model_cls):
        """When metadata has no added_columns, no feature engineering should run."""
        mock_model = _make_mock_model()
        mock_model.metadata.added_columns = {}
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        # Should succeed without any feature engineering
        assert isinstance(result, Forecast)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_temporal_only_replay(self, mock_model_cls):
        """When only temporal is in metadata, only temporal features should be applied."""
        mock_model = _make_mock_model()
        mock_model.metadata.added_columns = {"temporal": ["month_sin", "month_cos"]}
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        assert isinstance(result, Forecast)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_lag_replay_uses_metadata_lags_and_windows(self, mock_model_cls):
        """Lag feature replay should use lags and windows from metadata, not recalculate."""
        mock_model = _make_mock_model()
        mock_model.metadata.added_columns = {"lag": ["lag_24", "roll_mean_24"]}
        mock_model.metadata.lags = [24]
        mock_model.metadata.windows = [24]
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input(periods=200)
        result = predict(data)

        assert isinstance(result, Forecast)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesPredictionModel")
    def test_combined_temporal_and_lag_replay(self, mock_model_cls):
        """When both temporal and lag are in metadata, both should be applied."""
        mock_model = _make_mock_model()
        mock_model.metadata.added_columns = {
            "temporal": ["month_sin"],
            "lag": ["lag_24", "roll_mean_24"],
        }
        mock_model.metadata.lags = [24]
        mock_model.metadata.windows = [24]
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input(periods=200)
        result = predict(data)

        assert isinstance(result, Forecast)
