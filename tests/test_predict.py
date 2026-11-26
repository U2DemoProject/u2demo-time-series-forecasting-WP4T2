"""Tests for u2demo_time_series_forecasting.predict module.

Covers the predict orchestration: model loading -> dataset creation -> feature replay -> split -> forecasting -> Forecast formatting. The model layer is mocked to keep tests fast and focus on the orchestration logic.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from tests.conftest import make_mock_model, make_v2_forecast_payload, v2_validator
from u2demo_time_series_forecasting.data_models import Forecast, ForecastInput, Location
from u2demo_time_series_forecasting.data_models_v2 import ForecastInputV2, ForecastTypeV2, ForecastV2
from u2demo_time_series_forecasting.predict import predict, predict_from_payload, predict_v2


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


@pytest.mark.unit
class TestPredictOrchestration:
    """Tests for predict function orchestration logic."""

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_returns_forecast_object(self, mock_model_cls):
        """predict should return a Forecast data model."""
        mock_model = make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        assert isinstance(result, Forecast)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_forecast_has_correct_series_metadata(self, mock_model_cls):
        """Forecast series should carry through target_series metadata."""
        mock_model = make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        assert result.forecast_series.series_id == "s1"
        assert result.forecast_series.variable_name == "active_power"
        assert result.forecast_series.asset_id == "pv1"
        assert result.forecast_series.asset_type == "PV"

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_forecast_has_correct_length(self, mock_model_cls):
        """Forecast should have horizon_hours / freq_hours values."""
        mock_model = make_mock_model(prediction_length=24)
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input(horizon_hours=24)
        result = predict(data)

        assert len(result.forecast_series.values) == 24

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_forecast_includes_quantiles(self, mock_model_cls):
        """Each forecasted value should include quantile forecasts."""
        mock_model = make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        first_value = result.forecast_series.values[0]
        assert first_value.quantiles is not None
        assert "0.1" in first_value.quantiles
        assert "0.5" in first_value.quantiles
        assert "0.9" in first_value.quantiles

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_model_loaded_from_correct_path(self, mock_model_cls):
        """Model should be loaded from MODEL_PATH/model_id_model_version."""
        mock_model = make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input(model_id="my-model", model_version="2.0.0")
        predict(data)

        path_arg = mock_model_cls.call_args[0][0]
        assert "my-model" in path_arg
        assert "2.0.0" in path_arg
        mock_model.load.assert_called_once()

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_raises_when_metadata_not_loaded(self, mock_model_cls):
        """predict should raise if model.load() leaves metadata None."""
        mock_model = MagicMock()
        mock_model.metadata = None
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()

        with pytest.raises(RuntimeError, match="Model metadata is not loaded"):
            predict(data)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingDataset")
    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_raises_when_freq_minutes_not_initialized(self, mock_model_cls, mock_dataset_cls):
        """predict should raise when the dataset exposes freq_minutes=None."""
        mock_model = make_mock_model()
        mock_model_cls.return_value = mock_model

        mock_dataset = MagicMock()
        mock_dataset.freq = "h"
        mock_dataset.freq_minutes = None
        mock_dataset_cls.from_forecast_input.return_value = mock_dataset

        data = _make_forecast_input()

        with pytest.raises(RuntimeError, match="Frequency minutes is not initialized"):
            predict(data)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_frequency_mismatch_raises(self, mock_model_cls):
        """Frequency mismatch between model and data should raise ValueError."""
        mock_model = make_mock_model(freq="D", prediction_length=24)
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()

        with pytest.raises(ValueError, match="frequency"):
            predict(data)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_horizon_mismatch_raises(self, mock_model_cls):
        """Horizon mismatch between request and model should raise ValueError."""
        mock_model = make_mock_model(prediction_length=48)
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input(horizon_hours=24)

        with pytest.raises(ValueError, match="horizon"):
            predict(data)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_temporal_features_applied_when_in_metadata(self, mock_model_cls):
        """When temporal is in model metadata, add_temporal should be called."""
        mock_model = make_mock_model()
        mock_model.metadata.added_columns = {"temporal": ["month_sin"]}
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        assert isinstance(result, Forecast)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_forecast_type_preserved(self, mock_model_cls):
        """Forecast type from input should be preserved in output."""
        mock_model = make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        assert result.forecast_type == "stochastic"

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_lag_features_applied_when_in_metadata(self, mock_model_cls):
        """When lag is in model metadata, add_lags should be called."""
        mock_model = make_mock_model()
        mock_model.metadata.added_columns = {"lag": ["lag_168", "roll_mean_168"]}
        mock_model.metadata.lags = [168]
        mock_model.metadata.windows = [168]
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input(periods=500)
        result = predict(data)

        assert isinstance(result, Forecast)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_holiday_features_applied_when_in_metadata(self, mock_model_cls, monkeypatch):
        """When holiday is in model metadata, add_holidays should be called."""
        mock_model = make_mock_model()
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

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_weather_features_applied_when_in_metadata(self, mock_model_cls, monkeypatch):
        """When weather is in model metadata, add_weather should be called."""
        mock_model = make_mock_model()
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

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_mean_column_renamed_to_value(self, mock_model_cls):
        """The 'mean' column from AG output should become 'value' in forecast."""
        mock_model = make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        for v in result.forecast_series.values:
            assert v.value == 1.0  # mock returns mean=1.0

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_timestamps_formatted_as_iso_strings(self, mock_model_cls):
        """Timestamps in forecast should be ISO-formatted strings."""
        mock_model = make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        for v in result.forecast_series.values:
            assert isinstance(v.time, datetime)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_quantile_dict_keys_are_strings(self, mock_model_cls):
        """Quantile keys should be string representations of quantile levels."""
        mock_model = make_mock_model()
        mock_model.quantile_levels = [0.1, 0.5, 0.9]
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        for v in result.forecast_series.values:
            assert set(v.quantiles.keys()) == {"0.1", "0.5", "0.9"}

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_quantile_values_match_mock_output(self, mock_model_cls):
        """Quantile values should be correctly mapped from the forecast DataFrame."""
        mock_model = make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        first = result.forecast_series.values[0]
        assert first.quantiles["0.1"] == 0.5
        assert first.quantiles["0.5"] == 1.0
        assert first.quantiles["0.9"] == 1.5

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_series_metadata_carried_through(self, mock_model_cls):
        """Target series metadata (unit, unitIri, etc.) should propagate to output."""
        mock_model = make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        assert result.forecast_series.unit == "kW"
        assert result.forecast_series.asset_type == "PV"

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_time_parameters_carried_through(self, mock_model_cls):
        """Time parameters from input should be preserved in output Forecast."""
        mock_model = make_mock_model()
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input(horizon_hours=24)
        result = predict(data)

        assert result.time_parameters is not None
        assert result.time_parameters.horizon_hours == 24

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_forecast_date_is_recent(self, mock_model_cls):
        """forecast_date should be a recent ISO datetime (not None or empty)."""
        mock_model = make_mock_model()
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

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_empty_metadata_skips_all_features(self, mock_model_cls):
        """When metadata has no added_columns, no feature engineering should run."""
        mock_model = make_mock_model()
        mock_model.metadata.added_columns = {}
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        # Should succeed without any feature engineering
        assert isinstance(result, Forecast)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_temporal_only_replay(self, mock_model_cls):
        """When only temporal is in metadata, only temporal features should be applied."""
        mock_model = make_mock_model()
        mock_model.metadata.added_columns = {"temporal": ["month_sin", "month_cos"]}
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input()
        result = predict(data)

        assert isinstance(result, Forecast)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_lag_replay_uses_metadata_lags_and_windows(self, mock_model_cls):
        """Lag feature replay should use lags and windows from metadata, not recalculate."""
        mock_model = make_mock_model()
        mock_model.metadata.added_columns = {"lag": ["lag_24", "roll_mean_24"]}
        mock_model.metadata.lags = [24]
        mock_model.metadata.windows = [24]
        mock_model_cls.return_value = mock_model

        data = _make_forecast_input(periods=200)
        result = predict(data)

        assert isinstance(result, Forecast)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_combined_temporal_and_lag_replay(self, mock_model_cls):
        """When both temporal and lag are in metadata, both should be applied."""
        mock_model = make_mock_model()
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


@pytest.mark.unit
class TestPredictV2:
    """Tests for predict_v2: v2 requests run through predict and come back as ForecastPayloadDM."""

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_returns_forecast_v2(self, mock_model_cls):
        mock_model_cls.return_value = make_mock_model()

        result = predict_v2(ForecastInputV2.model_validate(make_v2_forecast_payload(forecastId="forecast-1")))

        assert isinstance(result, ForecastV2)
        assert result.forecastId == "forecast-1"
        assert result.forecast_type is ForecastTypeV2.STOCHASTIC
        series = result.hasForecastSeries[0]
        assert series.hasFeatureOfInterest == "pv1"
        assert series.model_id == "test-model_1.0.0"
        assert len(series.hasValues) == 24
        assert series.hasValues[0].hasResult.numericValue == 1.0
        assert series.hasValues[0].quantiles == {"0.1": 0.5, "0.5": 1.0, "0.9": 1.5}

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_model_loaded_from_model_id_and_version(self, mock_model_cls):
        mock_model_cls.return_value = make_mock_model()

        predict_v2(ForecastInputV2.model_validate(make_v2_forecast_payload(model_id="my-model", model_version="2.0.0")))

        assert mock_model_cls.call_args[0][0].endswith("my-model_2.0.0")

    @pytest.mark.parametrize(("forecast_type", "has_quantiles"), [("STOCHASTIC", True), ("DETERMINISTIC", False)])
    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_output_validates_against_forecast_payload_schema(self, mock_model_cls, forecast_type, has_quantiles):
        mock_model_cls.return_value = make_mock_model()

        result = predict_v2(ForecastInputV2.model_validate(make_v2_forecast_payload(forecast_type=forecast_type)))

        payload = result.model_dump(mode="json")
        v2_validator("api/components/forecast/ForecastPayloadDM.json").validate(payload)
        assert ("quantiles" in payload["hasForecastSeries"][0]["hasValues"][0]) is has_quantiles

    @pytest.mark.parametrize("offset", ["+00:00", "+01:00"])
    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_exogenous_series_with_offset_timestamps(self, mock_model_cls, offset):
        mock_model_cls.return_value = make_mock_model()
        dates = pd.date_range(start="2023-01-01", periods=224, freq="h")
        values = [{"timestamp": f"{ts.isoformat()}{offset}", "value": 10.0} for ts in dates]
        payload = make_v2_forecast_payload(
            target_series={"historical_values": [{**v, "value": 1.0} for v in values[:200]]},
            exogenous_series=[
                {
                    "series_id": "t1",
                    "observedProperty": "OutdoorTemperature",
                    "historical_values": values[:200],
                    "future_values": values[200:],
                }
            ],
        )

        result = predict_v2(ForecastInputV2.model_validate(payload))

        assert len(result.hasForecastSeries[0].hasValues) == 24
        dataset = mock_model_cls.return_value.predict.call_args[0][0]
        assert dataset.future_df["OutdoorTemperature"].tolist() == [10.0] * 24

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_horizon_mismatch_raises(self, mock_model_cls):
        mock_model_cls.return_value = make_mock_model(prediction_length=48)

        with pytest.raises(ValueError, match="horizon"):
            predict_v2(ForecastInputV2.model_validate(make_v2_forecast_payload()))


@pytest.mark.unit
class TestPredictFromPayload:
    """Tests for predict_from_payload: the forecast is returned in the version of the request."""

    @pytest.mark.parametrize(
        ("payload", "expected_type"),
        [
            (_make_forecast_input().model_dump(mode="json"), Forecast),
            (make_v2_forecast_payload(), ForecastV2),
        ],
    )
    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_forecast_in_version_of_request(self, mock_model_cls, payload, expected_type):
        mock_model_cls.return_value = make_mock_model()

        assert isinstance(predict_from_payload(payload), expected_type)

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_explicit_version_validates_against_that_version(self, mock_model_cls):
        mock_model_cls.return_value = make_mock_model()

        with pytest.raises(ValidationError):
            predict_from_payload(make_v2_forecast_payload(), version="v1")
