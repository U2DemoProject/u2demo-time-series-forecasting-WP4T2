"""Tests for u2demo_time_series_forecasting.data_models module.

Covers validation boundaries, required fields, type coercion, and serialization round-trips for all Pydantic models.
"""

from datetime import UTC, datetime

import pytest
from autogluon.timeseries.configs import get_predictor_presets
from pydantic import ValidationError

from u2demo_time_series_forecasting.data_models import (
    Forecast,
    ForecastedTimeValue,
    ForecastExogenousSeries,
    ForecastInput,
    ForecastSeries,
    ForecastTargetSeries,
    ForecastTrainingInput,
    Location,
    ModelConfig,
    TimeParameters,
    TimeValue,
    TrainingExogenousSeries,
    TrainingSplit,
    TrainingTargetSeries,
)


@pytest.mark.unit
class TestTimeValue:
    """Tests for TimeValue model."""

    def test_valid_time_value(self):
        tv = TimeValue(time=datetime(2023, 1, 1), value=42.0)
        assert tv.time == datetime(2023, 1, 1)
        assert tv.value == 42.0

    def test_iso_string_coercion(self):
        tv = TimeValue(time="2023-01-01T00:00:00", value=1.0)
        assert isinstance(tv.time, datetime)

    def test_missing_time_raises(self):
        with pytest.raises(ValidationError):
            TimeValue(value=1.0)

    def test_missing_value_raises(self):
        with pytest.raises(ValidationError):
            TimeValue(time=datetime(2023, 1, 1))


@pytest.mark.unit
class TestForecastedTimeValue:
    """Tests for ForecastedTimeValue model."""

    def test_without_quantiles(self):
        ftv = ForecastedTimeValue(time=datetime(2023, 1, 1), value=1.0)
        assert ftv.quantiles is None

    def test_with_valid_quantiles(self):
        ftv = ForecastedTimeValue(
            time=datetime(2023, 1, 1),
            value=1.0,
            quantiles={"0.1": 0.8, "0.5": 1.0, "0.9": 1.2},
        )
        assert ftv.quantiles["0.5"] == 1.0

    def test_quantile_key_boundary_zero(self):
        ftv = ForecastedTimeValue(time=datetime(2023, 1, 1), value=1.0, quantiles={"0": 0.5})
        assert "0" in ftv.quantiles

    def test_quantile_key_boundary_one(self):
        ftv = ForecastedTimeValue(time=datetime(2023, 1, 1), value=1.0, quantiles={"1": 1.5})
        assert "1" in ftv.quantiles

    def test_quantile_key_with_decimals(self):
        ftv = ForecastedTimeValue(time=datetime(2023, 1, 1), value=1.0, quantiles={"0.25": 0.9, "0.75": 1.1})
        assert len(ftv.quantiles) == 2

    def test_invalid_quantile_key_rejects(self):
        with pytest.raises(ValidationError):
            ForecastedTimeValue(time=datetime(2023, 1, 1), value=1.0, quantiles={"1.5": 2.0})

    def test_invalid_quantile_key_negative(self):
        with pytest.raises(ValidationError):
            ForecastedTimeValue(time=datetime(2023, 1, 1), value=1.0, quantiles={"-0.1": 0.0})

    def test_invalid_quantile_key_text(self):
        with pytest.raises(ValidationError):
            ForecastedTimeValue(time=datetime(2023, 1, 1), value=1.0, quantiles={"high": 1.0})


@pytest.mark.unit
class TestTimeParameters:
    """Tests for TimeParameters model."""

    def test_defaults(self):
        tp = TimeParameters(horizon_hours=24)
        assert tp.timestep_minutes is None
        assert tp.horizon_hours == 24
        assert tp.scheduling_time is None

    def test_horizon_minimum(self):
        tp = TimeParameters(horizon_hours=1)
        assert tp.horizon_hours == 1

    def test_horizon_zero_rejects(self):
        with pytest.raises(ValidationError):
            TimeParameters(horizon_hours=0)

    def test_timestep_minutes_minimum(self):
        tp = TimeParameters(timestep_minutes=1, horizon_hours=24)
        assert tp.timestep_minutes == 1

    def test_timestep_minutes_zero_rejects(self):
        with pytest.raises(ValidationError):
            TimeParameters(timestep_minutes=0, horizon_hours=24)


@pytest.mark.unit
class TestTrainingSplit:
    """Tests for TrainingSplit model."""

    def test_defaults(self):
        ts = TrainingSplit()
        assert ts.num_val_windows == 1
        assert ts.num_test_windows == 1

    def test_custom_values(self):
        ts = TrainingSplit(num_val_windows=5, num_test_windows=3)
        assert ts.num_val_windows == 5
        assert ts.num_test_windows == 3

    def test_zero_val_windows_rejects(self):
        with pytest.raises(ValidationError):
            TrainingSplit(num_val_windows=0)

    def test_zero_test_windows_rejects(self):
        with pytest.raises(ValidationError):
            TrainingSplit(num_test_windows=0)


@pytest.mark.unit
class TestModelConfig:
    """Tests for ModelConfig model."""

    def test_required_model_id(self):
        mc = ModelConfig(model_id="test-model")
        assert mc.model_id == "test-model"

    def test_defaults(self):
        mc = ModelConfig(model_id="test")
        assert mc.model_preset == "chronos2"
        assert mc.model_version is None
        assert mc.lookback_steps is None
        assert mc.eval_metric is None
        assert mc.quantile_levels is None

    def test_default_preset_is_valid_autogluon_preset(self):
        mc = ModelConfig(model_id="test")
        assert mc.model_preset in get_predictor_presets()

    def test_missing_model_id_rejects(self):
        with pytest.raises(ValidationError):
            ModelConfig()

    def test_lookback_steps_minimum(self):
        mc = ModelConfig(model_id="test", lookback_steps=1)
        assert mc.lookback_steps == 1

    def test_lookback_steps_zero_rejects(self):
        with pytest.raises(ValidationError):
            ModelConfig(model_id="test", lookback_steps=0)

    def test_extra_fields_allowed(self):
        mc = ModelConfig(model_id="test", custom_field="custom_value")
        assert mc.custom_field == "custom_value"

    def test_quantile_levels_valid(self):
        mc = ModelConfig(model_id="test", quantile_levels=[0.1, 0.5, 0.9])
        assert len(mc.quantile_levels) == 3

    def test_quantile_level_out_of_range_rejects(self):
        with pytest.raises(ValidationError):
            ModelConfig(model_id="test", quantile_levels=[1.5])

    def test_quantile_level_negative_rejects(self):
        with pytest.raises(ValidationError):
            ModelConfig(model_id="test", quantile_levels=[-0.1])


@pytest.mark.unit
class TestLocation:
    """Tests for Location model."""

    def test_valid_coordinates(self):
        loc = Location(latitude=48.7758, longitude=9.1829)
        assert loc.latitude == 48.7758
        assert loc.longitude == 9.1829

    def test_boundary_max(self):
        loc = Location(latitude=90, longitude=180)
        assert loc.latitude == 90

    def test_boundary_min(self):
        loc = Location(latitude=-90, longitude=-180)
        assert loc.latitude == -90

    def test_origin(self):
        loc = Location(latitude=0, longitude=0)
        assert loc.latitude == 0

    def test_latitude_above_90_rejects(self):
        with pytest.raises(ValidationError):
            Location(latitude=91, longitude=0)

    def test_latitude_below_minus_90_rejects(self):
        with pytest.raises(ValidationError):
            Location(latitude=-91, longitude=0)

    def test_longitude_above_180_rejects(self):
        with pytest.raises(ValidationError):
            Location(latitude=0, longitude=181)

    def test_longitude_below_minus_180_rejects(self):
        with pytest.raises(ValidationError):
            Location(latitude=0, longitude=-181)


@pytest.mark.unit
class TestTrainingTargetSeries:
    """Tests for TrainingTargetSeries model."""

    def test_valid_series(self):
        ts = TrainingTargetSeries(
            series_id="s1",
            variable_name="power",
            training_values=[
                TimeValue(time=datetime(2023, 1, 1), value=1.0),
                TimeValue(time=datetime(2023, 1, 2), value=2.0),
            ],
        )
        assert len(ts.training_values) == 2

    def test_single_value_rejects(self):
        with pytest.raises(ValidationError):
            TrainingTargetSeries(
                series_id="s1",
                variable_name="power",
                training_values=[TimeValue(time=datetime(2023, 1, 1), value=1.0)],
            )

    def test_empty_values_rejects(self):
        with pytest.raises(ValidationError):
            TrainingTargetSeries(series_id="s1", variable_name="power", training_values=[])

    def test_extra_fields_allowed(self):
        ts = TrainingTargetSeries(
            series_id="s1",
            variable_name="power",
            training_values=[
                TimeValue(time=datetime(2023, 1, 1), value=1.0),
                TimeValue(time=datetime(2023, 1, 2), value=2.0),
            ],
            custom_extra="extra",
        )
        assert ts.custom_extra == "extra"


@pytest.mark.unit
class TestForecastTargetSeries:
    """Tests for ForecastTargetSeries model."""

    def test_valid_with_one_value(self):
        fts = ForecastTargetSeries(
            series_id="s1",
            variable_name="power",
            historical_values=[TimeValue(time=datetime(2023, 1, 1), value=1.0)],
        )
        assert len(fts.historical_values) == 1

    def test_empty_historical_rejects(self):
        with pytest.raises(ValidationError):
            ForecastTargetSeries(series_id="s1", variable_name="power", historical_values=[])


@pytest.mark.unit
class TestTrainingExogenousSeries:
    """Tests for TrainingExogenousSeries model."""

    def test_default_not_known_covariate(self):
        es = TrainingExogenousSeries(
            series_id="e1",
            variable_name="temp",
            training_values=[
                TimeValue(time=datetime(2023, 1, 1), value=20.0),
                TimeValue(time=datetime(2023, 1, 2), value=21.0),
            ],
        )
        assert es.is_known_covariate is False

    def test_known_covariate_flag(self):
        es = TrainingExogenousSeries(
            series_id="e1",
            variable_name="temp",
            is_known_covariate=True,
            training_values=[
                TimeValue(time=datetime(2023, 1, 1), value=20.0),
                TimeValue(time=datetime(2023, 1, 2), value=21.0),
            ],
        )
        assert es.is_known_covariate is True

    def test_none_training_values_allowed(self):
        es = TrainingExogenousSeries(series_id="e1", variable_name="temp", training_values=None)
        assert es.training_values is None


@pytest.mark.unit
class TestForecastExogenousSeries:
    """Tests for ForecastExogenousSeries model."""

    def test_without_future_values(self):
        fes = ForecastExogenousSeries(
            series_id="e1",
            variable_name="temp",
            historical_values=[TimeValue(time=datetime(2023, 1, 1), value=20.0)],
        )
        assert fes.future_values is None

    def test_with_future_values(self):
        fes = ForecastExogenousSeries(
            series_id="e1",
            variable_name="temp",
            historical_values=[TimeValue(time=datetime(2023, 1, 1), value=20.0)],
            future_values=[TimeValue(time=datetime(2023, 1, 2), value=22.0)],
        )
        assert len(fes.future_values) == 1


@pytest.mark.unit
class TestForecastTrainingInput:
    """Tests for ForecastTrainingInput model."""

    def _make_minimal_input(self, **overrides):
        base = {
            "forecast_type": "stochastic",
            "time_parameters": {"horizon_hours": 24},
            "training_model_config": {"model_id": "test-model"},
            "target_series": {
                "series_id": "s1",
                "variable_name": "power",
                "training_values": [
                    {"time": "2023-01-01T00:00:00", "value": 1.0},
                    {"time": "2023-01-01T01:00:00", "value": 2.0},
                ],
            },
        }
        base.update(overrides)
        return base

    def test_valid_minimal(self):
        data = self._make_minimal_input()
        fti = ForecastTrainingInput(**data)
        assert fti.forecast_type == "stochastic"
        assert fti.training_model_config.model_id == "test-model"

    def test_model_config_alias(self):
        """model_config is aliased to training_model_config."""
        data = {
            "forecast_type": "deterministic",
            "time_parameters": {"horizon_hours": 24},
            "model_config": {"model_id": "alias-test"},
            "target_series": {
                "series_id": "s1",
                "variable_name": "power",
                "training_values": [
                    {"time": "2023-01-01T00:00:00", "value": 1.0},
                    {"time": "2023-01-01T01:00:00", "value": 2.0},
                ],
            },
        }
        fti = ForecastTrainingInput(**data)
        assert fti.training_model_config.model_id == "alias-test"

    def test_deterministic_forecast_type(self):
        data = self._make_minimal_input(forecast_type="deterministic")
        fti = ForecastTrainingInput(**data)
        assert fti.forecast_type == "deterministic"

    def test_invalid_forecast_type_rejects(self):
        data = self._make_minimal_input(forecast_type="invalid")
        with pytest.raises(ValidationError):
            ForecastTrainingInput(**data)

    def test_with_location(self):
        data = self._make_minimal_input(location={"latitude": 48.7758, "longitude": 9.1829})
        fti = ForecastTrainingInput(**data)
        assert fti.location.latitude == 48.7758

    def test_without_location(self):
        data = self._make_minimal_input()
        fti = ForecastTrainingInput(**data)
        assert fti.location is None

    def test_with_exogenous_series(self):
        data = self._make_minimal_input(
            exogenous_series=[
                {
                    "series_id": "e1",
                    "variable_name": "temp",
                    "training_values": [
                        {"time": "2023-01-01T00:00:00", "value": 20.0},
                        {"time": "2023-01-01T01:00:00", "value": 21.0},
                    ],
                }
            ]
        )
        fti = ForecastTrainingInput(**data)
        assert len(fti.exogenous_series) == 1

    def test_extra_fields_allowed(self):
        data = self._make_minimal_input(features=["temporal", "lag"])
        fti = ForecastTrainingInput(**data)
        assert fti.features == ["temporal", "lag"]

    def test_populate_by_name(self):
        """Can use either alias or field name."""
        data = self._make_minimal_input()
        fti = ForecastTrainingInput(**data)
        assert fti.training_model_config is not None


@pytest.mark.unit
class TestForecastInput:
    """Tests for ForecastInput model."""

    def test_valid_minimal(self):
        fi = ForecastInput(
            forecast_type="stochastic",
            time_parameters={"horizon_hours": 24},
            target_series={
                "series_id": "s1",
                "variable_name": "power",
                "historical_values": [{"time": "2023-01-01T00:00:00", "value": 1.0}],
            },
        )
        assert fi.model_id is None
        assert fi.model_version is None

    def test_with_model_id(self):
        fi = ForecastInput(
            forecast_type="deterministic",
            time_parameters={"horizon_hours": 24},
            model_id="my-model",
            model_version="1.0.0",
            target_series={
                "series_id": "s1",
                "variable_name": "power",
                "historical_values": [{"time": "2023-01-01T00:00:00", "value": 1.0}],
            },
        )
        assert fi.model_id == "my-model"
        assert fi.model_version == "1.0.0"


@pytest.mark.unit
class TestForecast:
    """Tests for Forecast output model."""

    def test_valid_forecast(self):
        forecast = Forecast(
            forecast_type="stochastic",
            forecast_date=datetime.now(UTC),
            forecast_series=ForecastSeries(
                series_id="s1",
                variable_name="power",
                values=[ForecastedTimeValue(time=datetime(2023, 1, 1), value=1.0)],
            ),
        )
        assert forecast.forecast_type == "stochastic"
        assert len(forecast.forecast_series.values) == 1

    def test_empty_forecast_values_rejects(self):
        with pytest.raises(ValidationError):
            Forecast(
                forecast_type="stochastic",
                forecast_date=datetime.now(UTC),
                forecast_series=ForecastSeries(
                    series_id="s1",
                    variable_name="power",
                    values=[],
                ),
            )

    def test_serialization_round_trip(self):
        forecast = Forecast(
            forecast_type="deterministic",
            forecast_date=datetime(2023, 6, 15, 12, 0, 0, tzinfo=UTC),
            forecast_series=ForecastSeries(
                series_id="s1",
                variable_name="power",
                values=[
                    ForecastedTimeValue(
                        time=datetime(2023, 6, 15, 13, 0, 0),
                        value=42.0,
                        quantiles={"0.5": 42.0},
                    )
                ],
            ),
        )
        json_str = forecast.model_dump_json()
        restored = Forecast.model_validate_json(json_str)
        assert restored.forecast_type == "deterministic"
        assert restored.forecast_series.values[0].value == 42.0
        assert restored.forecast_series.values[0].quantiles["0.5"] == 42.0
