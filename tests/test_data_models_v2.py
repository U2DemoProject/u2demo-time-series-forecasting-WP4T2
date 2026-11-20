"""Tests for u2demo_time_series_forecasting.data_models_v2 module.

Covers validation of the v2 request models (required fields, vocabulary, units, closed models), their conversion to
the v1 data models, the conversion of v1 forecasts to the v2 payload, and conformance with the vendored v2 schemas.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import AnyUrl, ValidationError

from tests.conftest import make_v2_forecast_payload, make_v2_training_payload, v2_schema, v2_validator
from u2demo_time_series_forecasting.data_models import (
    Forecast,
    ForecastedTimeValue,
    ForecastSeries,
    ForecastTrainingInput,
    TrainingSplit,
)
from u2demo_time_series_forecasting.data_models_v2 import (
    QUDT_TO_UNIT_SYMBOL,
    UNIT_SYMBOL_TO_QUDT,
    BaseTargetSeriesV2,
    ForecastedValueV2,
    ForecastExogenousSeriesV2,
    ForecastInputV2,
    ForecastSeriesV2,
    ForecastTargetSeriesV2,
    ForecastTrainingInputV2,
    ForecastTypeV2,
    ForecastV2,
    ObservedProperty,
    QuantityValueV2,
    TimeParametersV2,
    TrainingExogenousSeriesV2,
    UserForecastPreferencesV2,
    forecast_to_v2,
)
from u2demo_time_series_forecasting.features.names import FeatureName

FORECAST_PAYLOAD_SCHEMA = "api/components/forecast/ForecastPayloadDM.json"


def _training_input(**kwargs):
    """Validate a ForecastTrainingInputV2 built by make_v2_training_payload."""
    return ForecastTrainingInputV2.model_validate(make_v2_training_payload(**kwargs))


def _forecast_input(**kwargs):
    """Validate a ForecastInputV2 built by make_v2_forecast_payload."""
    return ForecastInputV2.model_validate(make_v2_forecast_payload(**kwargs))


def _target_series(**overrides):
    """Build a v2 forecast target series payload; a None override removes the field."""
    series = {**make_v2_forecast_payload(periods=1)["target_series"], **overrides}
    return {key: value for key, value in series.items() if value is not None}


def _exogenous_series(**overrides):
    """Build a v2 forecast exogenous series payload; a None override removes the field."""
    series = {
        "series_id": "e1",
        "observedProperty": "OutdoorTemperature",
        "historical_values": [{"timestamp": "2023-01-01T00:00:00Z", "value": 12.0}],
        **overrides,
    }
    return {key: value for key, value in series.items() if value is not None}


def _training_values(count=2):
    return [{"timestamp": f"2023-01-01T{hour:02d}:00:00Z", "value": float(hour)} for hour in range(count)]


def _v1_training_payload(**model_config):
    """Build the v1 training payload equivalent to make_v2_training_payload, with the given model_config."""
    return {
        "forecast_type": "stochastic",
        "time_parameters": {"timestep_minutes": 60, "horizon_hours": 24},
        "model_config": {"model_id": "test-model", "model_version": "1.0.0", **model_config},
        "target_series": {
            "series_id": "s1",
            "variable_name": "ActivePower",
            "training_values": [{"time": v["timestamp"], "value": v["value"]} for v in _training_values()],
        },
    }


def _v1_forecast(forecast_type="stochastic", times=None, value=1.0, forecast_date=None):
    """Build a v1 Forecast as returned by predict, with naive timestamps by default."""
    times = times or [datetime(2023, 1, 9, 8) + timedelta(hours=i) for i in range(3)]
    return Forecast(
        forecast_type=forecast_type,
        forecast_date=forecast_date or datetime(2023, 1, 9, 7, 30, tzinfo=UTC),
        time_parameters={"timestep_minutes": 60, "horizon_hours": 3},
        forecast_series=ForecastSeries(
            series_id="s1",
            variable_name="ActivePower",
            asset_id="pv1",
            asset_type="PV",
            unit="kW",
            values=[
                ForecastedTimeValue(time=time, value=value, quantiles={"0.1": 0.5, "0.5": 1.0, "0.9": 1.5})
                for time in times
            ],
        ),
    )


@pytest.mark.unit
class TestVocabulary:
    """The v2 enums, unit table and field names follow the vendored v2 schemas."""

    def test_observed_property_matches_schema(self):
        schema = v2_schema("forecast/ForecastSeriesDM.json")
        assert [p.value for p in ObservedProperty] == schema["properties"]["observedProperty"]["enum"]

    def test_forecast_type_matches_schema(self):
        schema = v2_schema("spatial/UserForecastPreferencesDM.json")
        assert [t.value for t in ForecastTypeV2] == schema["properties"]["forecast_type"]["enum"]

    def test_unit_table_is_one_to_one(self):
        assert len(set(UNIT_SYMBOL_TO_QUDT.values())) == len(UNIT_SYMBOL_TO_QUDT)
        assert {symbol: iri for iri, symbol in QUDT_TO_UNIT_SYMBOL.items()} == UNIT_SYMBOL_TO_QUDT

    def test_time_parameters_fields_are_time_parameters_dm_properties(self):
        properties = v2_schema("spatial/TimeParametersDM.json")["properties"]
        assert set(TimeParametersV2.model_fields) <= set(properties)

    def test_preferences_fields_are_user_forecast_preferences_dm_properties_or_extensions(self):
        properties = v2_schema("spatial/UserForecastPreferencesDM.json")["properties"]
        extensions = {"model_id", "training_split", "excluded_model_types"}
        assert set(UserForecastPreferencesV2.model_fields) - extensions <= set(properties)

    def test_target_series_fields_are_forecast_series_dm_properties(self):
        properties = v2_schema("forecast/ForecastSeriesDM.json")["properties"]
        assert set(BaseTargetSeriesV2.model_fields) <= set(properties)


@pytest.mark.unit
class TestTargetSeriesV2:
    """Validation of the target series of v2 requests."""

    def test_valid_series(self):
        series = ForecastTargetSeriesV2.model_validate(_target_series(asset_type="Meter"))
        assert series.observedProperty is ObservedProperty.ACTIVE_POWER
        assert series.hasFeatureOfInterest == "pv1"
        assert series.asset_type == "Meter"

    @pytest.mark.parametrize("field", ["series_id", "observedProperty", "hasFeatureOfInterest", "historical_values"])
    def test_required_fields(self, field):
        payload = _target_series()
        del payload[field]
        with pytest.raises(ValidationError, match=field):
            ForecastTargetSeriesV2.model_validate(payload)

    @pytest.mark.parametrize("observed_property", ["activePower", "energyImportTotal", "DayAheadPrice"])
    def test_unknown_observed_property_rejected(self, observed_property):
        with pytest.raises(ValidationError, match="observedProperty"):
            ForecastTargetSeriesV2.model_validate(_target_series(observedProperty=observed_property))

    def test_empty_feature_of_interest_rejected(self):
        with pytest.raises(ValidationError, match="hasFeatureOfInterest"):
            ForecastTargetSeriesV2.model_validate(_target_series(hasFeatureOfInterest=""))

    @pytest.mark.parametrize(
        "extra", [{"variable_name": "power"}, {"asset_id": "pv1"}, {"unitIri": "http://qudt.org/vocab/unit/KiloW"}]
    )
    def test_extra_fields_rejected(self, extra):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ForecastTargetSeriesV2.model_validate(_target_series(**extra))

    def test_value_without_timestamp_rejected(self):
        values = [{"time": "2023-01-01T00:00:00Z", "value": 1.0}]
        with pytest.raises(ValidationError, match="timestamp"):
            ForecastTargetSeriesV2.model_validate(_target_series(historical_values=values))

    def test_empty_historical_values_rejected(self):
        with pytest.raises(ValidationError, match="historical_values"):
            ForecastTargetSeriesV2.model_validate(_target_series(historical_values=[]))

    def test_unit_derived_from_symbol(self):
        series = ForecastTargetSeriesV2.model_validate(_target_series(unit_symbol="kWh"))
        assert series.unit == "unit:KiloW-HR"

    def test_symbol_derived_from_unit(self):
        series = ForecastTargetSeriesV2.model_validate(_target_series(unit_symbol=None, unit="unit:DEG_C"))
        assert series.unit_symbol == "°C"

    def test_unit_outside_table_kept_without_symbol(self):
        series = ForecastTargetSeriesV2.model_validate(_target_series(unit_symbol=None, unit="unit:M-PER-SEC"))
        assert series.unit == "unit:M-PER-SEC"
        assert series.unit_symbol is None

    @pytest.mark.parametrize(
        ("unit_symbol", "unit"),
        [
            ("kW", "unit:KiloW"),
            ("kW·h", "unit:KiloW-HR"),
            ("%", "unit:PERCENT_RH"),
            ("kvarh", "unit:KiloV-A_Reactive-HR"),
        ],
    )
    def test_symbol_and_unit_accepted_unless_the_table_maps_them_apart(self, unit_symbol, unit):
        series = ForecastTargetSeriesV2.model_validate(_target_series(unit_symbol=unit_symbol, unit=unit))
        assert (series.unit_symbol, series.unit) == (unit_symbol, unit)

    @pytest.mark.parametrize(("unit_symbol", "unit"), [("kW", "unit:KiloW-HR"), ("MWh", "unit:KiloW-HR")])
    def test_symbol_contradicting_unit_rejected(self, unit_symbol, unit):
        with pytest.raises(ValidationError, match="does not match unit_symbol"):
            ForecastTargetSeriesV2.model_validate(_target_series(unit_symbol=unit_symbol, unit=unit))

    def test_unknown_symbol_without_unit_rejected(self):
        with pytest.raises(ValidationError, match="unit_symbol 'kvarh' is not one of .*set unit"):
            ForecastTargetSeriesV2.model_validate(_target_series(unit_symbol="kvarh"))

    def test_missing_unit_rejected(self):
        with pytest.raises(ValidationError) as excinfo:
            ForecastTargetSeriesV2.model_validate(_target_series(unit_symbol=None))
        assert [(error["loc"], error["type"]) for error in excinfo.value.errors()] == [(("unit",), "missing")]

    @pytest.mark.parametrize("unit", ["kWh", "KiloW-HR", "http://qudt.org/vocab/unit/KiloW-HR", "unit:"])
    def test_unit_not_in_qudt_form_rejected(self, unit):
        with pytest.raises(ValidationError, match="not a QUDT unit IRI"):
            ForecastTargetSeriesV2.model_validate(_target_series(unit_symbol=None, unit=unit))

    def test_non_string_symbol_rejected(self):
        with pytest.raises(ValidationError, match="unit_symbol"):
            ForecastTargetSeriesV2.model_validate(_target_series(unit_symbol=["kW"]))

    def test_non_object_series_rejected(self):
        with pytest.raises(ValidationError, match="valid dictionary"):
            ForecastTargetSeriesV2.model_validate(["s1"])


@pytest.mark.unit
class TestExogenousSeriesV2:
    """Validation of the exogenous series of v2 requests."""

    def test_variable_name_defaults_to_observed_property(self):
        series = ForecastExogenousSeriesV2.model_validate(_exogenous_series())
        assert series.variable_name == "OutdoorTemperature"

    def test_variable_name_kept_when_given(self):
        series = ForecastExogenousSeriesV2.model_validate(_exogenous_series(variable_name="temperature"))
        assert series.variable_name == "temperature"
        assert series.observedProperty is ObservedProperty.OUTDOOR_TEMPERATURE

    def test_variable_name_without_observed_property(self):
        series = ForecastExogenousSeriesV2.model_validate(
            _exogenous_series(observedProperty=None, variable_name="spot_price")
        )
        assert series.variable_name == "spot_price"
        assert series.observedProperty is None

    def test_variable_name_or_observed_property_required(self):
        with pytest.raises(ValidationError, match="needs a variable_name or an observedProperty"):
            ForecastExogenousSeriesV2.model_validate(_exogenous_series(observedProperty=None))

    def test_unknown_observed_property_rejected(self):
        with pytest.raises(ValidationError, match="observedProperty"):
            ForecastExogenousSeriesV2.model_validate(_exogenous_series(observedProperty="Temperature"))

    def test_units_optional(self):
        series = ForecastExogenousSeriesV2.model_validate(_exogenous_series())
        assert series.unit is None
        assert series.unit_symbol is None

    def test_unit_derived_from_symbol(self):
        series = ForecastExogenousSeriesV2.model_validate(_exogenous_series(unit_symbol="°C"))
        assert series.unit == "unit:DEG_C"

    def test_unknown_symbol_kept_without_unit(self):
        series = ForecastExogenousSeriesV2.model_validate(_exogenous_series(unit_symbol="W/m²"))
        assert series.unit_symbol == "W/m²"
        assert series.unit is None

    def test_symbol_contradicting_unit_rejected(self):
        with pytest.raises(ValidationError, match="does not match unit_symbol"):
            ForecastExogenousSeriesV2.model_validate(_exogenous_series(unit_symbol="%", unit="unit:DEG_C"))

    def test_future_values_optional(self):
        series = ForecastExogenousSeriesV2.model_validate(_exogenous_series())
        assert series.future_values is None

    def test_training_series_defaults(self):
        series = TrainingExogenousSeriesV2.model_validate(
            {"series_id": "e1", "variable_name": "temperature", "training_values": _training_values()}
        )
        assert series.is_known_covariate is False

    def test_training_series_needs_two_values(self):
        with pytest.raises(ValidationError, match="training_values"):
            TrainingExogenousSeriesV2.model_validate(
                {"series_id": "e1", "variable_name": "temperature", "training_values": _training_values(1)}
            )

    def test_training_series_needs_values(self):
        with pytest.raises(ValidationError, match="training_values"):
            TrainingExogenousSeriesV2.model_validate({"series_id": "e1", "variable_name": "temperature"})

    def test_non_object_series_rejected(self):
        with pytest.raises(ValidationError, match="valid dictionary"):
            ForecastExogenousSeriesV2.model_validate("e1")


@pytest.mark.unit
class TestTrainingInputV2Validation:
    """Validation of v2 training requests."""

    def test_valid_minimal(self):
        data = _training_input()
        assert data.hasUserForecastPreferences.forecast_type is ForecastTypeV2.STOCHASTIC
        assert data.features == []
        assert data.num_weekly_lags == 0

    @pytest.mark.parametrize("field", ["hasTimeParameters", "hasUserForecastPreferences", "target_series"])
    def test_required_fields(self, field):
        payload = make_v2_training_payload()
        del payload[field]
        with pytest.raises(ValidationError, match=field):
            ForecastTrainingInputV2.model_validate(payload)

    @pytest.mark.parametrize("field", ["forecast_type", "model_id"])
    def test_required_preferences(self, field):
        payload = make_v2_training_payload()
        del payload["hasUserForecastPreferences"][field]
        with pytest.raises(ValidationError, match=field):
            ForecastTrainingInputV2.model_validate(payload)

    @pytest.mark.parametrize("forecast_type", ["stochastic", "PROBABILISTIC"])
    def test_forecast_type_outside_v2_enum_rejected(self, forecast_type):
        with pytest.raises(ValidationError, match="forecast_type"):
            _training_input(preferences={"forecast_type": forecast_type})

    @pytest.mark.parametrize(
        "extra",
        [
            {"forecast_type": "STOCHASTIC"},
            {"time_parameters": {"horizon_hours": 24}},
            {"model_config": {"model_id": "test-model"}},
        ],
    )
    def test_top_level_fields_outside_v2_request_rejected(self, extra):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _training_input(**extra)

    @pytest.mark.parametrize(
        ("block", "field"), [("preferences", "eval_metric"), ("has_time_parameters", "horizon_hour")]
    )
    def test_unknown_fields_rejected(self, block, field):
        with pytest.raises(ValidationError, match=f"{field}\n  Extra inputs are not permitted"):
            _training_input(**{block: {field: "x"}})

    def test_platform_record_identifiers_accepted(self):
        data = _training_input(
            has_time_parameters={
                "timeParametersId": "tp-1",
                "hasActor": "actor-1",
                "relatedToOptimizationInstance": "run-1",
            },
            preferences={"ufpId": "ufp-1", "hasActor": "actor-1", "relatedToOptimizationInstance": "run-1"},
        )
        assert data.hasTimeParameters.timeParametersId == "tp-1"
        assert data.hasUserForecastPreferences.ufpId == "ufp-1"

    @pytest.mark.parametrize("ratio", ["training_ratio", "validation_ratio", "test_ratio"])
    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_ratio_outside_unit_interval_rejected(self, ratio, value):
        with pytest.raises(ValidationError, match=ratio):
            _training_input(preferences={ratio: value})

    def test_ratios_adding_up_to_more_than_one_rejected(self):
        with pytest.raises(ValidationError, match="add up to 1.2, more than 1"):
            _training_input(preferences={"validation_ratio": 0.6, "test_ratio": 0.6})

    def test_ratios_adding_up_to_one_accepted(self):
        data = _training_input(preferences={"training_ratio": 0.7, "validation_ratio": 0.15, "test_ratio": 0.15})
        assert data.hasUserForecastPreferences.test_ratio == 0.15

    def test_zero_windows_rejected(self):
        with pytest.raises(ValidationError, match="num_val_windows"):
            _training_input(preferences={"training_split": {"num_val_windows": 0}})

    def test_unknown_feature_rejected(self):
        with pytest.raises(ValidationError, match="features"):
            _training_input(features=["calendar"])

    def test_ratio_without_timestep_rejected(self):
        with pytest.raises(ValidationError, match="timestep_minutes is required"):
            _training_input(preferences={"validation_ratio": 0.2}, has_time_parameters={"timestep_minutes": None})

    def test_ratio_without_timestep_accepted_when_windows_given(self):
        data = _training_input(
            preferences={
                "validation_ratio": 0.2,
                "test_ratio": 0.1,
                "training_split": {"num_val_windows": 2, "num_test_windows": 1},
            },
            has_time_parameters={"timestep_minutes": None},
        )
        assert data.hasTimeParameters.timestep_minutes is None


@pytest.mark.unit
class TestForecastInputV2Validation:
    """Validation of v2 forecast requests."""

    def test_valid_minimal(self):
        data = _forecast_input()
        assert data.forecastId is None
        assert data.forecast_type is ForecastTypeV2.STOCHASTIC
        assert data.location is None

    def test_forecast_id(self):
        assert _forecast_input(forecastId="forecast-1").forecastId == "forecast-1"

    @pytest.mark.parametrize("field", ["forecast_type", "hasTimeParameters", "target_series"])
    def test_required_fields(self, field):
        payload = make_v2_forecast_payload()
        del payload[field]
        with pytest.raises(ValidationError, match=field):
            ForecastInputV2.model_validate(payload)

    @pytest.mark.parametrize("extra", [{"time_parameters": {"horizon_hours": 24}}, {"features": ["temporal"]}])
    def test_fields_outside_v2_request_rejected(self, extra):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _forecast_input(**extra)

    def test_location_extra_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _forecast_input(location={"latitude": 52.07, "longitude": 4.3, "altitude": 10})

    def test_location_out_of_range_rejected(self):
        with pytest.raises(ValidationError, match="latitude"):
            _forecast_input(location={"latitude": 91, "longitude": 0})


@pytest.mark.unit
class TestTrainingInputToV1:
    """Conversion of v2 training requests to the v1 ForecastTrainingInput."""

    def test_maps_request_fields(self):
        data = _training_input(
            preferences={
                "forecast_type": "DETERMINISTIC",
                "lookback_steps": 48,
                "quantile_levels": [0.1, 0.9],
                "excluded_model_types": ["DeepAR"],
            },
            location={"latitude": 52.07, "longitude": 4.3},
            features=["temporal", "lag"],
            num_weekly_lags=2,
        )

        v1 = data.to_v1()

        assert v1.forecast_type == "deterministic"
        assert v1.time_parameters.model_dump() == {
            "timestep_minutes": 60,
            "horizon_hours": 24,
            "scheduling_time": None,
        }
        config = v1.training_model_config
        assert config.model_id == "test-model"
        assert config.model_version == "1.0.0"
        assert config.model_preset == "fast_training"
        assert config.lookback_steps == 48
        assert config.quantile_levels == [0.1, 0.9]
        assert config.excluded_model_types == ["DeepAR"]
        assert v1.location is not None
        assert (v1.location.latitude, v1.location.longitude) == (52.07, 4.3)
        assert v1.features == [FeatureName.TEMPORAL, FeatureName.LAG]
        assert v1.num_weekly_lags == 2

    def test_maps_target_series(self):
        target = _training_input().to_v1().target_series

        assert target.series_id == "s1"
        assert target.variable_name == "ActivePower"
        assert target.asset_id == "pv1"
        assert target.asset_type == "PV"
        assert target.unit == "kW"
        assert target.unitIri == AnyUrl("http://qudt.org/vocab/unit/KiloW")
        assert len(target.training_values) == 200
        assert target.training_values[1].time == datetime(2023, 1, 1, 1)
        assert target.training_values[1].value == 1.0

    def test_offsets_dropped_keeping_wall_clock_time(self):
        values = [{"timestamp": f"2023-01-01T{hour:02d}:00:00+02:00", "value": 1.0} for hour in range(3)]

        target = _training_input(target_series={"training_values": values}).to_v1().target_series

        assert [v.time for v in target.training_values] == [datetime(2023, 1, 1, hour) for hour in range(3)]

    def test_platform_record_identifiers_not_passed_on(self):
        data = _training_input(
            has_time_parameters={"timeParametersId": "tp-1", "hasActor": "actor-1"},
            preferences={"ufpId": "ufp-1", "hasActor": "actor-1", "relatedToOptimizationInstance": "run-1"},
        )

        v1 = data.to_v1()

        assert set(v1.time_parameters.model_dump()) == {"timestep_minutes", "horizon_hours", "scheduling_time"}
        assert not v1.training_model_config.model_extra

    def test_preferences_left_out_take_model_config_defaults(self):
        payload = make_v2_training_payload()
        del payload["hasUserForecastPreferences"]["model_preset"]

        config = ForecastTrainingInputV2.model_validate(payload).to_v1().training_model_config
        v1_config = ForecastTrainingInput.model_validate(_v1_training_payload()).training_model_config

        assert config == v1_config
        assert config.model_preset == "chronos2"
        assert config.model_fields_set == v1_config.model_fields_set | {"training_split"}

    def test_explicit_nulls_behave_as_in_v1(self):
        nulls = {"model_preset": None, "excluded_model_types": None, "lookback_steps": None}

        config = _training_input(preferences={**nulls, "loss_function": None}).to_v1().training_model_config
        v1_config = ForecastTrainingInput.model_validate(
            _v1_training_payload(eval_metric=None, **nulls)
        ).training_model_config

        assert config == v1_config
        assert config.model_preset is None
        assert "excluded_model_types" in config.model_fields_set

    @pytest.mark.parametrize(
        ("loss_function", "eval_metric"),
        [
            ("mse", "MSE"),
            ("mae", "MAE"),
            ("pinball", "WQL"),
            ("Pinball", "WQL"),
            ("rmse", "RMSE"),
            ("mape", "MAPE"),
            ("mase", "MASE"),
            ("wql", "WQL"),
            ("sql", "SQL"),
        ],
    )
    def test_loss_function_becomes_eval_metric(self, loss_function, eval_metric):
        data = _training_input(preferences={"loss_function": loss_function})
        assert data.to_v1().training_model_config.eval_metric == eval_metric

    def test_windows_used_as_given(self):
        data = _training_input(preferences={"training_split": {"num_val_windows": 3, "num_test_windows": 2}})
        assert data.to_v1().training_model_config.training_split == TrainingSplit(num_val_windows=3, num_test_windows=2)

    def test_ratios_converted_to_windows(self):
        # 480 hourly steps and a 24-step horizon: 0.25 -> 5 windows, 0.125 -> floor(2.5) = 2 windows
        data = _training_input(periods=480, preferences={"validation_ratio": 0.25, "test_ratio": 0.125})
        assert data.to_v1().training_model_config.training_split == TrainingSplit(num_val_windows=5, num_test_windows=2)

    def test_ratio_conversion_uses_timestep(self):
        # 960 quarter-hour steps and a 96-step horizon: floor(0.25 * 960 / 96) = 2
        data = _training_input(periods=960, timestep_minutes=15, preferences={"validation_ratio": 0.25})
        assert data.to_v1().training_model_config.training_split.num_val_windows == 2

    def test_ratio_conversion_not_lowered_by_float_error(self):
        # 0.35 * 1440 / 24 is 21, computed as 20.999999999999996
        data = _training_input(periods=1440, preferences={"validation_ratio": 0.35})
        assert data.to_v1().training_model_config.training_split.num_val_windows == 21

    @pytest.mark.parametrize("ratio", [0.0, 0.01])
    def test_ratio_converted_to_at_least_one_window(self, ratio):
        data = _training_input(periods=480, preferences={"validation_ratio": ratio, "test_ratio": ratio})
        assert data.to_v1().training_model_config.training_split == TrainingSplit(num_val_windows=1, num_test_windows=1)

    def test_windows_take_precedence_over_ratios(self):
        data = _training_input(
            periods=480,
            preferences={
                "validation_ratio": 0.25,
                "test_ratio": 0.25,
                "training_ratio": 0.5,
                "training_split": {"num_test_windows": 1},
            },
        )
        assert data.to_v1().training_model_config.training_split == TrainingSplit(num_val_windows=5, num_test_windows=1)

    def test_exogenous_series(self):
        exogenous = [
            {
                "series_id": "e1",
                "observedProperty": "OutdoorTemperature",
                "unit_symbol": "°C",
                "source": "station",
                "is_known_covariate": True,
                "training_values": _training_values(),
            },
            {"series_id": "e2", "variable_name": "spot_price", "training_values": _training_values()},
        ]

        v1 = _training_input(exogenous_series=exogenous).to_v1().exogenous_series

        assert v1 is not None
        temperature, price = v1
        assert temperature.variable_name == "OutdoorTemperature"
        assert temperature.is_known_covariate is True
        assert temperature.source == "station"
        assert temperature.unit == "°C"
        assert temperature.unitIri == AnyUrl("http://qudt.org/vocab/unit/DEG_C")
        assert temperature.training_values is not None
        assert temperature.training_values[0].time == datetime(2023, 1, 1)
        assert price.variable_name == "spot_price"
        assert price.unitIri is None
        assert price.is_known_covariate is False

    def test_without_optional_parts(self):
        v1 = _training_input().to_v1()
        assert v1.location is None
        assert v1.exogenous_series is None


@pytest.mark.unit
class TestForecastInputToV1:
    """Conversion of v2 forecast requests to the v1 ForecastInput."""

    def test_maps_request_fields(self):
        data = _forecast_input(forecast_type="DETERMINISTIC", location={"latitude": 52.07, "longitude": 4.3})

        v1 = data.to_v1()

        assert v1.forecast_type == "deterministic"
        assert v1.model_id == "test-model"
        assert v1.model_version == "1.0.0"
        assert v1.time_parameters.horizon_hours == 24
        assert v1.time_parameters.timestep_minutes == 60
        assert v1.location is not None
        assert v1.location.latitude == 52.07
        assert v1.target_series.variable_name == "ActivePower"
        assert v1.target_series.asset_id == "pv1"
        assert v1.target_series.unit == "kW"
        assert v1.target_series.unitIri == AnyUrl("http://qudt.org/vocab/unit/KiloW")
        assert len(v1.target_series.historical_values) == 200
        assert v1.target_series.historical_values[0].time == datetime(2023, 1, 1)
        assert v1.exogenous_series is None

    def test_exogenous_series(self):
        future = [{"timestamp": "2023-01-01T01:00:00Z", "value": 13.0}]
        data = _forecast_input(
            exogenous_series=[
                _exogenous_series(unit="unit:DEG_C", future_values=future),
                _exogenous_series(series_id="e2", observedProperty=None, variable_name="spot_price"),
            ]
        )

        v1 = data.to_v1().exogenous_series

        assert v1 is not None
        temperature, price = v1
        assert temperature.variable_name == "OutdoorTemperature"
        assert temperature.unit == "°C"
        assert temperature.unitIri == AnyUrl("http://qudt.org/vocab/unit/DEG_C")
        assert temperature.historical_values[0].value == 12.0
        assert temperature.future_values is not None
        assert temperature.future_values[0].time == datetime(2023, 1, 1, 1)
        assert price.variable_name == "spot_price"
        assert price.future_values is None
        assert price.unitIri is None


@pytest.mark.unit
class TestForecastToV2:
    """Conversion of v1 forecasts to the v2 payload."""

    def _request(self, last_timestamp=None, **kwargs):
        payload = make_v2_forecast_payload(**kwargs)
        if last_timestamp is not None:
            payload["target_series"]["historical_values"][-1]["timestamp"] = last_timestamp
        return ForecastInputV2.model_validate(payload)

    def test_payload_fields(self):
        result = forecast_to_v2(_v1_forecast(), self._request(forecastId="forecast-1"))

        assert isinstance(result, ForecastV2)
        assert result.forecastId == "forecast-1"
        assert result.forecast_date == datetime(2023, 1, 9, 7, 30, tzinfo=UTC)
        assert result.forecast_type is ForecastTypeV2.STOCHASTIC
        assert len(result.hasForecastSeries) == 1

    def test_series_identity_from_request(self):
        request = self._request(target_series={"hasFeatureOfInterest": "mp-7"})

        series = forecast_to_v2(_v1_forecast(), request).hasForecastSeries[0]

        assert series.series_id == "s1"
        assert series.observedProperty is ObservedProperty.ACTIVE_POWER
        assert series.hasFeatureOfInterest == "mp-7"
        assert series.unit_symbol == "kW"
        assert series.unit == "unit:KiloW"
        assert [v.hasResult for v in series.hasValues] == [QuantityValueV2(numericValue=1.0, unit="unit:KiloW")] * 3

    def test_model_id_is_model_directory_name(self):
        series = forecast_to_v2(_v1_forecast(), self._request(model_id="load", model_version="2.1.0"))
        assert series.hasForecastSeries[0].model_id == "load_2.1.0"

    def test_stochastic_keeps_quantiles(self):
        values = forecast_to_v2(_v1_forecast("stochastic"), self._request()).hasForecastSeries[0].hasValues
        assert all(v.quantiles == {"0.1": 0.5, "0.5": 1.0, "0.9": 1.5} for v in values)

    def test_deterministic_drops_quantiles(self):
        result = forecast_to_v2(_v1_forecast("deterministic"), self._request(forecast_type="DETERMINISTIC"))

        assert result.forecast_type is ForecastTypeV2.DETERMINISTIC
        assert all(v.quantiles is None for v in result.hasForecastSeries[0].hasValues)

    def test_asset_type_not_emitted(self):
        payload = forecast_to_v2(_v1_forecast(), self._request()).model_dump()
        assert "asset_type" not in payload["hasForecastSeries"][0]

    def test_forecast_id_generated_when_missing(self):
        request = self._request()

        first = forecast_to_v2(_v1_forecast(), request).forecastId
        second = forecast_to_v2(_v1_forecast(), request).forecastId

        assert uuid.UUID(first).version == 4
        assert first != second

    def test_empty_forecast_id_kept(self):
        assert forecast_to_v2(_v1_forecast(), self._request(forecastId="")).forecastId == ""

    def test_naive_timestamps_read_as_utc_for_naive_request(self):
        request = self._request(last_timestamp="2023-01-08T23:00:00")

        values = forecast_to_v2(_v1_forecast(), request).hasForecastSeries[0].hasValues

        assert values[0].timestamp == datetime(2023, 1, 9, 8, tzinfo=UTC)

    def test_naive_timestamps_take_request_offset(self):
        request = self._request(last_timestamp="2023-01-09T01:00:00+02:00")

        values = forecast_to_v2(_v1_forecast(), request).hasForecastSeries[0].hasValues

        assert values[0].timestamp.utcoffset() == timedelta(hours=2)
        assert values[0].timestamp.replace(tzinfo=None) == datetime(2023, 1, 9, 8)

    def test_aware_timestamps_kept(self):
        times = [datetime(2023, 1, 9, 8, tzinfo=UTC)]
        values = forecast_to_v2(_v1_forecast(times=times), self._request()).hasForecastSeries[0].hasValues
        assert values[0].timestamp == times[0]

    def test_naive_forecast_date_read_as_utc(self):
        forecast = _v1_forecast(forecast_date=datetime(2023, 1, 9, 7, 30))
        assert forecast_to_v2(forecast, self._request()).forecast_date == datetime(2023, 1, 9, 7, 30, tzinfo=UTC)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_value_rejected(self, value):
        with pytest.raises(ValidationError, match="finite number"):
            forecast_to_v2(_v1_forecast(value=value), self._request())


@pytest.mark.unit
class TestForecastV2Model:
    """Validation and serialisation of the v2 forecast payload model."""

    def _series(self, **overrides):
        series = {
            "series_id": "s1",
            "observedProperty": "ActivePower",
            "hasFeatureOfInterest": "pv1",
            "unit": "unit:KiloW",
            "model_id": "m_1.0.0",
            "hasValues": [
                {"timestamp": "2023-01-01T00:00:00Z", "hasResult": {"numericValue": 1, "unit": "unit:KiloW"}}
            ],
        }
        series.update(overrides)
        return series

    def _forecast(self, **series_overrides):
        return ForecastV2.model_validate(
            {
                "forecastId": "f1",
                "forecast_date": "2023-01-01T00:00:00Z",
                "forecast_type": "DETERMINISTIC",
                "hasForecastSeries": [self._series(**series_overrides)],
            }
        )

    def test_valid_payload(self):
        assert self._forecast().hasForecastSeries[0].hasValues[0].hasResult.numericValue == 1.0

    def test_optional_fields_without_value_left_out_of_dumps(self):
        forecast = self._forecast()

        series = forecast.model_dump()["hasForecastSeries"][0]

        assert "unit_symbol" not in series
        assert "quantiles" not in series["hasValues"][0]
        assert "null" not in forecast.model_dump_json()

    def test_series_extra_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ForecastSeriesV2.model_validate(self._series(asset_type="PV"))

    def test_empty_values_rejected(self):
        with pytest.raises(ValidationError, match="hasValues"):
            ForecastSeriesV2.model_validate(self._series(hasValues=[]))

    def test_empty_series_rejected(self):
        with pytest.raises(ValidationError, match="hasForecastSeries"):
            ForecastV2(
                forecastId="f1",
                forecast_date=datetime(2023, 1, 1, tzinfo=UTC),
                forecast_type=ForecastTypeV2.STOCHASTIC,
                hasForecastSeries=[],
            )

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValidationError, match="timezone"):
            ForecastedValueV2(timestamp=datetime(2023, 1, 1), hasResult=QuantityValueV2(numericValue=1, unit="unit:W"))

    def test_naive_forecast_date_rejected(self):
        with pytest.raises(ValidationError, match="timezone"):
            ForecastV2(
                forecastId="f1",
                forecast_date=datetime(2023, 1, 1),
                forecast_type=ForecastTypeV2.STOCHASTIC,
                hasForecastSeries=[ForecastSeriesV2.model_validate(self._series())],
            )


@pytest.mark.unit
class TestSchemaConformance:
    """forecast_to_v2 output validates against the vendored ForecastPayloadDM schema."""

    def _forecast(self, forecast_type="STOCHASTIC", **target):
        request = ForecastInputV2.model_validate(
            make_v2_forecast_payload(forecast_type=forecast_type, target_series=target)
        )
        return forecast_to_v2(_v1_forecast(forecast_type.lower()), request)

    @pytest.mark.parametrize("forecast_type", ["STOCHASTIC", "DETERMINISTIC"])
    def test_forecast_validates(self, forecast_type):
        v2_validator(FORECAST_PAYLOAD_SCHEMA).validate(self._forecast(forecast_type).model_dump(mode="json"))

    def test_forecast_without_unit_symbol_validates(self):
        payload = self._forecast(unit_symbol=None, unit="unit:M-PER-SEC").model_dump(mode="json")
        assert "unit_symbol" not in payload["hasForecastSeries"][0]
        v2_validator(FORECAST_PAYLOAD_SCHEMA).validate(payload)

    def test_json_dump_validates(self):
        forecast = self._forecast("DETERMINISTIC", unit_symbol=None, unit="unit:M-PER-SEC")
        v2_validator(FORECAST_PAYLOAD_SCHEMA).validate(json.loads(forecast.model_dump_json()))

    def test_validator_checks_timestamp_offsets(self):
        payload = self._forecast().model_dump(mode="json")
        payload["hasForecastSeries"][0]["hasValues"][0]["timestamp"] = "2023-01-09T08:00:00"
        assert not v2_validator(FORECAST_PAYLOAD_SCHEMA).is_valid(payload)

    def test_validator_rejects_asset_type(self):
        payload = self._forecast().model_dump(mode="json")
        payload["hasForecastSeries"][0]["asset_type"] = "PV"
        assert not v2_validator(FORECAST_PAYLOAD_SCHEMA).is_valid(payload)
