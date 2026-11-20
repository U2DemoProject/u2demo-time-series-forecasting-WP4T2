"""Tests for u2demo_time_series_forecasting.versioning module.

Covers payload version detection (one test per rule, the v1 fallback and payloads mixing both versions) and the
validation of payloads with a detected or an explicit version.
"""

import json
import logging

import pytest
from pydantic import ValidationError

from tests.conftest import DATA_DIR, make_v2_forecast_payload, make_v2_training_payload
from u2demo_time_series_forecasting.data_models import ForecastInput, ForecastTrainingInput
from u2demo_time_series_forecasting.data_models_v2 import ForecastInputV2, ForecastTrainingInputV2
from u2demo_time_series_forecasting.versioning import (
    detect_payload_version,
    parse_forecast_payload,
    parse_training_payload,
)


def _load(name):
    return json.loads((DATA_DIR / name).read_text())


def _v1_forecast_payload():
    return {
        "forecast_type": "stochastic",
        "time_parameters": {"timestep_minutes": 60, "horizon_hours": 24},
        "model_id": "test-model",
        "model_version": "1.0.0",
        "target_series": {
            "series_id": "s1",
            "variable_name": "active_power",
            "historical_values": [{"time": "2023-01-01T00:00:00", "value": 1.0}],
        },
    }


@pytest.mark.unit
class TestDetectPayloadVersion:
    """Tests for detect_payload_version."""

    @pytest.mark.parametrize("key", ["hasTimeParameters", "hasUserForecastPreferences", "forecastId"])
    def test_v2_top_level_key(self, key):
        assert detect_payload_version({key: {}}) == "v2"

    @pytest.mark.parametrize("forecast_type", ["DETERMINISTIC", "STOCHASTIC"])
    def test_upper_case_forecast_type_is_v2(self, forecast_type):
        assert detect_payload_version({"forecast_type": forecast_type}) == "v2"

    @pytest.mark.parametrize("forecast_type", ["deterministic", "stochastic"])
    def test_lower_case_forecast_type_is_v1(self, forecast_type):
        assert detect_payload_version({"forecast_type": forecast_type}) == "v1"

    @pytest.mark.parametrize("key", ["observedProperty", "hasFeatureOfInterest"])
    def test_v2_target_series_key(self, key):
        assert detect_payload_version({"target_series": {"series_id": "s1", key: "x"}}) == "v2"

    @pytest.mark.parametrize("payload", [{}, {"foo": "bar"}, {"forecast_type": "probabilistic"}])
    def test_payload_without_version_fields_is_v1(self, payload):
        assert detect_payload_version(payload) == "v1"

    def test_v1_payload(self):
        assert detect_payload_version(_v1_forecast_payload()) == "v1"

    @pytest.mark.parametrize("payload", [[{"forecast_type": "STOCHASTIC"}], "payload", None])
    def test_payload_that_is_not_an_object_rejected(self, payload):
        with pytest.raises(ValueError, match="must be a JSON object"):
            detect_payload_version(payload)

    def test_target_series_that_is_not_an_object_is_ignored(self):
        assert detect_payload_version({"target_series": [{"observedProperty": "ActivePower"}]}) == "v1"

    @pytest.mark.parametrize(
        ("name", "version"),
        [
            ("dummy_train.json", "v1"),
            ("dummy_predict.json", "v1"),
            ("dummy_train_v2.json", "v2"),
            ("dummy_predict_v2.json", "v2"),
        ],
    )
    def test_dummy_payloads(self, name, version):
        assert detect_payload_version(_load(name)) == version

    @pytest.mark.parametrize(
        ("payload", "v1_field", "v2_field"),
        [
            ({"time_parameters": {}, "hasTimeParameters": {}}, "time_parameters", "hasTimeParameters"),
            ({"model_config": {}, "hasUserForecastPreferences": {}}, "model_config", "hasUserForecastPreferences"),
            ({"forecast_type": "stochastic", "forecastId": "f1"}, "forecast_type=stochastic", "forecastId"),
            (
                {"forecast_type": "STOCHASTIC", "training_model_config": {}},
                "training_model_config",
                "forecast_type=STOCHASTIC",
            ),
            (
                {"target_series": {"variable_name": "power", "observedProperty": "ActivePower"}},
                "target_series.variable_name",
                "target_series.observedProperty",
            ),
            (
                {"target_series": {"asset_id": "pv1", "unitIri": "x", "hasFeatureOfInterest": "pv1"}},
                "target_series.unitIri",
                "target_series.hasFeatureOfInterest",
            ),
        ],
    )
    def test_mixed_payload_rejected(self, payload, v1_field, v2_field):
        with pytest.raises(ValueError, match="mixes v1 fields") as excinfo:
            detect_payload_version(payload)
        assert not isinstance(excinfo.value, ValidationError)
        assert v1_field in str(excinfo.value)
        assert v2_field in str(excinfo.value)

    def test_detected_version_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger="u2demo"):
            detect_payload_version({"hasTimeParameters": {}})
        assert "Detected payload version v2" in caplog.text


@pytest.mark.unit
class TestParsePayload:
    """Tests for parse_training_payload and parse_forecast_payload."""

    def test_v1_training_payload(self):
        assert isinstance(parse_training_payload(_load("dummy_train.json")), ForecastTrainingInput)

    def test_v2_training_payload(self):
        assert isinstance(parse_training_payload(make_v2_training_payload()), ForecastTrainingInputV2)

    def test_v1_forecast_payload(self):
        assert isinstance(parse_forecast_payload(_v1_forecast_payload()), ForecastInput)

    def test_v2_forecast_payload(self):
        assert isinstance(parse_forecast_payload(make_v2_forecast_payload()), ForecastInputV2)

    @pytest.mark.parametrize(
        ("payload", "version", "expected_type"),
        [
            (_v1_forecast_payload(), "v1", ForecastInput),
            (make_v2_forecast_payload(), "v2", ForecastInputV2),
        ],
    )
    def test_explicit_version_matching_payload(self, payload, version, expected_type):
        assert isinstance(parse_forecast_payload(payload, version), expected_type)

    def test_explicit_version_wins_over_detection(self):
        with pytest.raises(ValidationError, match="hasTimeParameters"):
            parse_forecast_payload(_v1_forecast_payload(), "v2")

    def test_explicit_version_skips_mixed_fields_check(self):
        payload = make_v2_training_payload(time_parameters={"horizon_hours": 24})
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            parse_training_payload(payload, "v2")

    def test_explicit_version_not_logged_as_detected(self, caplog):
        with caplog.at_level(logging.INFO, logger="u2demo"):
            parse_forecast_payload(make_v2_forecast_payload(), "v2")
        assert "Detected payload version" not in caplog.text

    @pytest.mark.parametrize("version", ["v3", "V2", "2"])
    def test_unknown_version_rejected(self, version):
        with pytest.raises(ValueError, match="Unknown payload version"):
            parse_training_payload(make_v2_training_payload(), version)

    def test_mixed_payload_without_version_rejected(self):
        payload = make_v2_forecast_payload(time_parameters={"horizon_hours": 24})
        with pytest.raises(ValueError, match="mixes v1 fields"):
            parse_forecast_payload(payload)
