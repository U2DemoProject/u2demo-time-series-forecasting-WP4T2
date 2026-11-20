"""Tests for cli.py module.

Covers the CLI entry points: train_command, predict_command, and sanity_check routing.
JSON loading, Pydantic validation, and file I/O are tested; the underlying train/predict
functions are mocked to avoid real model training.
"""

import json
import os
import sys
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from tests.conftest import DATA_DIR, make_mock_model, make_v2_forecast_payload, make_v2_training_payload, v2_validator
from u2demo_time_series_forecasting.cli import main, predict_command, train_command
from u2demo_time_series_forecasting.data_models import ForecastTrainingInput

FORECAST_PAYLOAD_SCHEMA = "api/components/forecast/ForecastPayloadDM.json"

# Key structure of the v1 forecast written by `predict` (nulls included)
V1_FORECAST_SHAPE = {
    "forecast_type": None,
    "forecast_date": None,
    "time_parameters": {"timestep_minutes": None, "horizon_hours": None, "scheduling_time": None},
    "forecast_series": {
        "series_id": None,
        "variable_name": None,
        "asset_id": None,
        "asset_type": None,
        "unit": None,
        "unitIri": None,
        "values": [{"time": None, "value": None, "quantiles": {"0.1": None, "0.5": None, "0.9": None}}],
    },
}


def _make_training_json(tmp_path, periods=200):
    """Write a valid training JSON file and return its path."""
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="h")
    payload = {
        "forecast_type": "stochastic",
        "time_parameters": {"timestep_minutes": 60, "horizon_hours": 24},
        "training_model_config": {
            "model_id": "cli-test",
            "model_preset": "fast_training",
            "model_version": "1.0.0",
            "training_split": {"num_val_windows": 1, "num_test_windows": 1},
        },
        "target_series": {
            "series_id": "s1",
            "variable_name": "active_power",
            "training_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
        },
    }
    path = str(tmp_path / "train_input.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def _make_forecast_json(tmp_path, periods=200):
    """Write a valid forecast JSON file and return its path."""
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="h")
    payload = {
        "forecast_type": "stochastic",
        "time_parameters": {"timestep_minutes": 60, "horizon_hours": 24},
        "model_id": "cli-test",
        "model_version": "1.0.0",
        "target_series": {
            "series_id": "s1",
            "variable_name": "active_power",
            "historical_values": [{"time": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)],
        },
    }
    path = str(tmp_path / "forecast_input.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def _write_payload(tmp_path, payload, name="input.json"):
    """Write a payload to a JSON file and return its path."""
    path = str(tmp_path / name)
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def _key_shape(value):
    """Reduce a decoded JSON document to its key structure, using the first element of lists."""
    if isinstance(value, dict):
        return {key: _key_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_key_shape(value[0])]
    return None


def _read_single_output(output_dir):
    """Return the name and the decoded content of the only file in ``output_dir``."""
    (name,) = os.listdir(output_dir)
    with open(os.path.join(output_dir, name)) as f:
        return name, json.load(f)


@pytest.mark.unit
class TestTrainCommand:
    """Tests for train_command CLI entry point."""

    @patch("u2demo_time_series_forecasting.train.train")
    def test_calls_train_with_validated_input(self, mock_train, tmp_path):
        """train_command should load JSON, validate it, and call train()."""
        path = _make_training_json(tmp_path)
        train_command(path)

        mock_train.assert_called_once()
        arg = mock_train.call_args[0][0]
        assert isinstance(arg, ForecastTrainingInput)

    def test_missing_file_raises(self):
        """Missing JSON file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            train_command("/nonexistent/path.json")

    def test_invalid_json_raises(self, tmp_path):
        """Malformed JSON should raise json.JSONDecodeError."""
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            train_command(path)

    def test_valid_json_bad_schema_raises(self, tmp_path):
        """Valid JSON that doesn't match ForecastTrainingInput schema should raise ValidationError."""
        path = str(tmp_path / "bad_schema.json")
        with open(path, "w") as f:
            json.dump({"foo": "bar"}, f)

        with pytest.raises(ValidationError):
            train_command(path)


@pytest.mark.unit
class TestPredictCommand:
    """Tests for predict_command CLI entry point."""

    @patch("u2demo_time_series_forecasting.cli.predict")
    def test_calls_predict_and_writes_output(self, mock_predict, tmp_path):
        """predict_command should call predict() and write forecast JSON to disk."""
        mock_forecast = MagicMock()
        mock_forecast.forecast_date = datetime(2023, 1, 9)
        mock_forecast.model_dump_json.return_value = '{"forecast": "data"}'
        mock_predict.return_value = mock_forecast

        path = _make_forecast_json(tmp_path)
        output_dir = str(tmp_path / "predictions")

        with patch("u2demo_time_series_forecasting.cli.OUTPUT_DIR", output_dir):
            predict_command(path)

        mock_predict.assert_called_once()
        assert os.path.exists(output_dir)
        output_files = os.listdir(output_dir)
        assert len(output_files) == 1
        assert output_files[0].endswith(".json")

    def test_missing_file_raises(self):
        """Missing JSON file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            predict_command("/nonexistent/path.json")

    def test_invalid_json_raises(self, tmp_path):
        """Malformed JSON should raise json.JSONDecodeError."""
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            predict_command(path)

    def test_valid_json_bad_schema_raises(self, tmp_path):
        """Valid JSON that doesn't match ForecastInput schema should raise ValidationError."""
        path = str(tmp_path / "bad_schema.json")
        with open(path, "w") as f:
            json.dump({"foo": "bar"}, f)

        with pytest.raises(ValidationError):
            predict_command(path)


@pytest.mark.unit
class TestMain:
    """Tests for the `main` entry point registered as the `u2` console script."""

    @patch("u2demo_time_series_forecasting.cli.Fire")
    def test_main_registers_all_commands(self, mock_fire):
        """main should dispatch train, predict, and sanity_check via Fire."""
        main()

        mock_fire.assert_called_once()
        commands = mock_fire.call_args[0][0]
        assert set(commands.keys()) == {"sanity_check", "predict", "train"}
        assert commands["train"] is train_command
        assert commands["predict"] is predict_command


@pytest.mark.unit
class TestTrainCommandVersions:
    """train_command validates the payload in its data model version and trains its v1 equivalent."""

    @pytest.mark.parametrize(
        ("name", "variable_name"), [("dummy_train.json", "energyImportTotal"), ("dummy_train_v2.json", "ActivePower")]
    )
    @patch("u2demo_time_series_forecasting.train.train")
    def test_dummy_version_detected(self, mock_train, name, variable_name):
        train_command(str(DATA_DIR / name))

        data = mock_train.call_args[0][0]
        assert isinstance(data, ForecastTrainingInput)
        assert data.target_series.variable_name == variable_name

    @patch("u2demo_time_series_forecasting.train.train")
    def test_explicit_v2(self, mock_train, tmp_path):
        train_command(_write_payload(tmp_path, make_v2_training_payload()), version="v2")

        assert mock_train.call_args[0][0].target_series.asset_id == "pv1"

    @patch("u2demo_time_series_forecasting.train.train")
    def test_explicit_version_mismatch_raises(self, mock_train, tmp_path):
        with pytest.raises(ValidationError):
            train_command(_write_payload(tmp_path, make_v2_training_payload()), version="v1")
        mock_train.assert_not_called()

    @patch("u2demo_time_series_forecasting.train.train")
    def test_mixed_payload_raises(self, mock_train, tmp_path):
        payload = make_v2_training_payload(time_parameters={"horizon_hours": 24})
        with pytest.raises(ValueError, match="mixes v1 fields"):
            train_command(_write_payload(tmp_path, payload))
        mock_train.assert_not_called()


@pytest.mark.unit
class TestPredictCommandVersions:
    """predict_command writes the forecast in the data model version of the request."""

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_v1_dummy_written_as_v1_forecast(self, mock_model_cls, tmp_path):
        mock_model_cls.return_value = make_mock_model()

        with patch("u2demo_time_series_forecasting.cli.OUTPUT_DIR", str(tmp_path)):
            predict_command(str(DATA_DIR / "dummy_predict.json"))

        name, output = _read_single_output(tmp_path)
        assert name.startswith("dummy_model_1.0.0_")
        assert _key_shape(output) == V1_FORECAST_SHAPE
        assert output["forecast_type"] == "stochastic"
        assert output["forecast_series"]["variable_name"] == "energyImportTotal"
        assert output["forecast_series"]["unitIri"] is None
        assert len(output["forecast_series"]["values"]) == 24

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_v2_dummy_written_as_forecast_payload(self, mock_model_cls, tmp_path):
        mock_model_cls.return_value = make_mock_model()

        with patch("u2demo_time_series_forecasting.cli.OUTPUT_DIR", str(tmp_path)):
            predict_command(str(DATA_DIR / "dummy_predict_v2.json"))

        name, output = _read_single_output(tmp_path)
        assert name.startswith("dummy_pavilion_load_1.0.0_")
        v2_validator(FORECAST_PAYLOAD_SCHEMA).validate(output)
        assert uuid.UUID(output["forecastId"]).version == 4
        assert output["forecast_type"] == "STOCHASTIC"
        series = output["hasForecastSeries"][0]
        assert series["observedProperty"] == "ActivePower"
        assert series["hasFeatureOfInterest"] == "load_pavilion_1"
        assert series["model_id"] == "dummy_pavilion_load_1.0.0"
        assert len(series["hasValues"]) == 24
        assert series["hasValues"][0]["hasResult"] == {"numericValue": 1.0, "unit": "unit:KiloW"}
        assert series["hasValues"][0]["timestamp"].endswith("Z")

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_v2_deterministic_output_has_no_nulls(self, mock_model_cls, tmp_path):
        mock_model_cls.return_value = make_mock_model()
        payload = make_v2_forecast_payload(forecast_type="DETERMINISTIC", target_series={"unit": "unit:M-PER-SEC"})
        payload["target_series"].pop("unit_symbol")
        output_dir = tmp_path / "output"

        with patch("u2demo_time_series_forecasting.cli.OUTPUT_DIR", str(output_dir)):
            predict_command(_write_payload(tmp_path, payload), version="v2")

        _, output = _read_single_output(output_dir)
        v2_validator(FORECAST_PAYLOAD_SCHEMA).validate(output)
        assert "null" not in json.dumps(output)

    def test_explicit_version_mismatch_raises(self, tmp_path):
        with pytest.raises(ValidationError):
            predict_command(_make_forecast_json(tmp_path), version="v2")

    def test_unknown_version_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown payload version"):
            predict_command(_make_forecast_json(tmp_path), version="v3")


@pytest.mark.unit
class TestVersionFlag:
    """`--version` reaches the commands through Fire: a v2 payload sent with --version v1 fails v1 validation."""

    @patch("u2demo_time_series_forecasting.predict.U2TimeSeriesForecastingModel")
    def test_predict_version_flag(self, mock_model_cls, tmp_path, monkeypatch):
        mock_model_cls.return_value = make_mock_model()
        path = _write_payload(tmp_path, make_v2_forecast_payload())
        monkeypatch.setattr(sys, "argv", ["u2demo-forecast", "predict", path, "--version", "v1"])

        with (
            patch("u2demo_time_series_forecasting.cli.OUTPUT_DIR", str(tmp_path / "output")),
            pytest.raises(ValidationError),
        ):
            main()
        mock_model_cls.assert_not_called()

    @patch("u2demo_time_series_forecasting.train.train")
    def test_train_version_flag(self, mock_train, tmp_path, monkeypatch):
        path = _write_payload(tmp_path, make_v2_training_payload())
        monkeypatch.setattr(sys, "argv", ["u2demo-forecast", "train", path, "--version", "v1"])

        with pytest.raises(ValidationError):
            main()
        mock_train.assert_not_called()
