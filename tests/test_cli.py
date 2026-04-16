"""Tests for cli.py module.

Covers the CLI entry points: train_command, predict_command, and sanity_check routing.
JSON loading, Pydantic validation, and file I/O are tested; the underlying train/predict
functions are mocked to avoid real model training.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from u2.cli import main, predict_command, train_command
from u2.data_models import ForecastTrainingInput


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


@pytest.mark.unit
class TestTrainCommand:
    """Tests for train_command CLI entry point."""

    @patch("u2.cli.train")
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

    @patch("u2.cli.predict")
    def test_calls_predict_and_writes_output(self, mock_predict, tmp_path):
        """predict_command should call predict() and write forecast JSON to disk."""
        mock_forecast = MagicMock()
        mock_forecast.forecast_date = "2023-01-09T00:00:00"
        mock_forecast.model_dump_json.return_value = '{"forecast": "data"}'
        mock_predict.return_value = mock_forecast

        path = _make_forecast_json(tmp_path)
        output_dir = str(tmp_path / "predictions")

        with patch("u2.cli.OUTPUT_DIR", output_dir):
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

    @patch("u2.cli.Fire")
    def test_main_registers_all_commands(self, mock_fire):
        """main should dispatch train, predict, and sanity_check via Fire."""
        main()

        mock_fire.assert_called_once()
        commands = mock_fire.call_args[0][0]
        assert set(commands.keys()) == {"sanity_check", "predict", "train"}
        assert commands["train"] is train_command
        assert commands["predict"] is predict_command
