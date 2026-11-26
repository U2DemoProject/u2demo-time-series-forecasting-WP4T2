"""Pytest configuration and shared fixtures."""

import json
import os
import tempfile
from collections.abc import Generator
from functools import cache
from pathlib import Path
from unittest.mock import MagicMock

# CI only: pin HF_HOME to /tmp so model downloads never enter cached paths.
# Must run at import time, before `transformers` / `huggingface_hub` reads it.
# Local runs are left alone to keep reusing the dev's shared HF cache.
if os.environ.get("CI"):
    os.environ.setdefault("HF_HOME", tempfile.mkdtemp(prefix="u2demo-hf-"))

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft7Validator
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource

from u2demo_time_series_forecasting.constants.columns import (
    TIME_SERIES_GROUP_COLUMN_NAME,
    TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)
from u2demo_time_series_forecasting.data_models import ForecastTrainingInput
from u2demo_time_series_forecasting.metadata import ModelMetadata, TrainingMetadata

DATA_DIR = Path(__file__).parent.parent / "data"
V2_SCHEMA_BASE_URI = "https://u2demo.eu/schema/v2.0.0/"
V2_SCHEMA_DIR = Path(__file__).parent / "fixtures" / "u2demo_data_models" / "v2.0.0"

# ── helpers ────────────────────────────────────────────────────────────────────


def _make_training_payload(
    dates: pd.DatetimeIndex,
    group: str,
    timestep_minutes: int = 60,
    include_location: bool = True,
    nan_indices: np.ndarray | None = None,
) -> dict:
    """Build a training JSON payload matching the data model."""
    values = np.random.rand(len(dates)).tolist()
    if nan_indices is not None:
        for i in nan_indices:
            values[i] = None  # JSON null

    scheduling_time = dates[-1].isoformat() + "Z" if len(dates) > 0 else "2023-01-01T00:00:00Z"

    payload: dict = {
        "forecast_type": "stochastic",
        "time_parameters": {
            "timestep_minutes": timestep_minutes,
            "horizon_hours": 24,
            "scheduling_time": scheduling_time,
        },
        "training_model_config": {
            "model_id": f"model-{group}",
            "model_preset": "fast_training",
            "model_version": "1.0.0",
            "lookback_steps": 96,
            "eval_metric": "WQL",
            "quantile_levels": [0.1, 0.5, 0.9],
            "training_split": {
                "num_val_windows": 14,
                "num_test_windows": 7,
            },
        },
        "target_series": {
            "series_id": f"series-{group}",
            "variable_name": "active_power",
            "asset_id": group,
            "asset_type": "PV",
            "unit": "kW",
            "unitIri": "http://qudt.org/vocab/unit/KiloW",
            "training_values": [
                {"time": ts.strftime("%Y-%m-%d %H:%M:%S"), "value": v} for ts, v in zip(dates, values, strict=True)
            ],
        },
        "features": ["weather"],
    }

    if include_location:
        payload["location"] = {"latitude": 48.7758, "longitude": 9.1829}

    return payload


def _write_json(payload: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_training_input_from_json(json_path: str) -> ForecastTrainingInput:
    """
    Load and validate JSON data as a ForecastTrainingInput Pydantic model.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing training input data.

    Returns
    -------
    ForecastTrainingInput
        Validated training input data model.

    """
    with open(json_path) as f:
        raw_data = json.load(f)
    return ForecastTrainingInput(**raw_data)


def v2_schema(path: str) -> dict:
    """
    Load a vendored U2Demo data models v2 schema.

    Parameters
    ----------
    path : str
        Path of the schema under ``v2.0.0/`` (e.g. ``"forecast/ForecastSeriesDM.json"``).

    Returns
    -------
    dict
        The JSON schema.

    """
    return json.loads((V2_SCHEMA_DIR / path).read_text())


def _retrieve_v2_schema(uri: str) -> Resource:
    """Resolve a ``https://u2demo.eu/schema/v2.0.0/<path>`` reference to the vendored schema at ``<path>``."""
    if not uri.startswith(V2_SCHEMA_BASE_URI):
        raise NoSuchResource(ref=uri)
    return Resource.from_contents(v2_schema(uri.removeprefix(V2_SCHEMA_BASE_URI)))


@cache
def v2_validator(path: str) -> Draft7Validator:
    """
    Build a validator for a vendored v2 schema, with ``$ref`` resolved against the vendored schemas.

    ``date-time`` formats are checked, so timestamps without a UTC offset are rejected.

    Parameters
    ----------
    path : str
        Path of the schema under ``v2.0.0/`` (e.g. ``"api/components/forecast/ForecastPayloadDM.json"``).

    Returns
    -------
    Draft7Validator
        The validator.

    """
    return Draft7Validator(
        v2_schema(path),
        registry=Registry(retrieve=_retrieve_v2_schema),
        format_checker=Draft7Validator.FORMAT_CHECKER,
    )


def _v2_values(periods: int, timestep_minutes: int) -> list[dict]:
    dates = pd.date_range(start="2023-01-01", periods=periods, freq=f"{timestep_minutes}min", tz="UTC")
    return [{"timestamp": ts.isoformat(), "value": float(i)} for i, ts in enumerate(dates)]


def make_v2_training_payload(
    periods: int = 200,
    timestep_minutes: int = 60,
    preferences: dict | None = None,
    has_time_parameters: dict | None = None,
    target_series: dict | None = None,
    **overrides: object,
) -> dict:
    """
    Build a v2 training payload (``ForecastTrainingInputV2``) with UTC values from 2023-01-01.

    Parameters
    ----------
    periods : int
        Number of training values.
    timestep_minutes : int
        Spacing of the training values, also set in ``hasTimeParameters``.
    preferences : dict, optional
        Fields merged into ``hasUserForecastPreferences``.
    has_time_parameters : dict, optional
        Fields merged into ``hasTimeParameters``.
    target_series : dict, optional
        Fields merged into ``target_series``.
    **overrides : object
        Top-level fields replacing the defaults.

    Returns
    -------
    dict
        The payload.

    """
    payload: dict = {
        "hasTimeParameters": {"timestep_minutes": timestep_minutes, "horizon_hours": 24, **(has_time_parameters or {})},
        "hasUserForecastPreferences": {
            "forecast_type": "STOCHASTIC",
            "model_id": "test-model",
            "model_version": "1.0.0",
            "model_preset": "fast_training",
            **(preferences or {}),
        },
        "target_series": {
            "series_id": "s1",
            "observedProperty": "ActivePower",
            "hasFeatureOfInterest": "pv1",
            "asset_type": "PV",
            "unit_symbol": "kW",
            "training_values": _v2_values(periods, timestep_minutes),
            **(target_series or {}),
        },
    }
    payload.update(overrides)
    return payload


def make_v2_forecast_payload(
    periods: int = 200,
    timestep_minutes: int = 60,
    has_time_parameters: dict | None = None,
    target_series: dict | None = None,
    **overrides: object,
) -> dict:
    """
    Build a v2 forecast payload (``ForecastInputV2``) with UTC values from 2023-01-01.

    Parameters
    ----------
    periods : int
        Number of historical values.
    timestep_minutes : int
        Spacing of the historical values, also set in ``hasTimeParameters``.
    has_time_parameters : dict, optional
        Fields merged into ``hasTimeParameters``.
    target_series : dict, optional
        Fields merged into ``target_series``.
    **overrides : object
        Top-level fields replacing the defaults.

    Returns
    -------
    dict
        The payload.

    """
    payload: dict = {
        "forecast_type": "STOCHASTIC",
        "hasTimeParameters": {"timestep_minutes": timestep_minutes, "horizon_hours": 24, **(has_time_parameters or {})},
        "model_id": "test-model",
        "model_version": "1.0.0",
        "target_series": {
            "series_id": "s1",
            "observedProperty": "ActivePower",
            "hasFeatureOfInterest": "pv1",
            "asset_type": "PV",
            "unit_symbol": "kW",
            "historical_values": _v2_values(periods, timestep_minutes),
            **(target_series or {}),
        },
    }
    payload.update(overrides)
    return payload


def make_model_metadata(
    freq: str = "h",
    prediction_length: int = 24,
    added_columns: dict | None = None,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> ModelMetadata:
    """
    Build the metadata of a trained model.

    Parameters
    ----------
    freq : str
        Frequency of the model.
    prediction_length : int
        Forecast horizon in steps.
    added_columns : dict, optional
        Engineered feature columns, by feature name.
    lags : list[int], optional
        Lags of the lag features.
    windows : list[int], optional
        Windows of the rolling-stat features.

    Returns
    -------
    ModelMetadata
        The metadata, with a minimum series length of 50.

    """
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


def make_mock_model(freq: str = "h", prediction_length: int = 24, metadata: ModelMetadata | None = None) -> MagicMock:
    """
    Build a mock of a loaded ``U2TimeSeriesForecastingModel``.

    Its forecast has a mean of 1.0 and quantiles 0.1, 0.5 and 0.9 of 0.5, 1.0 and 1.5.

    Parameters
    ----------
    freq : str
        Frequency of the model.
    prediction_length : int
        Forecast horizon in steps.
    metadata : ModelMetadata, optional
        Metadata of the model; built from ``freq`` and ``prediction_length`` when omitted.

    Returns
    -------
    MagicMock
        The mock model.

    """
    mock_model = MagicMock()
    mock_model.metadata = metadata or make_model_metadata(freq=freq, prediction_length=prediction_length)
    mock_model.freq = freq
    mock_model.prediction_length = prediction_length
    mock_model.quantile_levels = [0.1, 0.5, 0.9]

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


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def json_single_timeseries(temp_dir: str) -> str:
    """Single series, 168 hourly periods."""
    dates = pd.date_range(start="2023-01-01", periods=168, freq="h")
    payload = _make_training_payload(dates, "group_1")
    path = os.path.join(temp_dir, "single_group.json")
    _write_json(payload, path)
    return path


@pytest.fixture
def json_daily_freq(temp_dir: str) -> str:
    """Single series with daily (1440-minute) timestep, 28 periods."""
    dates = pd.date_range(start="2023-01-01", periods=28, freq="d")
    payload = _make_training_payload(dates, "group_1", timestep_minutes=1440)
    path = os.path.join(temp_dir, "daily_freq.json")
    _write_json(payload, path)
    return path


@pytest.fixture
def json_empty(temp_dir: str) -> str:
    """Series with an empty training_values list."""
    payload = _make_training_payload(pd.DatetimeIndex([]), "group_1")
    path = os.path.join(temp_dir, "empty.json")
    _write_json(payload, path)
    return path


@pytest.fixture
def json_short_periods(temp_dir: str) -> str:
    """Series with only five hourly periods."""
    dates = pd.date_range(start="2023-01-01", periods=5, freq="h")
    payload = _make_training_payload(dates, "group_1")
    path = os.path.join(temp_dir, "short_periods.json")
    _write_json(payload, path)
    return path


@pytest.fixture
def json_with_nan_values(temp_dir: str) -> str:
    """Series with 10 random null values in training_values."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    nan_indices = np.random.choice(100, size=10, replace=False)
    payload = _make_training_payload(dates, "group_1", nan_indices=nan_indices)
    path = os.path.join(temp_dir, "with_nan_values.json")
    _write_json(payload, path)
    return path


@pytest.fixture
def json_long_periods(temp_dir: str) -> str:
    """Single series with 2048 hourly periods."""
    dates = pd.date_range(end="2023-01-01", periods=2048, freq="h")
    payload = _make_training_payload(dates, "group_1")
    path = os.path.join(temp_dir, "long_periods.json")
    _write_json(payload, path)
    return path


@pytest.fixture
def multi_group_df() -> pd.DataFrame:
    """
    DataFrame with two groups of different lengths for testing cross-group isolation.

    Group 'group_a' has 200 hourly rows; 'group_b' has 150 hourly rows.
    Both start at the same date so their timestamps overlap, making any
    cross-group bleed immediately detectable.
    """
    rng = np.random.default_rng(42)
    dates_a = pd.date_range(start="2023-01-01", periods=200, freq="h")
    dates_b = pd.date_range(start="2023-01-01", periods=150, freq="h")

    df_a = pd.DataFrame(
        {
            TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates_a,
            TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: rng.random(200),
            TIME_SERIES_GROUP_COLUMN_NAME: "group_a",
        }
    )
    df_b = pd.DataFrame(
        {
            TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates_b,
            TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: rng.random(150) + 100.0,  # offset so group_b values are distinct
            TIME_SERIES_GROUP_COLUMN_NAME: "group_b",
        }
    )
    return pd.concat([df_a, df_b], ignore_index=True)


@pytest.fixture
def json_without_location(temp_dir: str) -> str:
    """Series whose payload omits the location object entirely."""
    dates = pd.date_range(start="2023-01-01", periods=168, freq="h")
    payload = _make_training_payload(dates, "group_1", include_location=False)
    path = os.path.join(temp_dir, "without_location.json")
    _write_json(payload, path)
    return path
