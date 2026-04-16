"""Pytest configuration and shared fixtures."""

import json
import os
import tempfile
from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest

from u2.constants.columns import (
    TIME_SERIES_GROUP_COLUMN_NAME,
    TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)
from u2.data_models import ForecastTrainingInput

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
