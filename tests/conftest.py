"""Pytest configuration and shared fixtures."""

import os
import tempfile
from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest

from u2.defaults import (
    DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME,
    DEFAULT_TIME_SERIES_LOCATION_COLUMN_NAME,
    DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def csv_multiple_groups_different_end_dates(temp_dir: str) -> str:
    """Create a CSV with multiple groups that have different end dates."""
    periods = 100
    dates_group1 = pd.date_range(end="2023-01-01", periods=periods, freq="h")
    dates_group2 = pd.date_range(end="2023-01-12", periods=periods, freq="h")
    dates_group3 = pd.date_range(end="2023-01-27", periods=periods, freq="h")

    data = []
    for dates, group in [(dates_group1, "group_1"), (dates_group2, "group_2"), (dates_group3, "group_3")]:
        group_data = {
            DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
            DEFAULT_TIME_SERIES_LOCATION_COLUMN_NAME: ["Berlin"] * len(dates),
            DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.rand(len(dates)),
            DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME: [group] * len(dates),
        }
        data.append(pd.DataFrame(group_data))

    df = pd.concat(data, ignore_index=True)
    csv_path = os.path.join(temp_dir, "multiple_groups_different_end_dates.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def csv_multiple_groups(temp_dir: str) -> str:
    """Create a CSV with multiple groups that have different end dates."""
    start = "2023-01-01"
    periods = 100
    dates_group1 = pd.date_range(end=start, periods=periods, freq="h")
    dates_group2 = pd.date_range(end=start, periods=periods, freq="h")
    dates_group3 = pd.date_range(end=start, periods=periods, freq="h")

    data = []
    for dates, group in [(dates_group1, "group_1"), (dates_group2, "group_2"), (dates_group3, "group_3")]:
        group_data = {
            DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
            DEFAULT_TIME_SERIES_LOCATION_COLUMN_NAME: ["Berlin"] * len(dates),
            DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.rand(len(dates)),
            DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME: [group] * len(dates),
        }
        data.append(pd.DataFrame(group_data))

    df = pd.concat(data, ignore_index=True)
    csv_path = os.path.join(temp_dir, "multiple_groups.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def csv_single_group(temp_dir: str) -> str:
    """Create a CSV with only one group."""
    dates = pd.date_range(start="2023-01-01", periods=168, freq="h")
    data = {
        DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
        DEFAULT_TIME_SERIES_LOCATION_COLUMN_NAME: ["Berlin"] * len(dates),
        DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.rand(len(dates)),
        DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME: ["group_1"] * len(dates),
    }
    df = pd.DataFrame(data)

    csv_path = os.path.join(temp_dir, "single_group.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def csv_daily_freq(temp_dir: str) -> str:
    """Create a CSV with only one group."""
    dates = pd.date_range(start="2023-01-01", periods=28, freq="d")
    data = {
        DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
        DEFAULT_TIME_SERIES_LOCATION_COLUMN_NAME: ["Berlin"] * len(dates),
        DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.rand(len(dates)),
        DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME: ["group_1"] * len(dates),
    }
    df = pd.DataFrame(data)

    csv_path = os.path.join(temp_dir, "daily_freq.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def csv_same_end_date_different_length(temp_dir: str) -> str:
    """Create a CSV with groups that have the same end date but different lengths."""
    end_date = pd.Timestamp("2023-01-31 23:00:00")

    dates_group1 = pd.date_range(end=end_date, periods=168, freq="h")
    dates_group2 = pd.date_range(end=end_date, periods=240, freq="h")
    dates_group3 = pd.date_range(end=end_date, periods=120, freq="h")

    data = []
    for dates, group in [(dates_group1, "group_1"), (dates_group2, "group_2"), (dates_group3, "group_3")]:
        group_data = {
            DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
            DEFAULT_TIME_SERIES_LOCATION_COLUMN_NAME: ["Berlin"] * len(dates),
            DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.rand(len(dates)),
            DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME: [group] * len(dates),
        }
        data.append(pd.DataFrame(group_data))

    df = pd.concat(data, ignore_index=True)
    csv_path = os.path.join(temp_dir, "same_end_date_different_length.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def csv_empty(temp_dir: str) -> str:
    """Create an empty CSV with just headers."""
    df = pd.DataFrame(
        columns=[
            DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME,
            DEFAULT_TIME_SERIES_LOCATION_COLUMN_NAME,
            DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
            DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME,
        ]
    )

    csv_path = os.path.join(temp_dir, "empty.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def csv_short_periods(temp_dir: str) -> str:
    """Create a CSV with only five periods."""
    dates = pd.date_range(start="2023-01-01", periods=5, freq="h")
    data = {
        DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
        DEFAULT_TIME_SERIES_LOCATION_COLUMN_NAME: ["Berlin"] * 5,
        DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: np.random.rand(5),
        DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME: ["group_1"] * 5,
    }
    df = pd.DataFrame(data)

    csv_path = os.path.join(temp_dir, "short_periods.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def csv_with_nan_values(temp_dir: str) -> str:
    """Create a CSV with NaN values in the value column."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    values = np.random.rand(100)

    nan_indices = np.random.choice(100, size=10, replace=False)
    values[nan_indices] = np.nan

    data = {
        DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME: dates,
        DEFAULT_TIME_SERIES_LOCATION_COLUMN_NAME: ["Berlin"] * len(dates),
        DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: values,
        DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME: ["group_1"] * len(dates),
    }
    df = pd.DataFrame(data)

    csv_path = os.path.join(temp_dir, "with_nan_values.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def csv_other_columns(temp_dir: str) -> str:
    """Create a CSV with non-default columns names."""
    dates = pd.date_range(start="2023-01-01", periods=168, freq="h")
    data = {
        "datetime": dates,
        "site": ["Berlin"] * len(dates),
        "energy": np.random.rand(len(dates)),
        "asset_name": ["group_1"] * len(dates),
    }
    df = pd.DataFrame(data)

    csv_path = os.path.join(temp_dir, "other_columns.csv")
    df.to_csv(csv_path, index=False)

    return csv_path
