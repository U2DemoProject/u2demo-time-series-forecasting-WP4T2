"""Tests for u2.features.weather module.

Covers fetch_weather (Open-Meteo API), enrich_weather (split historical/forecast, fetch both, merge), and API schema drift detection. External API calls are mocked in unit tests.
"""

import datetime as dt

import pandas as pd
import pytest
import requests

from u2.constants.envs import settings
from u2.constants.weather import DEFAULT_WEATHER_DAILY_VARS, DEFAULT_WEATHER_HOURLY_VARS
from u2.features.weather import enrich_weather, fetch_weather

WEATHER_FORECAST_API_URL = settings.weather_forecast_api_url
WEATHER_HISTORICAL_API_URL = settings.weather_historical_api_url


@pytest.mark.unit
class TestFetchWeather:
    """Tests for fetch_weather function."""

    def test_hourly_frequency_uses_hourly_key(self, monkeypatch):
        """Hourly frequency should send hourly params."""
        captured_params = {}

        def mock_get(url, params=None, **kwargs):
            captured_params.update(params or {})

            class MockResp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"hourly": {"time": ["2023-01-01T00:00"], "temperature_2m": [5.0]}}

            return MockResp()

        monkeypatch.setattr("u2.features.weather.requests.get", mock_get)

        result = fetch_weather(48.0, 9.0, "2023-01-01", "2023-01-02", ["temperature_2m"], "h")

        assert "hourly" in captured_params
        assert "daily" not in captured_params
        assert len(result) == 1

    def test_daily_frequency_uses_daily_key(self, monkeypatch):
        """Daily frequency should send daily params."""
        captured_params = {}

        def mock_get(url, params=None, **kwargs):
            captured_params.update(params or {})

            class MockResp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"daily": {"time": ["2023-01-01"], "temperature_2m_max": [10.0]}}

            return MockResp()

        monkeypatch.setattr("u2.features.weather.requests.get", mock_get)

        fetch_weather(48.0, 9.0, "2023-01-01", "2023-01-02", ["temperature_2m_max"], "D")

        assert "daily" in captured_params
        assert "hourly" not in captured_params

    def test_invalid_frequency_raises(self):
        """Unrecognized frequency string should raise ValueError."""
        with pytest.raises(ValueError, match="Unrecognized frequency"):
            fetch_weather(48.0, 9.0, "2023-01-01", "2023-01-02", ["temp"], "W")

    def test_historical_mode_uses_historical_url(self, monkeypatch):
        """Historical mode should call the archive API URL."""
        called_url = []

        def mock_get(url, params=None, **kwargs):
            called_url.append(url)

            class MockResp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"hourly": {"time": [], "temperature_2m": []}}

            return MockResp()

        monkeypatch.setattr("u2.features.weather.requests.get", mock_get)

        fetch_weather(48.0, 9.0, "2023-01-01", "2023-01-02", ["temperature_2m"], "h", mode="historical")

        assert "archive" in called_url[0]

    def test_forecast_mode_uses_forecast_url(self, monkeypatch):
        """Forecast mode should call the forecast API URL."""
        called_url = []

        def mock_get(url, params=None, **kwargs):
            called_url.append(url)

            class MockResp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"hourly": {"time": [], "temperature_2m": []}}

            return MockResp()

        monkeypatch.setattr("u2.features.weather.requests.get", mock_get)

        fetch_weather(48.0, 9.0, "2023-01-01", "2023-01-02", ["temperature_2m"], "h", mode="forecast")

        assert "archive" not in called_url[0]

    def test_converts_time_to_datetime(self, monkeypatch):
        """time column should be converted to datetime."""

        def mock_get(url, params=None, **kwargs):
            class MockResp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "hourly": {
                            "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
                            "temperature_2m": [5.0, 6.0],
                        }
                    }

            return MockResp()

        monkeypatch.setattr("u2.features.weather.requests.get", mock_get)

        result = fetch_weather(48.0, 9.0, "2023-01-01", "2023-01-01", ["temperature_2m"], "h")

        assert pd.api.types.is_datetime64_any_dtype(result["time"])

    def test_http_error_raises_runtime_error(self, monkeypatch):
        """HTTP errors should be wrapped in RuntimeError."""

        def mock_get(url, params=None, **kwargs):
            raise requests.exceptions.ConnectionError("connection failed")

        monkeypatch.setattr("u2.features.weather.requests.get", mock_get)

        with pytest.raises(RuntimeError, match="Failed to fetch data"):
            fetch_weather(48.0, 9.0, "2023-01-01", "2023-01-02", ["temperature_2m"], "h")


@pytest.mark.unit
class TestEnrichWeather:
    """Tests for enrich_weather function."""

    def _make_hourly_df(self, start="2023-01-01", periods=48):
        dates = pd.date_range(start=start, periods=periods, freq="h")
        return pd.DataFrame({"time": dates, "value": range(periods)})

    def _mock_fetch_weather(self, monkeypatch, variables=None):
        """Set up a mock fetch_weather that returns synthetic data."""
        if variables is None:
            variables = DEFAULT_WEATHER_HOURLY_VARS

        def mock_fetch(lat, lon, start_date, end_date, variables, frequency, mode="historical"):
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            freq = "h" if frequency.endswith(("h", "min")) else "D"
            times = pd.date_range(start, end, freq=freq)
            data = {"time": times}
            for v in variables:
                data[v] = [20.0] * len(times)
            return pd.DataFrame(data)

        monkeypatch.setattr("u2.features.weather.fetch_weather", mock_fetch)

    def test_enriches_with_default_hourly_variables(self, monkeypatch):
        """Should add default hourly weather variables when none specified."""
        df = self._make_hourly_df()
        self._mock_fetch_weather(monkeypatch)

        result_df, added_vars = enrich_weather(df, "time", (48.0, 9.0), None, "h")

        assert added_vars == DEFAULT_WEATHER_HOURLY_VARS
        for var in DEFAULT_WEATHER_HOURLY_VARS:
            assert var in result_df.columns

    def test_enriches_with_default_daily_variables(self, monkeypatch):
        """Should add default daily weather variables for daily frequency."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        df = pd.DataFrame({"time": dates, "value": range(30)})

        def mock_fetch(lat, lon, start_date, end_date, variables, frequency, mode="historical"):
            times = pd.date_range(start_date, end_date, freq="D")
            data = {"time": times}
            for v in variables:
                data[v] = [10.0] * len(times)
            return pd.DataFrame(data)

        monkeypatch.setattr("u2.features.weather.fetch_weather", mock_fetch)

        result_df, added_vars = enrich_weather(df, "time", (48.0, 9.0), None, "D")

        assert added_vars == DEFAULT_WEATHER_DAILY_VARS

    def test_present_time_splits_historical_forecast(self, monkeypatch):
        """When present_time is set, data before it uses historical, after uses forecast."""
        df = self._make_hourly_df(periods=48)
        present_time = pd.Timestamp("2023-01-02")

        modes_called = []

        def mock_fetch(lat, lon, start_date, end_date, variables, frequency, mode="historical"):
            modes_called.append(mode)
            times = pd.date_range(start_date, end_date, freq="h")
            data = {"time": times}
            for v in variables:
                data[v] = [20.0] * len(times)
            return pd.DataFrame(data)

        monkeypatch.setattr("u2.features.weather.fetch_weather", mock_fetch)

        enrich_weather(df, "time", (48.0, 9.0), present_time, "h")

        assert "historical" in modes_called
        assert "forecast" in modes_called

    def test_present_time_none_uses_all_historical(self, monkeypatch):
        """When present_time is None, all data treated as historical."""
        df = self._make_hourly_df(periods=24)
        modes_called = []

        def mock_fetch(lat, lon, start_date, end_date, variables, frequency, mode="historical"):
            modes_called.append(mode)
            times = pd.date_range(start_date, end_date, freq="h")
            data = {"time": times}
            for v in variables:
                data[v] = [20.0] * len(times)
            return pd.DataFrame(data)

        monkeypatch.setattr("u2.features.weather.fetch_weather", mock_fetch)

        enrich_weather(df, "time", (48.0, 9.0), None, "h")

        assert modes_called == ["historical"]

    def test_removes_time_aligned_column(self, monkeypatch):
        """The temporary time_aligned column should not remain."""
        df = self._make_hourly_df(periods=24)
        self._mock_fetch_weather(monkeypatch)

        result_df, _ = enrich_weather(df, "time", (48.0, 9.0), None, "h")

        assert "time_aligned" not in result_df.columns

    def test_preserves_original_columns(self, monkeypatch):
        """Original DataFrame columns should be preserved."""
        df = self._make_hourly_df(periods=24)
        self._mock_fetch_weather(monkeypatch)

        result_df, _ = enrich_weather(df, "time", (48.0, 9.0), None, "h")

        assert "time" in result_df.columns
        assert "value" in result_df.columns

    def test_empty_future_chunk_skipped(self, monkeypatch):
        """When present_time is after all data, the future chunk is empty and should be skipped."""
        df = self._make_hourly_df(start="2023-01-01", periods=24)
        # present_time after all data => df_past = full df, df_future = empty
        present_time = pd.Timestamp("2023-01-03")

        call_count = []

        def mock_fetch(lat, lon, start_date, end_date, variables, frequency, mode="historical"):
            call_count.append(mode)
            times = pd.date_range(start_date, end_date, freq="h")
            data = {"time": times}
            for v in variables:
                data[v] = [20.0] * len(times)
            return pd.DataFrame(data)

        monkeypatch.setattr("u2.features.weather.fetch_weather", mock_fetch)

        result_df, _ = enrich_weather(df, "time", (48.0, 9.0), present_time, "h")

        # Only historical should have been called (future chunk is empty, skipped)
        assert call_count == ["historical"]
        assert len(result_df) == 24

    def test_fetch_weather_failure_reraises(self, monkeypatch):
        """When fetch_weather raises, enrich_weather should re-raise the exception."""
        df = self._make_hourly_df(periods=24)

        def mock_fetch(lat, lon, start_date, end_date, variables, frequency, mode="historical"):
            raise RuntimeError("API down")

        monkeypatch.setattr("u2.features.weather.fetch_weather", mock_fetch)

        with pytest.raises(RuntimeError, match="API down"):
            enrich_weather(df, "time", (48.0, 9.0), None, "h")

    def test_empty_weather_data_raises(self, monkeypatch):
        """When fetch_weather returns empty DataFrame, enrich_weather should raise ValueError."""
        df = self._make_hourly_df(periods=24)

        def mock_fetch(lat, lon, start_date, end_date, variables, frequency, mode="historical"):
            return pd.DataFrame()

        monkeypatch.setattr("u2.features.weather.fetch_weather", mock_fetch)

        with pytest.raises(ValueError, match="No weather data"):
            enrich_weather(df, "time", (48.0, 9.0), None, "h")

    def test_explicit_variables_skip_defaults(self, monkeypatch):
        """When explicit variables are passed, defaults should be skipped."""
        df = self._make_hourly_df(periods=24)
        custom_vars = ["temperature_2m", "precipitation"]

        def mock_fetch(lat, lon, start_date, end_date, variables, frequency, mode="historical"):
            times = pd.date_range(start_date, end_date, freq="h")
            data = {"time": times}
            for v in variables:
                data[v] = [10.0] * len(times)
            return pd.DataFrame(data)

        monkeypatch.setattr("u2.features.weather.fetch_weather", mock_fetch)

        result_df, added_vars = enrich_weather(df, "time", (48.0, 9.0), None, "h", variables=custom_vars)

        assert added_vars == custom_vars
        assert "temperature_2m" in result_df.columns
        assert "precipitation" in result_df.columns
        # Default vars that were NOT requested should be absent
        assert "cloudcover" not in result_df.columns

    def test_minute_frequency_uses_hourly(self, monkeypatch):
        """Minute-based frequency should use hourly weather variables."""
        dates = pd.date_range("2023-01-01", periods=24, freq="30min")
        df = pd.DataFrame({"time": dates, "value": range(24)})
        self._mock_fetch_weather(monkeypatch)

        result_df, added_vars = enrich_weather(df, "time", (48.0, 9.0), None, "30min")

        assert added_vars == DEFAULT_WEATHER_HOURLY_VARS


@pytest.mark.unit
class TestEnrichWeatherTimezone:
    """Tests for present_time timezone handling in enrich_weather.

    The Open-Meteo historical forecast API is only available from 2022-01-01 UTC.
    present_time may arrive as a naive datetime (treated as UTC) or as a
    timezone-aware datetime (converted to UTC before the boundary check and
    before the df split so that naive df timestamps remain comparable).
    """

    def _make_hourly_df(self, start, periods=48):
        dates = pd.date_range(start=start, periods=periods, freq="h")
        return pd.DataFrame({"time": dates, "value": range(periods)})

    def _mode_tracking_mock(self, monkeypatch):
        modes_called = []

        def mock_fetch(lat, lon, start_date, end_date, variables, frequency, mode="historical"):
            modes_called.append(mode)
            times = pd.date_range(start_date, end_date, freq="h")
            data = {"time": times}
            for v in variables:
                data[v] = [20.0] * len(times)
            return pd.DataFrame(data)

        monkeypatch.setattr("u2.features.weather.fetch_weather", mock_fetch)
        return modes_called

    def test_naive_present_time_before_2022_uses_all_historical(self, monkeypatch):
        """Naive present_time before 2022 should use all-historical path (no split)."""
        df = self._make_hourly_df(start="2021-06-01", periods=24)
        present_time = df["time"].iloc[12]  # naive, 2021-06-01 12:00

        modes_called = self._mode_tracking_mock(monkeypatch)

        enrich_weather(df, "time", (48.0, 9.0), present_time, "h")

        assert modes_called == ["historical"]

    def test_utc_aware_present_time_after_2022_splits(self, monkeypatch):
        """UTC-aware present_time after 2022 should split df into historical + forecast."""
        df = self._make_hourly_df(start="2023-01-01", periods=48)
        present_time = df["time"].iloc[24].tz_localize("UTC")  # 2023-01-02 00:00 UTC

        modes_called = self._mode_tracking_mock(monkeypatch)

        enrich_weather(df, "time", (48.0, 9.0), present_time, "h")

        assert "historical" in modes_called
        assert "forecast" in modes_called

    def test_utc_aware_present_time_before_2022_uses_all_historical(self, monkeypatch):
        """UTC-aware present_time before 2022 should use all-historical path (no split)."""
        df = self._make_hourly_df(start="2021-06-01", periods=24)
        present_time = df["time"].iloc[12].tz_localize("UTC")  # 2021-06-01 12:00 UTC

        modes_called = self._mode_tracking_mock(monkeypatch)

        enrich_weather(df, "time", (48.0, 9.0), present_time, "h")

        assert modes_called == ["historical"]

    def test_positive_offset_that_maps_to_before_utc_cutoff_uses_all_historical(self, monkeypatch):
        """present_time that appears to be 2022-01-01 in local time but is still 2021 in UTC
        (e.g. UTC+3 at 01:00 local = 2021-12-31 22:00 UTC) must use all-historical."""
        # 2022-01-01 01:00:00+03:00 => 2021-12-31 22:00:00 UTC (before cutoff)
        present_time = dt.datetime(2022, 1, 1, 1, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=3)))
        df = self._make_hourly_df(start="2021-12-30", periods=72)

        modes_called = self._mode_tracking_mock(monkeypatch)

        enrich_weather(df, "time", (48.0, 9.0), present_time, "h")

        assert modes_called == ["historical"]

    def test_negative_offset_that_maps_to_after_utc_cutoff_splits(self, monkeypatch):
        """present_time that appears to be 2021-12-31 in local time but is 2022 in UTC
        (e.g. UTC-2 at 23:00 local = 2022-01-01 01:00 UTC) must split into historical + forecast."""
        # 2021-12-31 23:00:00-02:00 => 2022-01-01 01:00:00 UTC (after cutoff)
        present_time = dt.datetime(2021, 12, 31, 23, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-2)))
        # df timestamps span the UTC split point (naive 2022-01-01 01:00)
        df = self._make_hourly_df(start="2022-01-01", periods=48)

        modes_called = self._mode_tracking_mock(monkeypatch)

        enrich_weather(df, "time", (48.0, 9.0), present_time, "h")

        assert "historical" in modes_called
        assert "forecast" in modes_called


@pytest.mark.integration
class TestOpenMeteoAPISchema:
    """Lightweight checks that Open-Meteo API response schemas haven't drifted."""

    def test_historical_api_response_has_expected_fields(self):
        """Archive API should return hourly data with the requested variables."""
        params = {
            "latitude": 48.7758,
            "longitude": 9.1829,
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "hourly": "temperature_2m",
            "timezone": "UTC",
        }
        response = requests.get(WEATHER_HISTORICAL_API_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()

        assert "hourly" in payload, "Expected 'hourly' key in response"
        hourly = payload["hourly"]
        assert "time" in hourly, "Expected 'time' in hourly data"
        assert "temperature_2m" in hourly, "Expected 'temperature_2m' in hourly data"
        assert len(hourly["time"]) > 0, "Expected at least one timestamp"

    def test_forecast_api_response_has_expected_fields(self):
        """Forecast API should return hourly data with the requested variables."""
        params = {
            "latitude": 48.7758,
            "longitude": 9.1829,
            "start_date": "2026-04-10",
            "end_date": "2026-04-11",
            "hourly": "temperature_2m",
            "timezone": "UTC",
        }
        response = requests.get(WEATHER_FORECAST_API_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()

        assert "hourly" in payload, "Expected 'hourly' key in response"
        hourly = payload["hourly"]
        assert "time" in hourly
        assert "temperature_2m" in hourly
