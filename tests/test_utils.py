"""Tests for u2demo_time_series_forecasting.utils module.

Covers retry-eligibility logic (HTTP status codes, exception types), known-covariate filtering by horizon and feature family, min-timeseries-length calculation from lag/window lists, and reverse geocoding.
"""

from unittest.mock import Mock

import pytest
import requests

from u2demo_time_series_forecasting.utils import (
    get_known_covariates,
    get_min_timeseries_length,
    is_retryable_error,
    reverse_geocode,
)


@pytest.mark.unit
class TestIsRetryableError:
    """Tests for is_retryable_error function."""

    def test_retryable_504_error(self):
        """Test that 504 Gateway Timeout is retryable."""
        exc = requests.exceptions.RequestException()
        exc.response = Mock(status_code=504)
        assert is_retryable_error(exc) is True

    def test_retryable_429_error(self):
        """Test that 429 Too Many Requests is retryable."""
        exc = requests.exceptions.RequestException()
        exc.response = Mock(status_code=429)
        assert is_retryable_error(exc) is True

    def test_non_retryable_error_status(self):
        """Test that other status codes are not retryable."""
        exc = requests.exceptions.RequestException()
        exc.response = Mock(status_code=500)
        assert is_retryable_error(exc) is False

    def test_non_request_exception(self):
        """Test that non-RequestException errors are not retryable."""
        exc = ValueError("Some error")
        assert is_retryable_error(exc) is False

    def test_exception_without_response(self):
        """Test that exceptions without response attribute are not retryable."""
        exc = requests.exceptions.RequestException()
        assert is_retryable_error(exc) is False

    def test_exception_with_none_response(self):
        """Test that exceptions with None response are not retryable."""
        exc = requests.exceptions.RequestException()
        exc.response = None
        assert is_retryable_error(exc) is False


@pytest.mark.unit
class TestGetKnownCovariates:
    """Tests for get_known_covariates function."""

    def test_empty_columns_returns_empty(self):
        result = get_known_covariates({}, horizon=24)
        assert result == []

    def test_temporal_features_all_kept(self):
        """Temporal features are always known at prediction time."""
        added = {"temporal": ["day_of_week_0", "month_sin", "month_cos"]}
        result = get_known_covariates(added, horizon=24)
        assert set(result) == {"day_of_week_0", "month_sin", "month_cos"}

    def test_lag_below_horizon_removed(self):
        """Lag features where lag < horizon should be removed (data leakage)."""
        added = {"lag": ["lag_1", "lag_24", "lag_168"]}
        result = get_known_covariates(added, horizon=24)
        assert "lag_1" not in result
        assert "lag_24" in result
        assert "lag_168" in result

    def test_lag_equal_to_horizon_kept(self):
        """Lag equal to horizon is known (lag_24 with horizon=24 is safe)."""
        added = {"lag": ["lag_24"]}
        result = get_known_covariates(added, horizon=24)
        assert "lag_24" in result

    def test_rolling_stats_always_kept(self):
        """Rolling stats (roll_*) are kept even if their window < horizon."""
        added = {"lag": ["lag_1", "roll_mean_12", "roll_std_12"]}
        result = get_known_covariates(added, horizon=24)
        assert "roll_mean_12" in result
        assert "roll_std_12" in result
        assert "lag_1" not in result

    def test_multiple_feature_families_combined(self):
        """Features from all families should be combined."""
        added = {
            "temporal": ["month_sin"],
            "lag": ["lag_168"],
            "holiday": ["holiday_public"],
            "weather": ["temperature_2m"],
        }
        result = get_known_covariates(added, horizon=24)
        assert set(result) == {"month_sin", "lag_168", "holiday_public", "temperature_2m"}

    def test_horizon_zero_keeps_all_lags(self):
        """With horizon=0, no lag is below the horizon so all are kept."""
        added = {"lag": ["lag_1", "lag_7"]}
        result = get_known_covariates(added, horizon=0)
        assert "lag_1" in result
        assert "lag_7" in result


@pytest.mark.unit
class TestGetMinTimeseriesLength:
    """Tests for get_min_timeseries_length function."""

    def test_basic_calculation(self):
        result = get_min_timeseries_length(lags=[24, 48, 168], min_lookback=100)
        assert result == 168 + 100  # max(lags) + min_lookback

    def test_with_horizon(self):
        result = get_min_timeseries_length(lags=[24, 48], min_lookback=100, horizon=24)
        assert result == 48 + 100 + 24  # max(lags) + min_lookback + horizon

    def test_horizon_none_treated_as_zero(self):
        result = get_min_timeseries_length(lags=[24], min_lookback=100, horizon=None)
        assert result == 24 + 100

    def test_single_lag(self):
        result = get_min_timeseries_length(lags=[168], min_lookback=50)
        assert result == 218

    def test_zero_lag(self):
        """When lags=[0], max(lags) is 0."""
        result = get_min_timeseries_length(lags=[0], min_lookback=100)
        assert result == 100

    def test_empty_lags_list(self):
        """An empty lags list should be handled gracefully and return
        min_lookback + horizon without crashing on max() of an empty sequence.
        """
        result = get_min_timeseries_length(lags=[], min_lookback=100, horizon=24)
        assert result == 124  # 0 + 100 + 24


@pytest.mark.unit
class TestReverseGeocode:
    """Tests for reverse_geocode function."""

    def test_returns_country_and_region(self, monkeypatch):
        """Should return country_code and region_code from Nominatim."""
        mock_location = Mock()
        mock_location.raw = {"address": {"country_code": "de", "ISO3166-2-lvl4": "DE-BW"}}

        mock_geolocator = Mock()
        mock_geolocator.reverse.return_value = mock_location

        monkeypatch.setattr(
            "u2demo_time_series_forecasting.utils.Nominatim", lambda domain, user_agent: mock_geolocator
        )
        # Bypass joblib and tenacity caching/retry decorators by calling the wrapped function
        inner = reverse_geocode.__wrapped__.__wrapped__

        result = inner((48.7758, 9.1829))

        assert result["country_code"] == "de"
        assert result["region_code"] == "DE-BW"

    def test_none_location_raises(self, monkeypatch):
        """Should raise ValueError when geocoder returns None."""
        mock_geolocator = Mock()
        mock_geolocator.reverse.return_value = None

        monkeypatch.setattr(
            "u2demo_time_series_forecasting.utils.Nominatim", lambda domain, user_agent: mock_geolocator
        )
        inner = reverse_geocode.__wrapped__.__wrapped__

        with pytest.raises(ValueError, match="Could not geocode location"):
            inner((0.0, 0.0))

    def test_missing_address_fields(self, monkeypatch):
        """Should return None for missing address fields."""
        mock_location = Mock()
        mock_location.raw = {"address": {}}

        mock_geolocator = Mock()
        mock_geolocator.reverse.return_value = mock_location

        monkeypatch.setattr(
            "u2demo_time_series_forecasting.utils.Nominatim", lambda domain, user_agent: mock_geolocator
        )
        inner = reverse_geocode.__wrapped__.__wrapped__

        result = inner((10.0, 20.0))

        assert result["country_code"] is None
        assert result["region_code"] is None
