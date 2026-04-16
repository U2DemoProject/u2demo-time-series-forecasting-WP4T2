"""Tests for u2demo_time_series_forecasting.features.holidays module.

Covers fetch_holidays_by_year (Nager.Date API), encode_holiday_types (one-hot encoding), merge_holidays_with_dataframe (geocode -> fetch -> filter -> encode -> merge), and API schema drift detection. External API calls are mocked in unit tests.
"""

import pandas as pd
import pytest
import requests

from u2demo_time_series_forecasting.constants.envs import settings
from u2demo_time_series_forecasting.constants.holidays import DEFAULT_HOLIDAY_TYPES
from u2demo_time_series_forecasting.features.holidays import (
    encode_holiday_types,
    fetch_holidays_by_year,
    merge_holidays_with_dataframe,
)

PUBLIC_HOLIDAY_API_URL = settings.public_holiday_api_url


@pytest.mark.unit
class TestEncodeHolidayTypes:
    """Tests for encode_holiday_types function."""

    def test_single_type_encoded(self):
        df = pd.DataFrame({"date": ["2023-01-01"], "types": [["Public"]]})
        result_df, columns = encode_holiday_types(df)

        assert "holiday_public" in columns
        assert result_df["holiday_public"].iloc[0] == 1

    def test_multiple_types_encoded(self):
        df = pd.DataFrame({"date": ["2023-01-01"], "types": [["Public", "Bank"]]})
        result_df, columns = encode_holiday_types(df)

        assert result_df["holiday_public"].iloc[0] == 1
        assert result_df["holiday_bank"].iloc[0] == 1

    def test_all_default_types_create_columns(self):
        df = pd.DataFrame({"date": ["2023-01-01"], "types": [["Public"]]})
        _, columns = encode_holiday_types(df)

        expected = [f"holiday_{t.lower()}" for t in DEFAULT_HOLIDAY_TYPES]
        assert columns == expected

    def test_absent_types_are_zero(self):
        df = pd.DataFrame({"date": ["2023-01-01"], "types": [["Public"]]})
        result_df, _ = encode_holiday_types(df)

        assert result_df["holiday_bank"].iloc[0] == 0
        assert result_df["holiday_school"].iloc[0] == 0

    def test_multiple_rows(self):
        df = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-12-25"],
                "types": [["Public"], ["Public", "Bank"]],
            }
        )
        result_df, _ = encode_holiday_types(df)

        assert len(result_df) == 2
        assert result_df["holiday_public"].sum() == 2
        assert result_df["holiday_bank"].iloc[1] == 1

    def test_empty_types_all_zero(self):
        df = pd.DataFrame({"date": ["2023-03-15"], "types": [[]]})
        result_df, _ = encode_holiday_types(df)

        for col in [f"holiday_{t.lower()}" for t in DEFAULT_HOLIDAY_TYPES]:
            assert result_df[col].iloc[0] == 0

    def test_original_columns_preserved(self):
        df = pd.DataFrame({"date": ["2023-01-01"], "name": ["New Year"], "types": [["Public"]]})
        result_df, _ = encode_holiday_types(df)

        assert "date" in result_df.columns
        assert "name" in result_df.columns


@pytest.mark.unit
class TestMergeHolidaysWithDataframe:
    """Tests for merge_holidays_with_dataframe function."""

    def _make_hourly_df(self, start="2023-01-01", periods=48):
        dates = pd.date_range(start=start, periods=periods, freq="h")
        return pd.DataFrame({"time": dates, "value": range(periods)})

    def test_merges_holiday_columns(self, monkeypatch):
        """Holiday columns should be added and filled with 0/1."""
        df = self._make_hourly_df()

        monkeypatch.setattr(
            "u2demo_time_series_forecasting.features.holidays.reverse_geocode",
            lambda latlon: {"country_code": "de", "region_code": "DE-BW"},
        )

        holiday_data = pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "localName": ["Neujahr"],
                "types": [["Public"]],
                "counties": [None],
            }
        )
        monkeypatch.setattr(
            "u2demo_time_series_forecasting.features.holidays.fetch_holidays_by_year",
            lambda cc, year: holiday_data,
        )

        result_df, holiday_cols = merge_holidays_with_dataframe(df, (48.7758, 9.1829), "time")

        assert len(holiday_cols) == len(DEFAULT_HOLIDAY_TYPES)
        assert all(col in result_df.columns for col in holiday_cols)

    def test_non_holiday_dates_are_zero(self, monkeypatch):
        """Days without holidays should have all-zero columns."""
        df = self._make_hourly_df(start="2023-03-15", periods=24)

        monkeypatch.setattr(
            "u2demo_time_series_forecasting.features.holidays.reverse_geocode",
            lambda latlon: {"country_code": "de", "region_code": "DE-BW"},
        )
        monkeypatch.setattr(
            "u2demo_time_series_forecasting.features.holidays.fetch_holidays_by_year",
            lambda cc, year: pd.DataFrame({"date": ["2023-01-01"], "types": [["Public"]], "counties": [None]}),
        )

        result_df, holiday_cols = merge_holidays_with_dataframe(df, (48.7758, 9.1829), "time")

        for col in holiday_cols:
            assert (result_df[col] == 0).all()

    def test_preserves_original_row_count(self, monkeypatch):
        """Merging holidays should not add or remove rows."""
        df = self._make_hourly_df(periods=72)

        monkeypatch.setattr(
            "u2demo_time_series_forecasting.features.holidays.reverse_geocode",
            lambda latlon: {"country_code": "de", "region_code": None},
        )
        monkeypatch.setattr(
            "u2demo_time_series_forecasting.features.holidays.fetch_holidays_by_year",
            lambda cc, year: pd.DataFrame({"date": ["2023-01-01"], "types": [["Public"]], "counties": [None]}),
        )

        result_df, _ = merge_holidays_with_dataframe(df, (48.7758, 9.1829), "time")

        assert len(result_df) == len(df)

    def test_region_filtering(self, monkeypatch):
        """Holidays should be filtered by region when region_code is known."""
        df = self._make_hourly_df(start="2023-01-06", periods=24)

        monkeypatch.setattr(
            "u2demo_time_series_forecasting.features.holidays.reverse_geocode",
            lambda latlon: {"country_code": "de", "region_code": "DE-BW"},
        )

        holidays = pd.DataFrame(
            {
                "date": ["2023-01-06", "2023-01-06"],
                "types": [["Public"], ["Public"]],
                "counties": [["DE-BW", "DE-BY"], ["DE-NW"]],
            }
        )
        monkeypatch.setattr(
            "u2demo_time_series_forecasting.features.holidays.fetch_holidays_by_year",
            lambda cc, year: holidays,
        )

        result_df, _ = merge_holidays_with_dataframe(df, (48.7758, 9.1829), "time")

        assert result_df["holiday_public"].iloc[0] == 1

    def test_removes_temporary_columns(self, monkeypatch):
        """date and year helper columns should not remain in output."""
        df = self._make_hourly_df(periods=24)

        monkeypatch.setattr(
            "u2demo_time_series_forecasting.features.holidays.reverse_geocode",
            lambda latlon: {"country_code": "de", "region_code": None},
        )
        monkeypatch.setattr(
            "u2demo_time_series_forecasting.features.holidays.fetch_holidays_by_year",
            lambda cc, year: pd.DataFrame({"date": ["2023-01-01"], "types": [["Public"]], "counties": [None]}),
        )

        result_df, _ = merge_holidays_with_dataframe(df, (48.7758, 9.1829), "time")

        assert "date" not in result_df.columns
        assert "year" not in result_df.columns


@pytest.mark.unit
class TestFetchHolidaysByYear:
    """Tests for fetch_holidays_by_year function."""

    def test_returns_dataframe(self, monkeypatch):
        """Should return a DataFrame from the API response."""
        holiday_json = [
            {"date": "2023-01-01", "localName": "Neujahr", "types": ["Public"], "counties": None},
            {"date": "2023-12-25", "localName": "Weihnachten", "types": ["Public"], "counties": None},
        ]

        class MockResp:
            def raise_for_status(self):
                pass

            def json(self):
                return holiday_json

        monkeypatch.setattr(
            "u2demo_time_series_forecasting.features.holidays.requests.get", lambda url, **kwargs: MockResp()
        )

        # Bypass the joblib cache and tenacity retry decorators
        inner = fetch_holidays_by_year.__wrapped__.__wrapped__

        result = inner("de", "2023")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "date" in result.columns


@pytest.mark.unit
class TestHolidayMergeRowIntegrity:
    """Verify that merging holidays never changes the time-series row count."""

    def test_multiple_holidays_on_same_date_preserves_row_count(self, monkeypatch):
        """When a date has multiple holiday entries (e.g. New Year is both
        Public and Bank), the left merge should not duplicate time-series rows.

        The holiday DataFrame should be aggregated per date before joining so
        that each date maps to exactly one row of encoded holiday indicators.
        """
        monkeypatch.setattr(
            "u2demo_time_series_forecasting.features.holidays.reverse_geocode",
            lambda latlon: {"country_code": "DE", "region_code": "DE-BW"},
        )

        holiday_data = pd.DataFrame(
            [
                {"date": "2023-01-01", "types": ["Public"], "counties": None, "name": "New Year"},
                {"date": "2023-01-01", "types": ["Bank"], "counties": None, "name": "New Year Bank"},
                {"date": "2023-12-25", "types": ["Public"], "counties": None, "name": "Christmas"},
            ]
        )
        monkeypatch.setattr(
            "u2demo_time_series_forecasting.features.holidays.fetch_holidays_by_year",
            lambda country_code, year: holiday_data,
        )

        dates = pd.date_range(start="2023-01-01", periods=48, freq="h")
        df = pd.DataFrame({"time": dates, "value": range(len(dates))})

        result_df, _ = merge_holidays_with_dataframe(df, (48.7758, 9.1829), "time")

        assert len(result_df) == len(df)


@pytest.mark.integration
class TestNagerDateAPISchema:
    """Lightweight check that Nager.Date API response schema hasn't drifted."""

    def test_response_has_expected_fields(self):
        """Public holiday API response should contain the fields the code parses."""
        response = requests.get(f"{PUBLIC_HOLIDAY_API_URL}/publicholidays/2024/DE", timeout=10)
        response.raise_for_status()
        holidays = response.json()

        assert isinstance(holidays, list), "Expected a list of holidays"
        assert len(holidays) > 0, "Expected at least one holiday"

        first = holidays[0]
        required_fields = {"date", "localName", "types"}
        assert required_fields.issubset(first.keys()), f"Missing fields: {required_fields - first.keys()}"
        assert isinstance(first["types"], list)
        assert "counties" in first
