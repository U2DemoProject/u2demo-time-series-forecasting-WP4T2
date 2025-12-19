import os

import requests
from geopy.geocoders import Nominatim
from joblib import Memory
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

memory = Memory(location=os.getenv("CACHE_DIR", ".cache"), verbose=0)


def is_retryable_error(exception: BaseException) -> bool:
    """
    Determine whether an exception should trigger a retry during HTTP requests.

    An error is considered retryable if:
    - It is a `requests.exceptions.RequestException`, and
    - The attached HTTP response has a status code of 504 (Gateway Timeout) or 429 (Too Many Requests).

    Parameters
    ----------
    exception : BaseException
        The exception raised during the request.

    Returns
    -------
    bool
        True if the exception is retryable, False otherwise.

    """
    return (
        isinstance(exception, requests.exceptions.RequestException)
        and getattr(exception.response, "status_code", None) in (504, 429)  # 504 = Gateway Timeout
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(is_retryable_error),
    reraise=True,
)
@memory.cache
def geocode(place_name: str) -> dict:
    """
    Geocode a place name using Nominatim and return country, region, and coordinates.

    This function:
    - Queries the Nominatim geocoding service for the given place name.
    - Extracts country code, sub-region code, latitude, and longitude.
    - Caches results on disk to avoid repeated lookups.
    - Retries failed requests (e.g., HTTP 429 or 504) using exponential backoff.

    Parameters
    ----------
    place_name : str
        A human-readable place name (e.g., "Berlin", "New York City").

    Returns
    -------
    dict
        A dictionary containing:
        - "country_code": ISO 3166-1 alpha-2 country code or None
        - "region_code": ISO 3166-2 region code or None
        - "latitude": float
        - "longitude": float

    Raises
    ------
    ValueError
        If the place cannot be geocoded.
    requests.exceptions.RequestException
        If the request repeatedly fails and exceeds retry limits.

    """
    ret = {}

    geolocator = Nominatim(user_agent=os.getenv("NOMINATIM_USER_AGENT_NAME"))
    location = geolocator.geocode(place_name, addressdetails=True)
    if location is None:
        raise ValueError(f"Could not geocode location: {place_name}")

    address = location.raw.get("address", {})
    ret["country_code"], ret["region_code"] = address.get("country_code", None), address.get("ISO3166-2-lvl4", None)
    ret["latitude"], ret["longitude"] = location.latitude, location.longitude

    return ret


def get_known_covariates(time_series_added_columns: dict[str, list[str]], horizon: int) -> list[str]:
    """
    Derive the list of known covariate feature columns for forecasting.

    Given a mapping of feature categories to lists of added column names (as
    produced by the dataset feature engineering pipeline), this function filters
    out plain lag features whose lag value is smaller than the forecast horizon,
    since those lags depend on unknown future target values at prediction time.

    Parameters
    ----------
    time_series_added_columns : dict[str, list[str]]
        Dictionary mapping feature family (e.g., "temporal", "lag", "holiday",
        "weather") to lists of added column names.
    horizon : int
        Forecast horizon (number of future steps to predict).

    Returns
    -------
    list[str]
        List of column names that can be treated as known covariates during
        forecasting.

    Notes
    -----
    - Rolling statistics columns (those starting with "roll_") are kept.
    - Plain lag columns (e.g., "lag_1", "lag_24") are removed if their lag is
      strictly smaller than `horizon`.

    """
    known_covariates = [item for v in time_series_added_columns.values() for item in v]
    if "lag" in time_series_added_columns:
        unknown_lag_columns = []
        for lag_column in time_series_added_columns["lag"]:
            if "lag" in lag_column and "roll" not in lag_column:
                lag = int(lag_column.split("_")[-1])
                if lag < horizon:
                    unknown_lag_columns.append(lag_column)
        known_covariates = list(set(known_covariates) - set(unknown_lag_columns))
    return known_covariates


def get_min_group_length(lags: list[int], min_lookback: int) -> int:
    """
    Gets the minimum group length to have min_lookback steps after adding lags.

    Parameters
    ----------
    lags : list[int]
        List of lag features to be added.
    min_lookback : int
        Minimum amount of steps of historical data to have.

    Returns
    -------
    int
        The minimum group length expected.

    """
    return max(lags) + min_lookback
