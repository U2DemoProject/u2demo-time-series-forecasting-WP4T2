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


def parse_prefixed_int_columns(columns: list[str], prefix: str) -> list[int]:
    """
    Extract integer suffixes from column names that begin with a given prefix.

    This function filters for column names starting with `prefix`, then splits on
    the first underscore and converts the portion after the underscore into an integer.

    Parameters
    ----------
    columns : list[str]
        A list of column name strings (e.g., ["val_1", "val_2", "other"]).
    prefix : str
        The required prefix (e.g., "val_").

    Returns
    -------
    list[int]
        A list of integers extracted from the matching column names.

    """
    return [int(c.split("_")[1]) for c in columns if c.startswith(prefix)]


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
