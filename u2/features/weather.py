import datetime as dt

import pandas as pd
import requests
from joblib import Memory
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from u2.constants.envs import settings
from u2.constants.paths import CACHE_DIR
from u2.constants.weather import DEFAULT_WEATHER_DAILY_VARS, DEFAULT_WEATHER_HOURLY_VARS
from u2.core.logger import logger
from u2.utils import is_retryable_error

memory = Memory(location=CACHE_DIR, verbose=0)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(is_retryable_error),
    reraise=True,
)
def fetch_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    variables: list[str],
    frequency: str,
    mode: str = "historical",  # or "forecast"
) -> pd.DataFrame:
    """
    Fetch weather data from the Open-Meteo API.

    The function:
    - Selects API endpoint based on `mode` ("historical" or "forecast")
    - Chooses hourly or daily variables based on `frequency`
    - Requests weather data over the given date range
    - Converts the `time` field to datetime
    - Returns the weather records as a DataFrame

    Retries are automatically performed for retryable HTTP errors (429, 504).

    Parameters
    ----------
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    start_date : str
        Start date in ISO format (YYYY-MM-DD).
    end_date : str
        End date in ISO format (YYYY-MM-DD).
    variables : list[str]
        List of weather variable names to request.
    frequency : str
        Frequency string (e.g., 'h' for hourly, 'D' for daily) to determine variable defaults and alignment.
    mode : str, optional
        Whether to call historical or forecast endpoint. Defaults to "historical".

    Returns
    -------
    pandas.DataFrame
        Weather data for the requested period and variables.

    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }

    if frequency.endswith(("h", "min")):
        params["hourly"] = ",".join(variables)
        key = "hourly"
    elif frequency.endswith("D"):
        params["daily"] = ",".join(variables)
        key = "daily"
    else:
        raise ValueError(f"Unrecognized frequency `{frequency}`")

    url = settings.weather_forecast_api_url if mode == "forecast" else settings.weather_historical_api_url

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch data from {url}: {e}") from e

    data = payload.get(key, {})
    df = pd.DataFrame(data)
    if not df.empty and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df


def enrich_weather(
    df: pd.DataFrame,
    time_col: str,
    latlon: tuple[float, float],
    present_time: dt.datetime | None,
    frequency: str,
    variables: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Enrich a time-series DataFrame with weather variables aligned per timestamp.

    The enrichment process:
    1. Converts timestamps to datetime and adds lat/lon via geocoding.
    2. Validates that all locations have known coordinates.
    3. Selects default weather variables if none provided
       (daily or hourly depending on series frequency).
    4. Aligns timestamps to the nearest hour or day boundary.
    5. Splits data per location and per historical/forecast intervals.
    6. Fetches relevant weather ranges using `fetch_weather()`.
    7. Merges the weather data back into the DataFrame by aligned timestamps.

    Parameters
    ----------
    df : pandas.DataFrame
        Input time-series DataFrame containing datetime and location columns.
    time_col : str
        Column name containing timestamps.
    latlon : tuple[float, float]
        Latitude and longitude coordinates for the location.
    present_time : datetime.datetime | None
        The point dividing historical vs. future intervals (None if all data is historical).
    frequency : str
        Frequency string (e.g., 'h' for hourly, 'D' for daily) to determine variable defaults and alignment.
    variables : list[str], optional
        Specific weather variable names to fetch. Defaults are based on frequency.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        - The enriched DataFrame with weather variables merged in.
        - The list of weather variable names added.

    Raises
    ------
    ValueError
        If any location cannot be geocoded or if weather API returns no data.

    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="raise")

    # Normalize present_time to a naive UTC datetime so it is directly comparable to both
    # the naive df timestamps and the naive 2022 cutoff below.  Callers may pass a naive
    # datetime (assumed UTC) or a timezone-aware one; strip the tzinfo after converting.
    if present_time is not None and present_time.tzinfo is not None:
        present_time = present_time.astimezone(dt.UTC).replace(tzinfo=None)

    if not variables:
        variables = DEFAULT_WEATHER_DAILY_VARS if frequency.endswith("D") else DEFAULT_WEATHER_HOURLY_VARS

    df["time_aligned"] = df[time_col].dt.floor("h") if frequency.endswith(("h", "min")) else df[time_col].dt.normalize()
    enriched_parts: list[pd.DataFrame] = []

    def _process_subset(df_chunk: pd.DataFrame, mode: str, lat: float, lon: float) -> pd.DataFrame:
        if df_chunk.empty:
            return df_chunk

        min_date = df_chunk[time_col].min().date()
        max_date = df_chunk[time_col].max().date()
        try:
            weather_df = fetch_weather(
                lat=lat,
                lon=lon,
                start_date=min_date.isoformat(),
                end_date=max_date.isoformat(),
                variables=variables,
                frequency=frequency,
                mode=mode,
            )
        except (requests.RequestException, RuntimeError):
            logger.error(f"Failed to fetch weather for ({lat}, {lon}) in mode `{mode}`")
            raise

        if weather_df.empty:
            error = f"No weather data for ({lat}, {lon}) in mode `{mode}`"
            logger.error(error)
            raise ValueError(error)

        weather_df["time_aligned"] = (
            weather_df["time"].dt.floor("h") if frequency.endswith(("h", "min")) else weather_df["time"].dt.normalize()
        )
        weather_df = weather_df.drop(columns=["time"])
        return pd.merge(df_chunk, weather_df, on="time_aligned", how="left")

    # Past weather forecasts from Open Meteo are only available starting from 2022 (UTC).
    # present_time has already been normalized to naive UTC above.
    if present_time is None or present_time < dt.datetime(2022, 1, 1):
        final_df = _process_subset(df, "historical", latlon[0], latlon[1])
    else:
        df_past = df[df[time_col] <= present_time]
        df_future = df[df[time_col] > present_time]

        enriched_parts.append(_process_subset(df_past, "historical", latlon[0], latlon[1]))
        enriched_parts.append(_process_subset(df_future, "forecast", latlon[0], latlon[1]))

        final_df = pd.concat(enriched_parts, ignore_index=True)
    return final_df.drop(columns=["time_aligned"]), variables
