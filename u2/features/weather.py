import datetime as dt
import os

import pandas as pd
import requests
from joblib import Memory
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from u2.defaults import DEFAULT_WEATHER_DAILY_VARS, DEFAULT_WEATHER_HOURLY_VARS
from u2.logging_config import logger
from u2.utils import geocode, is_retryable_error

HTTP_GATEWAY_TIMEOUT = 504
memory = Memory(location=os.getenv("CACHE_DIR", ".cache"), verbose=0)


def add_lat_lon_columns(df: pd.DataFrame, location_col: str) -> pd.DataFrame:
    """
    Add latitude and longitude columns by geocoding unique place names.

    This function:
    - Extracts unique non-null place names from the specified column
    - Calls `geocode()` for each place to retrieve coordinates
    - Maps the coordinates back into the DataFrame
    - Returns a copy with populated 'latitude' and 'longitude' columns

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing place identifiers in `location_col`.
    location_col : str
        Column name containing place names to geocode.

    Returns
    -------
    pandas.DataFrame
        A copy of the DataFrame with two new columns:
        - 'latitude'
        - 'longitude'

    """
    df = df.copy()
    unique_places = df[location_col].dropna().unique()
    coords_map = {}

    for place in unique_places:
        geocode_result = geocode(place)
        coords = geocode_result["latitude"], geocode_result["longitude"]
        coords_map[place] = coords if coords else (None, None)

    df["latitude"] = df[location_col].map(lambda p: coords_map.get(p, (None, None))[0])
    df["longitude"] = df[location_col].map(lambda p: coords_map.get(p, (None, None))[1])
    return df


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
        'h' for hourly or 'D' for daily weather data.
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

    if frequency == "h":
        params["hourly"] = ",".join(variables)
    elif frequency == "D":
        params["daily"] = ",".join(variables)
    else:
        raise ValueError(f"Unrecognized frequency `{frequency}`")

    url_env_key = "WEATHER_FORECAST_API_URL" if mode == "forecast" else "WEATHER_HISTORICAL_API_URL"
    url = os.getenv(url_env_key)

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as e:
        raise e

    key = {"h": "hourly", "D": "daily"}[frequency]
    data = payload.get(key, {})
    df = pd.DataFrame(data)
    if not df.empty and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df


def enrich_weather(
    df: pd.DataFrame,
    time_col: str,
    location_col: str,
    present_time: dt.datetime,
    frequency: str,
    variables: list[str] = None,
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
    location_col : str
        Column containing place names to geocode.
    present_time : datetime.datetime
        The point dividing historical vs. future intervals.
    frequency : str
        'h' for hourly time series, 'D' for daily.
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
    df = add_lat_lon_columns(df, location_col)

    if df[["latitude", "longitude"]].isnull().any(axis=None):
        missing = df[df[["latitude", "longitude"]].isnull().any(axis=1)][location_col].unique()
        raise ValueError(f"Could not geocode locations: {missing.tolist()}")

    if not variables:
        variables = DEFAULT_WEATHER_DAILY_VARS if frequency == "D" else DEFAULT_WEATHER_HOURLY_VARS

    df["time_aligned"] = df[time_col].dt.floor("h") if frequency == "h" else df[time_col].dt.normalize()
    enriched_parts = []

    for lat, lon in df[["latitude", "longitude"]].drop_duplicates().itertuples(index=False):
        df_sub = df[(df["latitude"] == lat) & (df["longitude"] == lon)].copy()
        df_past = df_sub[df_sub[time_col] <= present_time]
        df_future = df_sub[df_sub[time_col] > present_time]

        def process_subset(df_chunk: pd.DataFrame, mode: str, lat: float, lon: float) -> pd.DataFrame:
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
            except Exception as e:
                error = f"Failed to fetch weather for ({lat}, {lon}) in mode `{mode}`"
                logger.error(error)
                raise e

            if weather_df.empty:
                error = f"No weather data for ({lat}, {lon}) in mode `{mode}`"
                logger.error(error)
                raise ValueError(error)

            weather_df["time_aligned"] = (
                weather_df["time"].dt.floor("h") if frequency == "h" else weather_df["time"].dt.normalize()
            )
            weather_df = weather_df.drop(columns=["time"])
            return pd.merge(df_chunk, weather_df, on="time_aligned", how="left")

        enriched_parts.append(process_subset(df_past, "historical", lat, lon))
        enriched_parts.append(process_subset(df_future, "forecast", lat, lon))

    final_df = pd.concat(enriched_parts, ignore_index=True)
    return final_df.drop(columns=["time_aligned"]), variables
