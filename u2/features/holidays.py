import pandas as pd
import requests
from joblib import Memory
from sklearn.preprocessing import MultiLabelBinarizer
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from u2.defaults import CACHE_DIR, DEFAULT_HOLIDAY_TYPES, PUBLIC_HOLIDAY_API_URL
from u2.utils import is_retryable_error, reverse_geocode

memory = Memory(location=CACHE_DIR, verbose=0)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(is_retryable_error),
    reraise=True,
)
@memory.cache
def fetch_holidays_by_year(country_code: str, year: str) -> pd.DataFrame:
    """
    Retrieve public holidays for a given country and year from the Nager.Date API.

    This function:
    - Calls the public holiday API configured via `PUBLIC_HOLIDAY_API_URL`
    - Automatically retries on retryable HTTP errors (429, 504)
    - Caches results on disk to avoid repeated API calls
    - Returns the raw holiday records as a pandas DataFrame

    Parameters
    ----------
    country_code : str
        ISO 3166-1 alpha-2 country code (e.g., "DE", "US").
    year : str
        Year for which holidays should be fetched (e.g., "2024").

    Returns
    -------
    pandas.DataFrame
        DataFrame containing holiday metadata for the requested year.

    Raises
    ------
    requests.exceptions.RequestException
        If the HTTP request fails after retries.

    """
    api_url = f"{PUBLIC_HOLIDAY_API_URL}/publicholidays/{year}/{country_code}"
    response = requests.get(api_url)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def encode_holiday_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    One-hot encode the 'types' column of a holiday DataFrame.

    Uses a `MultiLabelBinarizer` preconfigured with the project-wide
    `DEFAULT_HOLIDAY_TYPES` list. Produces columns named
    `holiday_<type>` (lowercased).

    Parameters
    ----------
    df : pandas.DataFrame
        Holiday DataFrame containing a 'types' column with lists of holiday type labels.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        A tuple containing:
        - The input DataFrame with added one-hot encoded columns
        - A list of the new column names

    """
    holiday_types = DEFAULT_HOLIDAY_TYPES
    column_names = [f"holiday_{t.lower()}" for t in DEFAULT_HOLIDAY_TYPES]

    encoder = MultiLabelBinarizer(classes=holiday_types)
    encoded = encoder.fit_transform(df["types"])

    encoded_df = pd.DataFrame(encoded, columns=column_names, index=df.index)
    return pd.concat([df, encoded_df], axis=1), column_names


def merge_holidays_with_dataframe(
    df: pd.DataFrame, latlon: tuple[float, float], datetime_column: str = "timestamp"
) -> pd.DataFrame:
    """
    Add holiday indicator features to a DataFrame based on dates and a geocoded place.

    This function:
    - Extracts date and year components from the specified datetime column
    - Uses geocoding to determine the place's country and region
    - Fetches holidays for all relevant years
    - Filters holidays by matching region (when available)
    - One-hot encodes holiday types
    - Merges encoded holiday rows into the original DataFrame by date
    - Fills missing holiday indicators with 0

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least one datetime column.
    latlon : tuple[float, float]
        Latitude and longitude coordinates used for geocoding the place.
    datetime_column : str, optional
        Name of the datetime64 column used for merging by date.
        Defaults to "timestamp".

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        A tuple containing:
        - The merged DataFrame with holiday features
        - List of holiday feature column names (`holiday_<type>`)

    """
    date_column = "date"
    year_column = "year"
    df = df.copy()
    df[date_column] = df[datetime_column].dt.date
    df[year_column] = df[datetime_column].dt.year

    geocode_result = reverse_geocode(latlon)
    country_code, region_code = geocode_result["country_code"], geocode_result["region_code"]

    holidays = pd.concat([fetch_holidays_by_year(country_code, str(year)) for year in df[year_column].unique()])

    filtered_holidays = holidays[
        holidays["counties"].apply(lambda counties: region_code in counties if counties else True)
    ]

    encoded_holidays, holiday_columns = encode_holiday_types(filtered_holidays)
    encoded_holidays[date_column] = pd.to_datetime(encoded_holidays[date_column]).dt.date
    encoded_holidays = encoded_holidays[[date_column] + holiday_columns]
    # Collapse multiple holidays on the same date to avoid row duplication during merge
    encoded_holidays = encoded_holidays.groupby(date_column, as_index=False)[holiday_columns].max()

    merged_df = df.merge(encoded_holidays, left_on=date_column, right_on=date_column, how="left")

    merged_df.drop(columns=[date_column, year_column], inplace=True)

    merged_df[holiday_columns] = merged_df[holiday_columns].fillna(0).astype(int)
    return merged_df, [f"holiday_{t.lower()}" for t in DEFAULT_HOLIDAY_TYPES]
