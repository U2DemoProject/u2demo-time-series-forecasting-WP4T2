DEFAULT_NOMINATIM_USER_AGENT_NAME = "U2 Demo Time Series Prediction"
DEFAULT_WEATHER_ARCHIVE_ENDPOINT_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_OUTPUT_PREDICTION_PATH = "./output"

DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME = "timestamp"
DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME = "value"
DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME = "item_id"
DEFAULT_LOCATION_COLUMN_NAME = "location"

DEFAULT_KNOWN_COVARIATE_NAMES = [
    # Below are from the holiday API: https://date.nager.at/api/v3/publicholidays/{year}/{country_code}
    # See holiday_fetcher.py
    "holiday_public",
    "holiday_bank",
    "holiday_school",
    "holiday_authorities",
    "holiday_optional",
    "holiday_observance",
    # See temporal.py
    "day_of_week_0",
    "day_of_week_3",
    "day_of_week_2",
    "day_of_week_4",
    "day_of_week_5",
    "day_of_week_6",
    "day_of_week_1",
    "day_of_year_spline_3",
    "day_of_year_spline_6",
    "day_of_year_spline_7",
    "day_of_year_spline_1",
    "day_of_year_spline_8",
    "day_of_year_spline_11",
    "day_of_year_spline_5",
    "day_of_year_spline_4",
    "day_of_year_spline_10",
    "day_of_year_spline_0",
    "day_of_year_spline_2",
    "day_of_year_spline_9",
    "month_sin",
    "month_cos",
    # See weather_fetcher.py
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "cloudcover_mean",
    "shortwave_radiation_sum",
    # See lag.py
    "Energy__kWh__lag_21",
    "Energy__kWh__lag_28",
]
DEFAULT_HORIZON = 24
DEFAULT_GRANULARITY = "1d"
DEFAULT_EVAL_METRIC = "MAE"
DEFAULT_HOLIDAY_TYPES = [
    "Public",
    "Bank",
    "School",
    "Authorities",
    "Optional",
    "Observance",
]
DEFAULT_WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "windspeed_10m",
    "precipitation",
    "cloudcover",
    "shortwave_radiation",
]
DEFAULT_WEATHER_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "cloudcover_mean",
    "shortwave_radiation_sum",
]
DEFAULT_LAGS = [24, 48]
DEFAULT_LAG_WINDOWS = [24]
