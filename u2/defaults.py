DEFAULT_NOMINATIM_USER_AGENT_NAME = "U2 Demo Time Series Prediction"
DEFAULT_WEATHER_ARCHIVE_ENDPOINT_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_OUTPUT_PREDICTION_PATH = "./output"
DEFAULT_MODEL_TRAINING_RESULT_LEADERBOARD_PATH = "u2/training_result_leaderboard.csv"
DEFAULT_MODEL_METADATA_PATH = "u2/metadata.json"

DEFAULT_TIME_SERIES_FEATURES = ["temporal", "lag", "holiday", "weather"]
DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME = "timestamp"
DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME = "value"
DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME = "item_id"
DEFAULT_TIME_SERIES_LOCATION_COLUMN_NAME = "location"
DEFAULT_TRAIN_EVAL_METRIC = "MAE"
DEFAULT_TRAIN_NUM_TEST_WINDOWS = 1
DEFAULT_TRAIN_MIN_LOOKBACK = 1000

TIMESTEPS_IN_WEEKLY_CYCLE = {"h": 168, "D": 7}

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
