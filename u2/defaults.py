import os

from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

NOMINATIM_DOMAIN = os.getenv("NOMINATIM_DOMAIN", "nominatim.openstreetmap.org")
NOMINATIM_USER_AGENT_NAME = os.getenv("NOMINATIM_USER_AGENT_NAME", "U2 Demo Time Series Prediction")

PUBLIC_HOLIDAY_API_URL = os.getenv("PUBLIC_HOLIDAY_API_URL", "https://date.nager.at/api/v3")

WEATHER_HISTORICAL_API_URL = os.getenv("WEATHER_HISTORICAL_API_URL", "https://archive-api.open-meteo.com/v1/archive")
WEATHER_FORECAST_API_URL = os.getenv("WEATHER_FORECAST_API_URL", "https://api.open-meteo.com/v1/forecast")

OUTPUT_PREDICTION_PATH = os.getenv("OUTPUT_PREDICTION_PATH", "./output")
MODEL_PATH = os.getenv("MODEL_PATH", "./models")
MODEL_TRAINING_RESULT_LEADERBOARD_PATH = os.getenv(
    "MODEL_TRAINING_RESULT_LEADERBOARD_PATH", "u2/training_result_leaderboard.csv"
)
MODEL_METADATA_PATH = os.getenv("MODEL_METADATA_PATH", "u2/metadata.json")

TIME_SERIES_TIMESTAMP_COLUMN_NAME = "time"
TIME_SERIES_PREDICT_VALUE_COLUMN_NAME = "value"
TIME_SERIES_GROUP_COLUMN_NAME = "item_id"

DEFAULT_TRAIN_EVAL_METRIC = "WQL"
DEFAULT_TRAIN_MIN_LOOKBACK = 1024

AVAILABLE_FEATURES = ["temporal", "lag", "holiday", "weather"]

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
