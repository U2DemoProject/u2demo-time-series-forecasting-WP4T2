from pydantic_settings import BaseSettings, SettingsConfigDict

from u2demo_time_series_forecasting.constants.paths import BASEDIR


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=BASEDIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"
    nominatim_domain: str = "nominatim.openstreetmap.org"
    nominatim_user_agent_name: str = "u2demo-time-series-forecasting"
    public_holiday_api_url: str = "https://date.nager.at/api/v3"
    weather_historical_api_url: str = "https://archive-api.open-meteo.com/v1/archive"
    weather_forecast_api_url: str = "https://api.open-meteo.com/v1/forecast"


settings = Settings()
