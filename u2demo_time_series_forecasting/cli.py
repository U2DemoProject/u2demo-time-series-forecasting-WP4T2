import json
import os

from fire import Fire

from u2demo_time_series_forecasting.constants.paths import OUTPUT_DIR
from u2demo_time_series_forecasting.core.logger import logger
from u2demo_time_series_forecasting.data_models_v2 import ForecastInputV2
from u2demo_time_series_forecasting.diagnostic import sanity_check
from u2demo_time_series_forecasting.predict import predict, predict_v2
from u2demo_time_series_forecasting.train import train_from_payload
from u2demo_time_series_forecasting.versioning import PayloadVersion, parse_forecast_payload


def predict_command(data_json_path: str, version: PayloadVersion | None = None) -> None:
    """
    Run a full time-series forecasting pipeline from a JSON input file and write forecasts to disk.

    This function:
    - Loads the input and validates it against ``ForecastInput`` (v1) or ``ForecastInputV2`` (v2)
    - Loads the matching trained U2Demo time-series forecasting model
    - Ensures the model and data use the same timestamp frequency
    - Extends the dataset by the forecast horizon
    - Replays the feature-engineering steps recorded at training time
    - Splits past and future data for forecasting
    - Runs the model's ``predict()`` method
    - Saves the resulting forecast, in the data model version of the input, to a JSON file under ``OUTPUT_DIR``

    Parameters
    ----------
    data_json_path : str
        Path to a JSON file matching the ``ForecastInput`` or ``ForecastInputV2`` schema
        (see ``u2demo_time_series_forecasting/data_models.py`` and ``data_models_v2.py``).
    version : {"v1", "v2"}, optional
        Data model version of the input. Detected from the payload when omitted.

    Returns
    -------
    None
        The function writes a JSON file containing forecasts.

    """
    with open(data_json_path) as f:
        raw_data = json.load(f)

    # Validate data model
    data = parse_forecast_payload(raw_data, version)

    forecast = predict_v2(data) if isinstance(data, ForecastInputV2) else predict(data)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    forecast_date_str = forecast.forecast_date.strftime("%Y-%m-%dT%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"{data.model_id}_{data.model_version}_{forecast_date_str}.json")
    with open(output_path, "w") as f:
        f.write(forecast.model_dump_json(indent=4))

    logger.info(f"Forecast written to {output_path}")


def train_command(data_json_path: str, version: PayloadVersion | None = None) -> None:
    """
    Train a new U2Demo time-series forecasting model from a JSON input file.

    This function:
    - Loads the input and validates it against ``ForecastTrainingInput`` (v1) or ``ForecastTrainingInputV2`` (v2)
    - Engineers the requested temporal, lag, holiday, and weather features
    - Constructs and fits an AutoGluon ``TimeSeriesPredictor``
    - Optionally saves average leaderboards over rolling test cutoffs

    Parameters
    ----------
    data_json_path : str
        Path to a JSON file matching the ``ForecastTrainingInput`` or ``ForecastTrainingInputV2`` schema
        (see ``u2demo_time_series_forecasting/data_models.py`` and ``data_models_v2.py``).
    version : {"v1", "v2"}, optional
        Data model version of the input. Detected from the payload when omitted.

    Returns
    -------
    None
        Trained model artifacts are saved to disk under ``MODEL_DIR``.

    """
    with open(data_json_path) as f:
        raw_data = json.load(f)

    train_from_payload(raw_data, version)


def main() -> None:
    """Command-line interface for U2Demo time-series forecasting model training and inference."""
    Fire(
        {
            "sanity_check": sanity_check,
            "predict": predict_command,
            "train": train_command,
        }
    )


if __name__ == "__main__":
    main()
