import json
import os

from fire import Fire

from u2.data_models import ForecastInput, ForecastTrainingInput
from u2.defaults import OUTPUT_PREDICTION_PATH
from u2.diagnostic import sanity_check
from u2.predict import predict
from u2.train import train


def predict_command(data_json_path: str) -> None:
    """
    Run a full time-series forecasting pipeline on a CSV input file and write predictions to disk.

    This function:
    - Loads a previously trained U2 time-series prediction model
    - Loads and validates the input time-series CSV
    - Ensures the model and data use the same timestamp frequency
    - Extends the dataset by the forecast horizon
    - Applies all required feature-engineering steps (temporal, lag, holiday, weather)
    - Splits past and future data for prediction
    - Runs the model's `predict()` method
    - Saves the resulting forecast to a JSON file

    Parameters
    ----------
    data_json_path : str
        Path to the input data JSON file.

    Returns
    -------
    None
        The function writes a JSON file containing forecasts.

    """
    with open(data_json_path) as f:
        raw_data = json.load(f)

    # Validate data model
    data = ForecastInput(**raw_data)

    forecast = predict(data)

    os.makedirs(OUTPUT_PREDICTION_PATH, exist_ok=True)

    with open(
        os.path.join(OUTPUT_PREDICTION_PATH, f"{data.model_id}_{data.model_version}_{forecast.forecast_date}.json"),
        "w",
    ) as f:
        f.write(forecast.model_dump_json(indent=4))


def train_command(
    data_json_path: str,
) -> None:
    """
    Train a new U2 time-series prediction model from an input JSON data.

    This function:
    - Loads and validates the input data model
    - Engineers temporal, lag, holiday, and weather features
    - Constructs and fits an AutoGluon `TimeSeriesPredictor`
    - Optionally saves average leaderboards over rolling test cutoffs

    Parameters
    ----------
    data_json_path : str
        Path to the input data JSON file.

    Returns
    -------
    None
        Trained model artifacts are saved to disk.

    """
    with open(data_json_path) as f:
        raw_data = json.load(f)

    # Validate data model
    data = ForecastTrainingInput(**raw_data)

    train(data)


def main() -> None:
    """Command-line interface for U2 time-series prediction model training and inference."""
    Fire(
        {
            "sanity_check": sanity_check,
            "predict": predict_command,
            "train": train_command,
        }
    )


if __name__ == "__main__":
    main()
