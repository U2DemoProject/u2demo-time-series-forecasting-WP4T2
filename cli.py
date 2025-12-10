import os
from datetime import datetime

from dotenv import load_dotenv
from fire import Fire

from u2.dataset import U2TimeSeriesPredictionDataset
from u2.defaults import (
    DEFAULT_HORIZON,
    DEFAULT_LOCATION_COLUMN_NAME,
    DEFAULT_OUTPUT_PREDICTION_PATH,
    DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME,
    DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)
from u2.diagnostic import sanity_check
from u2.logging_config import logger
from u2.model import U2TimeSeriesPredictionModel
from u2.utils import parse_prefixed_int_columns


def predict(
    time_series_csv: str,
    model_path: str,
    output_prediction_csv: str = None,
    location_col_name: str = DEFAULT_LOCATION_COLUMN_NAME,
    timestamp_col_name: str = DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME,
    predict_value_col_name: str = DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    group_col_name: str = DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME,
    horizon: int = DEFAULT_HORIZON,
    from_csv_kwargs: dict = None,
) -> None:
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
    - Saves the resulting forecast to a CSV file

    Parameters
    ----------
    time_series_csv : str
        Path to the input time-series CSV file.
    model_path : str
        Path to the saved U2 model directory.
    output_prediction_csv : str, optional
        Path where the prediction CSV should be saved.
        If None, a timestamped filename under `DEFAULT_OUTPUT_PREDICTION_PATH` is generated.
    location_col_name : str, optional
        Name of the location column in the input CSV.
    timestamp_col_name : str, optional
        Name of the timestamp column in the input CSV.
    predict_value_col_name : str, optional
        Name of the target value column in the input CSV.
    group_col_name : str, optional
        Column identifying independent time-series groups.
    horizon : int, optional
        Forecast horizon (number of future steps to predict).
    from_csv_kwargs : dict, optional
        Additional keyword arguments passed to `U2TimeSeriesPredictionDataset.from_csv()`.

    Returns
    -------
    None
        The function writes a CSV file containing forecasts.
        Predictions are also logged.

    Raises
    ------
    AssertionError
        If the model frequency does not match the time-series frequency.
    ValueError
        If feature engineering or enrichment steps encounter invalid inputs.
    FileNotFoundError
        If the input CSV file cannot be read.

    """
    # Load the model early because we will use some of its properties to enrich the input data
    model = U2TimeSeriesPredictionModel(model_path, require_version_match=True)
    model.load()

    # this should be read by some kind of model metadata
    lags = [24 * 7, 24 * 14, 24 * 21, 24 * 28]

    if from_csv_kwargs is None:
        from_csv_kwargs = {}
    time_series = U2TimeSeriesPredictionDataset.from_csv(
        time_series_csv,
        max(lags),
        location_col_name,
        timestamp_col_name,
        predict_value_col_name,
        group_col_name,
        **from_csv_kwargs,
    )

    # Model is trained on a specific frequency so it must be verified
    if model.predictor.freq != time_series.freq:
        error = (
            f"Model frequency (`{model.predictor.freq}`) does not match time series frequency (`{time_series.freq}`)"
        )
        logger.error(error)
        raise AssertionError(error)

    time_series.extend_by_horizon(horizon)
    time_series.add_temporal()
    # @Hubert this must somehow dictate the min rows per group right?
    time_series.add_lags(
        parse_prefixed_int_columns(model.predictor.known_covariates_names, "lag_"),
        lags,  # Windows are not known covariates
    )
    time_series.add_holidays()
    time_series.add_weather()
    time_series.split_past_and_future_dfs(horizon)

    if output_prediction_csv is None:
        output_prediction_csv = (
            datetime.now().strftime(f"{DEFAULT_OUTPUT_PREDICTION_PATH}/prediction_%Y%m%d-%H%M%S") + ".csv"
        )
    os.makedirs(os.path.dirname(output_prediction_csv), exist_ok=True)

    prediction_df = model.predict(time_series)
    with open(output_prediction_csv, "w") as f:
        prediction_df.to_csv(f, index=True)
    logger.info(f"Prediction csv saved to {output_prediction_csv}.")


def main():  # NOQA: D103, ANN201
    Fire(
        {
            "sanity_check": sanity_check,
            "predict": predict,
        }
    )


if __name__ == "__main__":
    load_dotenv()
    main()
