import os
from datetime import datetime

from dotenv import load_dotenv
from fire import Fire

from u2.dataset import U2TimeSeriesPredictionDataset
from u2.defaults import (
    DEFAULT_OUTPUT_PREDICTION_PATH,
    DEFAULT_TIME_SERIES_FEATURES,
    DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME,
    DEFAULT_TIME_SERIES_LOCATION_COLUMN_NAME,
    DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME,
    DEFAULT_TRAIN_EVAL_METRIC,
    DEFAULT_TRAIN_MIN_LOOKBACK,
    DEFAULT_TRAIN_NUM_TEST_WINDOWS,
    TIMESTEPS_IN_WEEKLY_CYCLE,
)
from u2.diagnostic import sanity_check
from u2.logging_config import logger
from u2.model import U2TimeSeriesPredictionModel
from u2.utils import get_known_covariates, get_min_group_length


def predict(
    time_series_csv: str,
    model_path: str,
    output_prediction_csv: str = None,
    location_col_name: str = DEFAULT_TIME_SERIES_LOCATION_COLUMN_NAME,
    timestamp_col_name: str = DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME,
    predict_value_col_name: str = DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    group_col_name: str = DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME,
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
    horizon = model.predictor.prediction_length

    if from_csv_kwargs is None:
        from_csv_kwargs = {}
    time_series = U2TimeSeriesPredictionDataset.from_csv(
        time_series_csv,
        location_col_name,
        timestamp_col_name,
        predict_value_col_name,
        group_col_name,
        **from_csv_kwargs,
    )

    time_series.drop_short_groups(model.metadata["min_group_length"])

    # Model is trained on a specific frequency so it must be verified
    if model.predictor.freq != time_series.freq:
        error = (
            f"Model frequency (`{model.predictor.freq}`) does not match time series frequency (`{time_series.freq}`)"
        )
        logger.error(error)
        raise AssertionError(error)

    time_series.extend_by_horizon(horizon)
    if "temporal" in model.metadata["added_columns"]:
        time_series.add_temporal()
    if "lag" in model.metadata["added_columns"]:
        time_series.add_lags(model.metadata["lags"], model.metadata["windows"])
    if "holiday" in model.metadata["added_columns"]:
        time_series.add_holidays()
    if "weather" in model.metadata["added_columns"]:
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


def train(
    time_series_csv: str,
    model_path: str,
    horizon: int,
    num_weekly_lags: int,
    features: list[str] = DEFAULT_TIME_SERIES_FEATURES,
    num_test_windows: int = DEFAULT_TRAIN_NUM_TEST_WINDOWS,
    min_lookback: int = DEFAULT_TRAIN_MIN_LOOKBACK,
    location_col_name: str = DEFAULT_TIME_SERIES_LOCATION_COLUMN_NAME,
    timestamp_col_name: str = DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME,
    predict_value_col_name: str = DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    group_col_name: str = DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME,
    save_leaderboards: bool = False,
    from_csv_kwargs: dict = None,
    training_kwargs: dict = None,
) -> None:
    """
    Train a new U2 time-series prediction model from an input CSV.

    This function:
    - Loads and validates the input time-series CSV
    - Engineers temporal, lag, holiday, and weather features
    - Constructs and fits an AutoGluon `TimeSeriesPredictor`
    - Optionally saves average leaderboards over rolling test cutoffs

    Parameters
    ----------
    time_series_csv : str
        Path to the input time-series CSV file.
    model_path : str
        Directory where the trained model and metadata will be stored.
    horizon : int
        Forecast horizon (number of future steps to predict).
    num_weekly_lags : int
        Number of weekly multiples used to derive lag and rolling windows.
    features : list[str]
        Features to include in training, can be chosen from `["temporal", "lag", "holiday", "weather"]`. Default uses
        all.
    num_test_windows : int, optional
        Number of backtesting windows used for evaluation when training. Defaults to
        `DEFAULT_TRAIN_NUM_TEST_WINDOWS`.
    min_lookback : int, optional
        Minimum lookback used to form the training backtest split. Defaults to
        `DEFAULT_TRAIN_MIN_LOOKBACK`.
    location_col_name : str, optional
        Name of the location column in the input CSV.
    timestamp_col_name : str, optional
        Name of the timestamp column in the input CSV.
    predict_value_col_name : str, optional
        Name of the target value column in the input CSV.
    group_col_name : str, optional
        Column identifying independent time-series groups.
    save_leaderboards : bool, optional
        If True, writes averaged leaderboard metrics to the model directory.
    from_csv_kwargs : dict, optional
        Additional keyword arguments passed to `U2TimeSeriesPredictionDataset.from_csv()`.
    training_kwargs : dict, optional
        Additional keyword arguments forwarded to `TimeSeriesPredictor.fit()`.

    Returns
    -------
    None
        Trained model artifacts are saved to disk under `model_path`.

    """
    for feature in features:
        if feature not in DEFAULT_TIME_SERIES_FEATURES:
            raise ValueError(f"Unrecognized time series feature: {feature}")

    model = U2TimeSeriesPredictionModel(model_path, require_version_match=True)

    if from_csv_kwargs is None:
        from_csv_kwargs = {}
    if training_kwargs is None:
        training_kwargs = {}

    time_series = U2TimeSeriesPredictionDataset.from_csv(
        time_series_csv,
        location_col_name,
        timestamp_col_name,
        predict_value_col_name,
        group_col_name,
        **from_csv_kwargs,
    )

    if "lag" in features:
        lags = [TIMESTEPS_IN_WEEKLY_CYCLE[time_series.freq] * (1 + i) for i in range(num_weekly_lags)]
    else:
        lags = [0]
    min_group_length = get_min_group_length(lags=lags, min_lookback=min_lookback)
    time_series.drop_short_groups(min_group_length)

    if "temporal" in features:
        time_series.add_temporal()
    if "lag" in features:
        time_series.add_lags(
            lags,
            lags,  #  Windows does not necessarily need to be the same as lags but up to now it was not found to be necessary for a different value
        )
    if "holiday" in features:
        time_series.add_holidays()
    if "weather" in features:
        time_series.add_weather()

    known_covariates = get_known_covariates(time_series.added_columns, horizon)

    model.new(
        DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
        known_covariates,
        horizon,
        time_series.freq,
        DEFAULT_TRAIN_EVAL_METRIC,
    )

    model.train(
        time_series,
        min_lookback=min_lookback,
        num_test_windows=num_test_windows,
        save_leaderboards=save_leaderboards,
        training_kwargs=training_kwargs,
    )


def main():  # NOQA: D103, ANN201
    Fire(
        {
            "sanity_check": sanity_check,
            "predict": predict,
            "train": train,
        }
    )


if __name__ == "__main__":
    load_dotenv()
    main()
