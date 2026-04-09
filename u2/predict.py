import os
from datetime import UTC, datetime

from u2.data_models import Forecast, ForecastInput, ForecastSeries
from u2.dataset import U2TimeSeriesPredictionDataset
from u2.defaults import (
    MODEL_PATH,
    TIME_SERIES_GROUP_COLUMN_NAME,
    TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)
from u2.model import U2TimeSeriesPredictionModel


def predict(data: ForecastInput) -> Forecast:
    """
    Run a full time-series forecasting pipeline.

    This function:
    - Loads a previously trained U2 time-series prediction model
    - Ensures the model and data use the same timestamp frequency
    - Extends the dataset by the forecast horizon
    - Applies all required feature-engineering steps (temporal, lag, holiday, weather)
    - Splits past and future data for prediction
    - Runs the model's `predict()` method

    Parameters
    ----------
    data : ForecastInput
        The input data for making predictions.

    Returns
    -------
    Forecast
        A `Forecast` object containing the forecasted time series and associated metadata.

    Raises
    ------
    AssertionError
        If the model frequency does not match the time-series frequency.
    ValueError
        If feature engineering or enrichment steps encounter invalid inputs.


    """
    # Load the model early because we will use some of its properties to enrich the input data
    model_path = os.path.join(
        MODEL_PATH,
        f"{data.model_id}_{data.model_version}",
    )

    model = U2TimeSeriesPredictionModel(model_path, require_version_match=True)
    model.load()

    features = model.metadata.get("added_columns", [])

    time_series = U2TimeSeriesPredictionDataset.from_forecast_input(data)

    time_series.drop_short_timeseries(model.metadata["min_timeseries_length"])

    # Model is trained on a specific frequency so it must be verified
    if model.predictor.freq != time_series.freq:
        error = (
            f"Model frequency (`{model.predictor.freq}`) does not match time series frequency (`{time_series.freq}`)"
        )
        raise AssertionError(error)

    horizon_steps = data.time_parameters.horizon_hours * 60 // time_series.freq_minutes
    if horizon_steps != model.predictor.prediction_length:
        raise AssertionError(
            f"Requested forecast horizon ({horizon_steps} steps) does not match model's prediction length ({model.predictor.prediction_length} steps)."
        )

    if "temporal" in features:
        time_series.add_temporal()
    if "lag" in features:
        time_series.add_lags(model.metadata["lags"], model.metadata["windows"])
    if "holiday" in features:
        time_series.add_holidays()
    if "weather" in features:
        time_series.add_weather()

    time_series.split_past_and_future_dfs(horizon_steps)

    quantile_levels = model.predictor.quantile_levels

    prediction_df = model.predict(time_series)
    prediction_df = (
        prediction_df.reset_index()
        .drop(columns=[TIME_SERIES_GROUP_COLUMN_NAME])
        .rename(columns={"timestamp": TIME_SERIES_TIMESTAMP_COLUMN_NAME, "mean": "value"})
    )
    prediction_df[TIME_SERIES_TIMESTAMP_COLUMN_NAME] = prediction_df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].dt.strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )

    records = prediction_df.to_dict("records")
    prediction = [
        {
            TIME_SERIES_TIMESTAMP_COLUMN_NAME: r[TIME_SERIES_TIMESTAMP_COLUMN_NAME],
            TIME_SERIES_PREDICT_VALUE_COLUMN_NAME: r[TIME_SERIES_PREDICT_VALUE_COLUMN_NAME],
            "quantiles": {str(q): r[str(q)] for q in quantile_levels},
        }
        for r in records
    ]

    return Forecast(
        forecast_type=data.forecast_type,
        forecast_date=datetime.now(UTC).isoformat(),
        time_parameters=data.time_parameters,
        forecast_series=ForecastSeries(
            series_id=data.target_series.series_id,
            variable_name=data.target_series.variable_name,
            asset_id=data.target_series.asset_id,
            asset_type=data.target_series.asset_type,
            unit=data.target_series.unit,
            unitIri=data.target_series.unitIri,
            values=prediction,
        ),
    )
