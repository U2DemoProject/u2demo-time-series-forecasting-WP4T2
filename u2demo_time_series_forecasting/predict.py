import os
from datetime import UTC, datetime

from u2demo_time_series_forecasting.constants.columns import (
    TIME_SERIES_GROUP_COLUMN_NAME,
    TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)
from u2demo_time_series_forecasting.constants.paths import MODEL_DIR
from u2demo_time_series_forecasting.data_models import Forecast, ForecastedTimeValue, ForecastInput, ForecastSeries
from u2demo_time_series_forecasting.dataset import U2TimeSeriesForecastingDataset
from u2demo_time_series_forecasting.features.pipeline import apply_features
from u2demo_time_series_forecasting.model import U2TimeSeriesForecastingModel


def predict(data: ForecastInput) -> Forecast:
    """
    Run a full time-series forecasting pipeline.

    This function:
    - Loads a previously trained U2Demo time-series forecasting model
    - Ensures the model and data use the same timestamp frequency
    - Extends the dataset by the forecast horizon
    - Applies all required feature-engineering steps (temporal, lag, holiday, weather)
    - Splits past and future data for forecasting
    - Runs the model's `predict()` method

    Parameters
    ----------
    data : ForecastInput
        The input data for making forecasts.

    Returns
    -------
    Forecast
        A `Forecast` object containing the forecasted time series and associated metadata.

    Raises
    ------
    ValueError
        If the model frequency does not match the time-series frequency, if the
        requested forecast horizon does not match the model's prediction length,
        or if feature engineering or enrichment steps encounter invalid inputs.

    """
    # Load the model early because we will use some of its properties to enrich the input data
    model_path = os.path.join(
        MODEL_DIR,
        f"{data.model_id}_{data.model_version}",
    )

    model = U2TimeSeriesForecastingModel(model_path, require_version_match=True)
    model.load()
    if model.metadata is None:
        raise RuntimeError("Model metadata is not loaded")

    features = list(model.metadata.added_columns.keys())

    time_series = U2TimeSeriesForecastingDataset.from_forecast_input(data)

    time_series.drop_short_timeseries(model.metadata.min_timeseries_length)

    # Model is trained on a specific frequency so it must be verified
    if model.freq != time_series.freq:
        error = f"Model frequency (`{model.freq}`) does not match time series frequency (`{time_series.freq}`)"
        raise ValueError(error)

    if time_series.freq_minutes is None:
        raise RuntimeError("Frequency minutes is not initialized")
    horizon_steps = max(1, data.time_parameters.horizon_hours * 60 // time_series.freq_minutes)
    if horizon_steps != model.prediction_length:
        raise ValueError(
            f"Requested forecast horizon ({horizon_steps} steps) does not match model's prediction length ({model.prediction_length} steps)."
        )

    apply_features(
        time_series,
        features,
        lags=model.metadata.lags,
        windows=model.metadata.windows,
    )

    time_series.split_past_and_future_dfs(horizon_steps)

    quantile_levels = model.quantile_levels

    forecast_df = model.predict(time_series)
    forecast_df = (
        forecast_df.reset_index()
        .drop(columns=[TIME_SERIES_GROUP_COLUMN_NAME])
        .rename(columns={"timestamp": TIME_SERIES_TIMESTAMP_COLUMN_NAME, "mean": "value"})
    )
    forecast_df[TIME_SERIES_TIMESTAMP_COLUMN_NAME] = forecast_df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].dt.strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )

    records = forecast_df.to_dict("records")
    forecasts = [
        ForecastedTimeValue(
            time=r[TIME_SERIES_TIMESTAMP_COLUMN_NAME],
            value=r[TIME_SERIES_PREDICT_VALUE_COLUMN_NAME],
            quantiles={str(q): r[str(q)] for q in quantile_levels},
        )
        for r in records
    ]

    return Forecast(
        forecast_type=data.forecast_type,
        forecast_date=datetime.now(UTC),
        time_parameters=data.time_parameters,
        forecast_series=ForecastSeries(
            series_id=data.target_series.series_id,
            variable_name=data.target_series.variable_name,
            asset_id=data.target_series.asset_id,
            asset_type=data.target_series.asset_type,
            unit=data.target_series.unit,
            unitIri=data.target_series.unitIri,
            values=forecasts,
        ),
    )
