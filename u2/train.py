import os

import pandas as pd

from u2.data_models import ForecastTrainingInput
from u2.dataset import U2TimeSeriesPredictionDataset
from u2.defaults import (
    AVAILABLE_FEATURES,
    DEFAULT_TRAIN_EVAL_METRIC,
    DEFAULT_TRAIN_MIN_LOOKBACK,
    MODEL_PATH,
    TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
)
from u2.model import U2TimeSeriesPredictionModel
from u2.utils import get_known_covariates, get_min_timeseries_length


def train(
    data: ForecastTrainingInput,
) -> None:
    """
    Train a new U2 time-series prediction model.

    This function:
    - Engineers temporal, lag, holiday, and weather features
    - Constructs and fits an AutoGluon `TimeSeriesPredictor`
    - Optionally saves average leaderboards over rolling test cutoffs

    Parameters
    ----------
    data : ForecastTrainingInput
        The input data for training the model.

    Returns
    -------
    None
        Trained model artifacts are saved to disk under `model_path`.

    """
    time_series = U2TimeSeriesPredictionDataset.from_training_input(data)

    model_path = os.path.join(
        MODEL_PATH,
        f"{data.training_model_config.model_id}_{data.training_model_config.model_version}",
    )
    model = U2TimeSeriesPredictionModel(model_path, require_version_match=True)

    features = data.features if "features" in data.model_fields_set else []

    num_weekly_lags = data.num_weekly_lags if "num_weekly_lags" in data.model_fields_set else 0

    for feature in features:
        if feature not in AVAILABLE_FEATURES:
            raise ValueError(f"Unrecognized time series feature: {feature}")

    if "temporal" in features:
        time_series.add_temporal()
    if "lag" in features:
        freq_weeks = int(
            (60 * 60 * 24 * 7) / pd.Timedelta(pd.tseries.frequencies.to_offset(freq=time_series.freq)).total_seconds()
        )
        lags = [freq_weeks * (1 + i) for i in range(num_weekly_lags)]
        time_series.add_lags(
            lags,
            lags,  #  Windows does not necessarily need to be the same as lags but up to now it was not found to be necessary for a different value
        )
        if not lags:  # num_weekly_lags=0 produces empty list; normalize for downstream min_timeseries_length
            lags = [0]
    else:
        lags = [0]
    if "holiday" in features:
        time_series.add_holidays()
    if "weather" in features:
        time_series.add_weather()

    # Ensure at least 1 step when horizon < freq (e.g. 12h horizon with daily data)
    horizon = max(1, data.time_parameters.horizon_hours * 60 // time_series.freq_minutes)
    min_lookback = data.training_model_config.lookback_steps or DEFAULT_TRAIN_MIN_LOOKBACK

    known_covariate_names = (
        [exog_series.variable_name for exog_series in data.exogenous_series if exog_series.is_known_covariate]
        if data.exogenous_series
        else []
    ) + get_known_covariates(time_series.added_columns, horizon)

    min_timeseries_length = get_min_timeseries_length(lags=lags, min_lookback=min_lookback, horizon=horizon)
    time_series.drop_short_timeseries(min_timeseries_length)

    if data.forecast_type == "deterministic":
        quantile_levels = [0.5]
    elif data.forecast_type == "stochastic" and data.training_model_config.quantile_levels:
        quantile_levels = data.training_model_config.quantile_levels
    else:
        quantile_levels = None

    eval_metric = (
        data.training_model_config.eval_metric.upper()
        if data.training_model_config.eval_metric
        else DEFAULT_TRAIN_EVAL_METRIC
    )

    model.new(
        target_column=TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
        known_covariates_names=known_covariate_names,
        prediction_length=horizon,
        freq=time_series.freq,
        eval_metric=eval_metric,
        quantile_levels=quantile_levels,
    )

    training_kwargs = {
        "presets": data.training_model_config.model_preset,
    }

    if "excluded_model_types" in data.training_model_config.model_fields_set:
        training_kwargs["excluded_model_types"] = data.training_model_config.excluded_model_types

    model.train(
        time_series=time_series,
        min_lookback=min_lookback,
        num_val_windows=data.training_model_config.training_split.num_val_windows,
        num_test_windows=data.training_model_config.training_split.num_test_windows,
        save_leaderboards=True,
        training_kwargs=training_kwargs,
    )
