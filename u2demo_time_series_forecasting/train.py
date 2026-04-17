import os

from u2demo_time_series_forecasting.constants.columns import TIME_SERIES_PREDICT_VALUE_COLUMN_NAME
from u2demo_time_series_forecasting.constants.paths import MODEL_DIR
from u2demo_time_series_forecasting.constants.training import DEFAULT_TRAIN_EVAL_METRIC, DEFAULT_TRAIN_MIN_LOOKBACK
from u2demo_time_series_forecasting.data_models import ForecastTrainingInput
from u2demo_time_series_forecasting.dataset import U2TimeSeriesForecastingDataset
from u2demo_time_series_forecasting.features.names import FeatureName
from u2demo_time_series_forecasting.features.pipeline import apply_features
from u2demo_time_series_forecasting.model import U2TimeSeriesForecastingModel
from u2demo_time_series_forecasting.utils import get_known_covariates, get_min_timeseries_length


def train(
    data: ForecastTrainingInput,
) -> None:
    """
    Train a new U2Demo time-series forecasting model.

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
    time_series = U2TimeSeriesForecastingDataset.from_training_input(data)

    model_path = os.path.join(
        MODEL_DIR,
        f"{data.training_model_config.model_id}_{data.training_model_config.model_version}",
    )
    model = U2TimeSeriesForecastingModel(model_path, require_version_match=True)

    features = data.features
    lags: list[int] = [0]
    if FeatureName.LAG in features:
        freq_weeks = 60 * 24 * 7 // time_series.freq_minutes
        computed_lags = [freq_weeks * (1 + i) for i in range(data.num_weekly_lags)]
        # num_weekly_lags=0 produces empty list; normalize for downstream min_timeseries_length
        lags = computed_lags or [0]
    else:
        computed_lags = []

    apply_features(time_series, features, lags=computed_lags, windows=computed_lags)

    # Ensure at least 1 step when horizon < freq (e.g. 12h horizon with daily data)
    if time_series.freq_minutes is None or time_series.freq is None:
        raise RuntimeError("Frequency is not initialized")
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

    training_kwargs: dict[str, str | list[str] | None] = {
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
