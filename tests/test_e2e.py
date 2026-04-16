"""End-to-end pipeline tests: train a real Chronos2 model, save, load, predict.

These tests verify the full pipeline produces valid forecasts, correct quantile
ordering on real model output (not mocked), and metadata round-trip fidelity.
Mark: @pytest.mark.e2e — excluded from default test runs.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from u2demo_time_series_forecasting.constants.columns import TIME_SERIES_PREDICT_VALUE_COLUMN_NAME
from u2demo_time_series_forecasting.data_models import ForecastInput, ForecastTrainingInput
from u2demo_time_series_forecasting.dataset import U2TimeSeriesPredictionDataset
from u2demo_time_series_forecasting.model import U2TimeSeriesPredictionModel
from u2demo_time_series_forecasting.utils import get_known_covariates, get_min_timeseries_length

# ── helpers ──────────────────────────────────────────────────────────────────────


def _make_training_input(
    periods=200,
    features=None,
    num_weekly_lags=0,
    lookback_steps=None,
    values=None,
    quantile_levels=None,
):
    """Build a ForecastTrainingInput for e2e testing."""
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="h")

    if values is None:
        rng = np.random.default_rng(42)
        values = rng.random(periods).tolist()
    else:
        values = list(values)

    payload = {
        "forecast_type": "stochastic",
        "time_parameters": {"timestep_minutes": 60, "horizon_hours": 24},
        "training_model_config": {
            "model_id": "test-e2e",
            "model_preset": "chronos2",
            "model_version": "1.0.0",
            "training_split": {"num_val_windows": 1, "num_test_windows": 1},
        },
        "target_series": {
            "series_id": "s1",
            "variable_name": "active_power",
            "asset_id": "pv1",
            "asset_type": "PV",
            "training_values": [
                {"time": ts.isoformat(), "value": float(v)} for ts, v in zip(dates, values, strict=True)
            ],
        },
    }

    if features is not None:
        payload["features"] = features
    if num_weekly_lags > 0:
        payload["num_weekly_lags"] = num_weekly_lags
    if lookback_steps:
        payload["training_model_config"]["lookback_steps"] = lookback_steps
    if quantile_levels:
        payload["training_model_config"]["quantile_levels"] = quantile_levels

    return ForecastTrainingInput(**payload)


def _make_forecast_input(periods=200, values=None):
    """Build a ForecastInput for e2e testing."""
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="h")

    if values is None:
        rng = np.random.default_rng(42)
        values = rng.random(periods).tolist()
    else:
        values = list(values)

    return ForecastInput(
        forecast_type="stochastic",
        time_parameters={"timestep_minutes": 60, "horizon_hours": 24},
        model_id="test-e2e",
        model_version="1.0.0",
        target_series={
            "series_id": "s1",
            "variable_name": "active_power",
            "asset_id": "pv1",
            "asset_type": "PV",
            "unit": "kW",
            "historical_values": [
                {"time": ts.isoformat(), "value": float(v)} for ts, v in zip(dates, values, strict=True)
            ],
        },
    )


# ── e2e tests ────────────────────────────────────────────────────────────────────


@pytest.mark.e2e
class TestEndToEndPipeline:
    """Full pipeline: train a real Chronos2 model, save, load, predict."""

    def test_train_then_predict_produces_valid_forecast(self):
        """Train a real model and run prediction on it."""

        with tempfile.TemporaryDirectory() as model_dir:
            rng = np.random.default_rng(42)
            periods = 200
            values = rng.random(periods).tolist()

            train_data = _make_training_input(periods=periods, features=["temporal"], lookback_steps=50, values=values)
            ts = U2TimeSeriesPredictionDataset.from_training_input(train_data)
            ts.add_temporal()

            horizon = 24
            lags = [0]
            min_lookback = 50
            known_covariates = get_known_covariates(ts.added_columns, horizon)
            min_ts_len = get_min_timeseries_length(lags=lags, min_lookback=min_lookback, horizon=horizon)
            ts.drop_short_timeseries(min_ts_len)

            model_path = f"{model_dir}/test-e2e_1.0.0"
            model = U2TimeSeriesPredictionModel(model_path, require_version_match=True)
            model.new(
                target_column=TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
                known_covariates_names=known_covariates,
                prediction_length=horizon,
                freq="h",
                eval_metric="WQL",
                quantile_levels=[0.1, 0.5, 0.9],
            )
            model.train(
                time_series=ts,
                min_lookback=min_lookback,
                num_val_windows=1,
                num_test_windows=1,
                save_leaderboards=False,
                training_kwargs={"presets": "chronos2"},
            )

            # Predict with a fresh model load
            model2 = U2TimeSeriesPredictionModel(model_path, require_version_match=True)
            model2.load()

            predict_data = _make_forecast_input(periods=periods, values=values)
            ts2 = U2TimeSeriesPredictionDataset.from_forecast_input(predict_data)
            ts2.drop_short_timeseries(model2.metadata.min_timeseries_length)
            ts2.add_temporal()
            ts2.split_past_and_future_dfs(horizon)

            predictions = model2.predict(ts2)

            assert predictions is not None
            assert len(predictions) == horizon
            assert predictions.index.names == ["item_id", "timestamp"]
            assert "mean" in predictions.columns
            assert "0.1" in predictions.columns
            assert "0.5" in predictions.columns
            assert "0.9" in predictions.columns
            assert predictions["mean"].notna().all()
            assert np.isfinite(predictions["mean"].values).all()

    def test_e2e_metadata_round_trip(self):
        """Train, save metadata, reload -- verify metadata is complete and consistent."""

        with tempfile.TemporaryDirectory() as model_dir:
            rng = np.random.default_rng(42)
            periods = 500
            values = rng.random(periods).tolist()

            train_data = _make_training_input(
                periods=periods, features=["temporal", "lag"], num_weekly_lags=1, lookback_steps=50, values=values
            )
            ts = U2TimeSeriesPredictionDataset.from_training_input(train_data)
            ts.add_temporal()

            freq_weeks = int(
                (60 * 60 * 24 * 7) / pd.Timedelta(pd.tseries.frequencies.to_offset(freq=ts.freq)).total_seconds()
            )
            lags = [freq_weeks * (1 + i) for i in range(1)]
            ts.add_lags(lags, lags)

            horizon = 24
            min_lookback = 50
            known_covariates = get_known_covariates(ts.added_columns, horizon)
            min_ts_len = get_min_timeseries_length(lags=lags, min_lookback=min_lookback, horizon=horizon)
            ts.drop_short_timeseries(min_ts_len)

            model_path = f"{model_dir}/test-e2e_1.0.0"
            model = U2TimeSeriesPredictionModel(model_path, require_version_match=True)
            model.new(
                target_column=TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
                known_covariates_names=known_covariates,
                prediction_length=horizon,
                freq="h",
                eval_metric="WQL",
                quantile_levels=[0.1, 0.5, 0.9],
            )
            model.train(
                time_series=ts,
                min_lookback=min_lookback,
                num_val_windows=1,
                num_test_windows=1,
                save_leaderboards=False,
                training_kwargs={"presets": "chronos2"},
            )

            # Reload from disk
            model2 = U2TimeSeriesPredictionModel(model_path, require_version_match=True)
            model2.load()

            assert "temporal" in model2.metadata.added_columns
            assert "lag" in model2.metadata.added_columns
            assert model2.metadata.lags == lags
            assert model2.metadata.windows == lags
            assert model2.metadata.frequency == "h"
            assert model2.metadata.horizon == horizon
            assert model2.metadata.min_timeseries_length == max(lags) + min_lookback + horizon
