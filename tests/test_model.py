import os
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import pandas as pd
import pytest
from autogluon.timeseries import TimeSeriesPredictor
from pandas.testing import assert_frame_equal

from tests.conftest import load_training_input_from_json
from u2.dataset import U2TimeSeriesPredictionDataset
from u2.defaults import TIME_SERIES_GROUP_COLUMN_NAME
from u2.model import U2TimeSeriesPredictionModel


class TestU2TimeSeriesPredictionModel:
    """Tests for U2TimeSeriesPredictionModel class."""

    def test_predict_when_unloaded(self, json_single_timeseries: str):
        model = U2TimeSeriesPredictionModel("/tmp/nonexistent_path")
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)
        with pytest.raises(RuntimeError):
            model.predict(dataset)

    def test_new_initializes_predictor(self, temp_dir: str):
        model_path = os.path.join(temp_dir, "init_model")
        horizon = 82
        model = U2TimeSeriesPredictionModel(model_path)
        model.new(
            target_column="value", known_covariates_names=["x"], prediction_length=horizon, freq="h", eval_metric="MAE"
        )
        assert isinstance(model.predictor, TimeSeriesPredictor)
        assert model.predictor.target == "value"
        assert model.predictor.prediction_length == horizon

    def test_save_and_load(self, temp_dir: str, json_single_timeseries: str):
        horizon = 5
        freq = "h"

        model_path = os.path.join(temp_dir, "saved_model")
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

        model = U2TimeSeriesPredictionModel(model_path)
        model.new(
            target_column="value", known_covariates_names=[], prediction_length=horizon, freq=freq, eval_metric="WQL"
        )
        training_kwargs = {"enable_ensemble": False, "hyperparameters": {"SeasonalNaive": {}}, "random_seed": 54}
        model.train(dataset, min_lookback=24, num_val_windows=1, num_test_windows=0, training_kwargs=training_kwargs)

        new_model = U2TimeSeriesPredictionModel(model_path)
        new_model.load()

        assert new_model.predictor is not None
        assert new_model.predictor.freq == freq
        assert new_model.predictor.prediction_length == horizon
        assert new_model.metadata is not None
        assert new_model.metadata["horizon"] == horizon
        assert new_model.metadata["frequency"] == freq

    def test_predict_returns_correct_format(self, temp_dir: str, json_single_timeseries: str):
        horizon = 5
        freq = "h"

        model_path = os.path.join(temp_dir, "predict_test")
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

        model = U2TimeSeriesPredictionModel(model_path)
        model.new(
            target_column="value", known_covariates_names=[], prediction_length=horizon, freq=freq, eval_metric="WQL"
        )
        training_kwargs = {"enable_ensemble": False, "hyperparameters": {"SeasonalNaive": {}}, "random_seed": 54}
        model.train(dataset, min_lookback=24, num_val_windows=1, num_test_windows=0, training_kwargs=training_kwargs)

        predictions = model.predict(dataset)

        assert isinstance(predictions, pd.DataFrame)

        groups = predictions.index.get_level_values(TIME_SERIES_GROUP_COLUMN_NAME).unique()

        assert len(groups) == len(dataset.df[TIME_SERIES_GROUP_COLUMN_NAME].unique())
        for item_id in groups:
            assert (predictions.index.get_level_values(TIME_SERIES_GROUP_COLUMN_NAME) == item_id).sum() == horizon

    def test_predict_before_and_after_load(self, temp_dir: str, json_single_timeseries: str):
        horizon = 5
        freq = "h"
        random_seed = 54

        model_path = os.path.join(temp_dir, "predict_load")
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

        model = U2TimeSeriesPredictionModel(model_path)
        model.new(
            target_column="value", known_covariates_names=[], prediction_length=horizon, freq=freq, eval_metric="WQL"
        )
        training_kwargs = {
            "enable_ensemble": False,
            "hyperparameters": {"RecursiveTabular": {"trainer_kwargs": {"accelerator": "cpu"}}},
            "random_seed": random_seed,
        }
        model.train(dataset, min_lookback=24, num_val_windows=1, num_test_windows=0, training_kwargs=training_kwargs)

        pred_kwargs = {"use_cache": False, "random_seed": random_seed}

        predictions = model.predict(dataset, **pred_kwargs)

        new_model = U2TimeSeriesPredictionModel(model_path)
        new_model.load()

        new_predictions = new_model.predict(dataset, **pred_kwargs)

        assert_frame_equal(predictions.to_data_frame(), new_predictions.to_data_frame(), atol=1e-8)

    @pytest.mark.parametrize(
        ("json_fixture", "freq", "expectation"),
        [
            ("json_long_periods", "D", does_not_raise()),
            (
                "json_single_timeseries",
                "D",
                pytest.raises(ValueError, match="At least some time series in train_data must have"),
            ),
            ("json_daily_freq", "h", does_not_raise()),
        ],
    )
    def test_mismatch_frequency(
        self,
        temp_dir: str,
        json_fixture: str,
        freq: str,
        expectation: AbstractContextManager,
        request: pytest.FixtureRequest,
    ):
        horizon = 5

        model_path = os.path.join(temp_dir, "test_mismatch_freq")
        json_path = request.getfixturevalue(json_fixture)
        data = load_training_input_from_json(json_path)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

        with expectation:
            model = U2TimeSeriesPredictionModel(model_path)
            model.new(
                target_column="value",
                known_covariates_names=[],
                prediction_length=horizon,
                freq=freq,
                eval_metric="WQL",
            )
            training_kwargs = {"enable_ensemble": False, "hyperparameters": {"SeasonalNaive": {}}, "random_seed": 54}
            model.train(
                dataset, min_lookback=12, num_val_windows=1, num_test_windows=0, training_kwargs=training_kwargs
            )

            model.predict(dataset)

            assert model.predictor.freq == freq
            assert model.metadata["frequency"] == freq

    def test_mismatch_covariates(self, temp_dir: str, json_single_timeseries: str):
        horizon = 5
        freq = "h"
        lags = [5, 10, 15]

        model_path = os.path.join(temp_dir, "test_mismatch_cov")
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

        dataset.add_temporal()
        dataset.add_lags(lags, lags)

        model = U2TimeSeriesPredictionModel(model_path)
        model.new(
            target_column="value", known_covariates_names=[], prediction_length=horizon, freq=freq, eval_metric="WQL"
        )
        training_kwargs = {"enable_ensemble": False, "hyperparameters": {"DirectTabular": {}}, "random_seed": 54}
        model.train(dataset, min_lookback=24, num_val_windows=1, num_test_windows=0, training_kwargs=training_kwargs)

        new_data = load_training_input_from_json(json_single_timeseries)
        new_dataset = U2TimeSeriesPredictionDataset.from_training_input(new_data)

        with pytest.raises(ValueError, match=r"^(\d+)\s+columns are missing from data:"):
            model.predict(new_dataset)
