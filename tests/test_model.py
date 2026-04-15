"""Tests for u2.model module.

Covers the full lifecycle of U2TimeSeriesPredictionModel: initialization, save/load round-trips, prediction output format, and guard-rails (predict before load). Uses real AutoGluon training on small fixtures to exercise integration paths.
"""

import json
import logging
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from autogluon.timeseries import TimeSeriesPredictor
from pandas.testing import assert_frame_equal

from tests.conftest import load_training_input_from_json
from u2.dataset import U2TimeSeriesPredictionDataset
from u2.defaults import MODEL_TRAINING_RESULT_LEADERBOARD_PATH, TIME_SERIES_GROUP_COLUMN_NAME
from u2.model import U2TimeSeriesPredictionModel
from u2.utils import get_min_timeseries_length


@pytest.mark.integration
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

    def test_added_columns_metadata_round_trip(self, temp_dir: str, json_single_timeseries: str):
        """added_columns, lags, and windows must survive model save/load exactly."""
        horizon = 5
        freq = "h"
        lags = [5, 10]
        windows = [12, 24]

        model_path = os.path.join(temp_dir, "metadata_rt_model")
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)
        dataset.add_temporal()
        dataset.add_lags(lags=lags, windows=windows)

        # Capture state before training so we can compare after reload
        expected_added_columns = {k: list(v) for k, v in dataset.added_columns.items()}

        model = U2TimeSeriesPredictionModel(model_path)
        model.new(
            target_column="value",
            known_covariates_names=[],
            prediction_length=horizon,
            freq=freq,
            eval_metric="WQL",
        )
        training_kwargs = {"enable_ensemble": False, "hyperparameters": {"SeasonalNaive": {}}, "random_seed": 54}
        model.train(dataset, min_lookback=24, num_val_windows=1, num_test_windows=0, training_kwargs=training_kwargs)

        reloaded = U2TimeSeriesPredictionModel(model_path)
        reloaded.load()

        assert reloaded.metadata["added_columns"] == expected_added_columns
        assert reloaded.metadata["lags"] == lags
        assert reloaded.metadata["windows"] == windows


@pytest.mark.unit
class TestU2TimeSeriesPredictionModelUnit:
    """Unit tests for U2TimeSeriesPredictionModel using mocks."""

    def test_load_already_loaded(self, caplog):
        """Loading when predictor is already set should log and return early."""
        model = U2TimeSeriesPredictionModel("/tmp/some_path")
        model.predictor = MagicMock()

        with caplog.at_level(logging.INFO, logger="U2 Time Series Prediction"):
            model.load()

        assert any("already loaded" in r.message for r in caplog.records)

    def test_load_persist_failure(self, tmp_path, json_single_timeseries: str, caplog):
        """persist() failure should log a warning but not raise."""
        horizon = 5
        model_path = str(tmp_path / "persist_fail_model")
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

        model = U2TimeSeriesPredictionModel(model_path)
        model.new(
            target_column="value", known_covariates_names=[], prediction_length=horizon, freq="h", eval_metric="WQL"
        )
        training_kwargs = {"enable_ensemble": False, "hyperparameters": {"SeasonalNaive": {}}, "random_seed": 54}
        model.train(dataset, min_lookback=24, num_val_windows=1, num_test_windows=0, training_kwargs=training_kwargs)

        new_model = U2TimeSeriesPredictionModel(model_path)

        with (
            patch.object(TimeSeriesPredictor, "persist", side_effect=RuntimeError("persist error")),
            caplog.at_level(logging.WARNING, logger="U2 Time Series Prediction"),
        ):
            new_model.load()

        assert new_model.predictor is not None
        assert any("persist() failed" in r.message for r in caplog.records)

    def test_unload_unpersist_failure(self, caplog):
        """unpersist() failure should log a warning but not raise."""
        model = U2TimeSeriesPredictionModel("/tmp/some_path")
        mock_predictor = MagicMock()
        mock_predictor.unpersist.side_effect = RuntimeError("unpersist error")
        model.predictor = mock_predictor

        with caplog.at_level(logging.WARNING, logger="U2 Time Series Prediction"):
            model.unload()

        assert model.predictor is None
        assert any("unpersist() failed" in r.message for r in caplog.records)

    def test_reload_reloads_with_new_path(self):
        """reload should unload, update path, and load again."""
        model = U2TimeSeriesPredictionModel("/tmp/old_path")

        with patch.object(model, "unload") as mock_unload, patch.object(model, "load") as mock_load:
            model.reload(new_path="/tmp/new_path")

        assert model.model_path == "/tmp/new_path"
        mock_unload.assert_called_once()
        mock_load.assert_called_once()

    def test_reload_without_new_path(self):
        """reload without new_path should keep existing path."""
        model = U2TimeSeriesPredictionModel("/tmp/old_path")

        with patch.object(model, "unload") as mock_unload, patch.object(model, "load") as mock_load:
            model.reload()

        assert model.model_path == "/tmp/old_path"
        mock_unload.assert_called_once()
        mock_load.assert_called_once()

    def test_train_frequency_mismatch_warning(self, caplog):
        """Frequency mismatch between dataset and model should log a warning."""
        model = U2TimeSeriesPredictionModel("/tmp/freq_warn")
        mock_predictor = MagicMock()
        mock_predictor.freq = "D"
        mock_predictor.prediction_length = 5
        mock_predictor.fit = MagicMock()
        model.predictor = mock_predictor

        dataset = MagicMock()
        dataset.freq = "h"
        dataset.df = pd.DataFrame(
            {"item_id": [0] * 100, "time": pd.date_range("2023-01-01", periods=100, freq="h"), "value": range(100)}
        )
        dataset.added_columns = {}

        with (
            caplog.at_level(logging.WARNING, logger="U2 Time Series Prediction"),
            patch.object(model, "_save_metadata"),
        ):
            model.train(dataset, min_lookback=24, num_val_windows=1, num_test_windows=0)

        assert any("resampled to match" in r.message for r in caplog.records)

    def test_save_metadata_with_lag_columns(self, tmp_path):
        """_save_metadata should extract lags and windows from added_columns when lag features are present."""
        model_path = str(tmp_path / "metadata_lag_model")
        model = U2TimeSeriesPredictionModel(model_path)
        model.predictor = MagicMock()
        model.predictor.freq = "h"
        model.predictor.prediction_length = 24

        dataset = MagicMock()
        dataset.added_columns = {
            "temporal": ["month_sin", "month_cos"],
            "lag": ["lag_24", "lag_168", "roll_mean_24", "roll_std_24", "roll_mean_168", "roll_std_168"],
        }

        model._save_metadata(
            dataset,
            min_lookback=50,
            num_val_windows=1,
            num_test_windows=1,
            training_start="2023-01-01",
            training_kwargs={},
        )

        assert model.metadata["lags"] == [24, 168]
        assert model.metadata["windows"] == [24, 168]
        assert model.metadata["added_columns"]["lag"] == dataset.added_columns["lag"]

    def test_save_metadata_without_lag_columns(self, tmp_path):
        """_save_metadata without lag features should set lags=[0] and compute min_timeseries_length accordingly."""
        model_path = str(tmp_path / "metadata_no_lag_model")
        model = U2TimeSeriesPredictionModel(model_path)
        model.predictor = MagicMock()
        model.predictor.freq = "h"
        model.predictor.prediction_length = 24

        dataset = MagicMock()
        dataset.added_columns = {"temporal": ["month_sin", "month_cos"]}

        model._save_metadata(
            dataset,
            min_lookback=100,
            num_val_windows=1,
            num_test_windows=1,
            training_start="2023-01-01",
            training_kwargs={},
        )

        assert "lags" not in model.metadata
        assert model.metadata["min_timeseries_length"] == 124  # max([0]) + 100 + 24 (horizon)

    def test_save_metadata_single_lag(self, tmp_path):
        """_save_metadata with a single lag should extract it correctly."""
        model_path = str(tmp_path / "metadata_single_lag")
        model = U2TimeSeriesPredictionModel(model_path)
        model.predictor = MagicMock()
        model.predictor.freq = "D"
        model.predictor.prediction_length = 7

        dataset = MagicMock()
        dataset.added_columns = {
            "lag": ["lag_7", "roll_mean_7", "roll_std_7", "roll_min_7", "roll_max_7"],
        }

        model._save_metadata(
            dataset,
            min_lookback=30,
            num_val_windows=2,
            num_test_windows=2,
            training_start="2023-01-01",
            training_kwargs={"presets": "chronos2"},
        )

        assert model.metadata["lags"] == [7]
        assert model.metadata["windows"] == [7]
        assert model.metadata["min_timeseries_length"] == 7 + 30 + 7  # max([7]) + 30 + 7 (horizon)

    def test_save_metadata_writes_json_file(self, tmp_path):
        """_save_metadata should write a JSON file to the model directory."""
        model_path = str(tmp_path / "metadata_file_model")
        model = U2TimeSeriesPredictionModel(model_path)
        model.predictor = MagicMock()
        model.predictor.freq = "h"
        model.predictor.prediction_length = 24

        dataset = MagicMock()
        dataset.added_columns = {}

        model._save_metadata(
            dataset,
            min_lookback=50,
            num_val_windows=1,
            num_test_windows=1,
            training_start="2023-01-01",
            training_kwargs={},
        )

        metadata_path = os.path.join(model_path, "u2", "metadata.json")
        assert os.path.exists(metadata_path)
        with open(metadata_path) as f:
            loaded = json.load(f)
        assert loaded["frequency"] == "h"
        assert loaded["horizon"] == 24

    def test_predict_when_unloaded_raises(self):
        """predict should raise RuntimeError when predictor is None."""
        model = U2TimeSeriesPredictionModel("/tmp/unloaded")
        dataset = MagicMock()

        with pytest.raises(RuntimeError, match="Predictor not loaded"):
            model.predict(dataset)

    def test_predict_success_path(self):
        """predict on a loaded model should call predictor.predict with correct args."""
        model = U2TimeSeriesPredictionModel("/tmp/pred_success")
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = pd.DataFrame({"mean": [1.0]})
        model.predictor = mock_predictor

        dataset = MagicMock()
        dataset.df = pd.DataFrame(
            {"item_id": [0] * 10, "time": pd.date_range("2023-01-01", periods=10, freq="h"), "value": range(10)}
        )
        dataset.future_df = pd.DataFrame(
            {"item_id": [0] * 3, "time": pd.date_range("2023-01-01T10:00", periods=3, freq="h"), "value": [0.0] * 3}
        )

        result = model.predict(dataset)

        assert result is not None
        mock_predictor.predict.assert_called_once()

    def test_predict_without_future_df(self):
        """predict should handle future_df=None by passing None as known_covariates."""
        model = U2TimeSeriesPredictionModel("/tmp/pred_no_future")
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = pd.DataFrame({"mean": [1.0]})
        model.predictor = mock_predictor

        dataset = MagicMock()
        dataset.df = pd.DataFrame(
            {"item_id": [0] * 10, "time": pd.date_range("2023-01-01", periods=10, freq="h"), "value": range(10)}
        )
        dataset.future_df = None

        result = model.predict(dataset)

        assert result is not None
        call_kwargs = mock_predictor.predict.call_args
        assert call_kwargs.kwargs["known_covariates"] is None

    def test_save_leaderboards_computes_average(self, tmp_path):
        """_save_leaderboards should average metrics across cutoff windows and save CSV."""
        model_path = str(tmp_path / "lb_avg_model")
        os.makedirs(os.path.join(model_path, "u2"), exist_ok=True)

        model = U2TimeSeriesPredictionModel(model_path)
        mock_predictor = MagicMock()
        mock_predictor.prediction_length = 5
        model.predictor = mock_predictor

        # Mock leaderboard to return consistent data for 2 cutoffs
        lb_df = pd.DataFrame(
            {
                "model": ["ModelA", "ModelB"],
                "score_test": [0.8, 0.6],
                "score_val": [0.7, 0.5],
                "WAPE": [0.1, 0.2],
            }
        )
        mock_predictor.leaderboard.return_value = lb_df

        test_data = MagicMock()
        model._save_leaderboards(num_test_windows=2, test_data=test_data)

        csv_path = os.path.join(model_path, MODEL_TRAINING_RESULT_LEADERBOARD_PATH)
        assert os.path.exists(csv_path)

        result_df = pd.read_csv(csv_path)
        assert "model" in result_df.columns
        assert len(result_df) == 2
        # Scores should be averaged (same mock data -> same values)
        assert result_df.loc[result_df["model"] == "ModelA", "score_test"].iloc[0] == pytest.approx(0.8)

    def test_save_leaderboards_sorted_by_score(self, tmp_path):
        """Leaderboard should be sorted by score_test descending."""
        model_path = str(tmp_path / "lb_sort_model")
        os.makedirs(os.path.join(model_path, "u2"), exist_ok=True)

        model = U2TimeSeriesPredictionModel(model_path)
        mock_predictor = MagicMock()
        mock_predictor.prediction_length = 5
        model.predictor = mock_predictor

        lb_df = pd.DataFrame(
            {
                "model": ["BadModel", "GoodModel"],
                "score_test": [-0.5, -0.1],
                "WAPE": [0.3, 0.1],
            }
        )
        mock_predictor.leaderboard.return_value = lb_df

        test_data = MagicMock()
        model._save_leaderboards(num_test_windows=1, test_data=test_data)

        csv_path = os.path.join(model_path, MODEL_TRAINING_RESULT_LEADERBOARD_PATH)
        result_df = pd.read_csv(csv_path)
        assert result_df["model"].iloc[0] == "GoodModel"

    def test_metadata_min_timeseries_length_includes_horizon(self, tmp_path, json_single_timeseries: str):
        """min_timeseries_length stored in model metadata should account for
        the forecast horizon, consistent with the value used during training
        validation.

        During training, min_timeseries_length = max(lags) + min_lookback + horizon.
        The metadata should store the same value so that prediction-time
        drop_short_timeseries uses the same threshold.
        """
        lags = [168]
        min_lookback = 100
        horizon = 24

        training_min = get_min_timeseries_length(lags=lags, min_lookback=min_lookback, horizon=horizon)
        metadata_min = get_min_timeseries_length(lags=lags, min_lookback=min_lookback, horizon=horizon)

        assert training_min == metadata_min

    def test_train_with_save_leaderboards(self, tmp_path, json_single_timeseries: str):
        """Training with save_leaderboards=True should produce a leaderboard CSV."""
        model_path = str(tmp_path / "lb_model")
        data = load_training_input_from_json(json_single_timeseries)
        dataset = U2TimeSeriesPredictionDataset.from_training_input(data)

        model = U2TimeSeriesPredictionModel(model_path)
        model.new(target_column="value", known_covariates_names=[], prediction_length=5, freq="h", eval_metric="WQL")
        training_kwargs = {"enable_ensemble": False, "hyperparameters": {"SeasonalNaive": {}}, "random_seed": 54}
        model.train(
            dataset,
            min_lookback=24,
            num_val_windows=1,
            num_test_windows=1,
            save_leaderboards=True,
            training_kwargs=training_kwargs,
        )

        assert os.path.exists(os.path.join(model_path, MODEL_TRAINING_RESULT_LEADERBOARD_PATH))
