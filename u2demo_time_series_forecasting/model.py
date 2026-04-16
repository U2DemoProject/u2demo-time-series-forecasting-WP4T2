import os
import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from u2demo_time_series_forecasting._version import __version__
from u2demo_time_series_forecasting.constants.columns import TIME_SERIES_TIMESTAMP_COLUMN_NAME
from u2demo_time_series_forecasting.constants.paths import LEADERBOARD_FILENAME, METADATA_FILENAME
from u2demo_time_series_forecasting.core.logger import logger
from u2demo_time_series_forecasting.dataset import U2TimeSeriesPredictionDataset
from u2demo_time_series_forecasting.features.names import FeatureName
from u2demo_time_series_forecasting.metadata import ModelMetadata, TrainingMetadata
from u2demo_time_series_forecasting.utils import get_min_timeseries_length


class U2TimeSeriesPredictionModel:
    """
    Thread-safe wrapper around an AutoGluon TimeSeriesPredictor for loading,
    unloading, reloading, and running predictions.

    This class manages a predictor stored on disk, enforcing serialized access
    using an RLock. It optionally checks version compatibility on load and
    provides convenience methods for lifecycle management.

    Parameters
    ----------
    model_path : str
        Filesystem path to the saved AutoGluon TimeSeriesPredictor.
    require_version_match : bool, optional
        Whether AutoGluon should enforce version compatibility when loading the
        predictor. Defaults to True.

    Attributes
    ----------
    model_path : str
        Path to the predictor on disk.
    require_version_match : bool
        Whether version compatibility is required when loading.
    predictor : TimeSeriesPredictor | None
        Loaded predictor instance or None if not yet loaded.
    lock : threading.RLock
        Lock used to ensure thread-safe operations.

    """

    def __init__(self, model_path: str, require_version_match: bool = True):
        self.model_path = model_path
        self.require_version_match = require_version_match
        self.predictor: TimeSeriesPredictor | None = None
        self.lock = threading.RLock()
        self.metadata: ModelMetadata | None = None

    def _require_metadata(self) -> ModelMetadata:
        if self.metadata is None:
            raise RuntimeError("Model metadata is not loaded")
        return self.metadata

    def _require_predictor(self) -> TimeSeriesPredictor:
        if self.predictor is None:
            raise RuntimeError("Predictor is not loaded")
        return self.predictor

    @property
    def freq(self) -> str:
        """Frequency string (e.g., ``"h"``, ``"D"``) recorded at training time."""
        return self._require_metadata().frequency

    @property
    def prediction_length(self) -> int:
        """Forecast horizon in steps, recorded at training time."""
        return self._require_metadata().horizon

    @property
    def quantile_levels(self) -> list[float]:
        """Quantile levels configured on the loaded predictor."""
        return self._require_predictor().quantile_levels

    def _save_leaderboards(self, num_test_windows: int, test_data: TimeSeriesDataFrame) -> None:
        """
        Compute and save average leaderboard metrics over multiple cutoff windows.

        Parameters
        ----------
        num_test_windows : int
            Number of rolling cutoffs to evaluate.
        test_data : TimeSeriesDataFrame
            Held-out dataset used for leaderboard evaluation.

        Returns
        -------
        None
            Writes a CSV file with averaged metrics to the model directory.

        """
        predictor = self._require_predictor()
        horizon = predictor.prediction_length
        cutoffs = range(-num_test_windows * horizon, -horizon + 1, horizon)
        leaderboards: list[pd.DataFrame] = []
        for cutoff in cutoffs:
            lb = predictor.leaderboard(test_data, cutoff=cutoff, extra_metrics=["WAPE"], silent=True)
            lb["cutoff"] = cutoff
            leaderboards.append(lb)

        # Concatenate all leaderboards
        all_lb = pd.concat(leaderboards, ignore_index=True)

        # Group by model and average metrics (excluding non-numeric columns)
        metrics = all_lb.select_dtypes(include="number").columns.difference(["cutoff"])
        avg_lb = all_lb.groupby("model")[metrics].mean().reset_index().sort_values("score_test", ascending=False)

        # Reorder columns to match the original leaderboard order
        original_cols = leaderboards[0].columns
        avg_lb = avg_lb.reindex(columns=[col for col in original_cols if col in avg_lb.columns])
        avg_lb.to_csv(f"{self.model_path}/{LEADERBOARD_FILENAME}", index=False)

    def _save_metadata(
        self,
        time_series: U2TimeSeriesPredictionDataset,
        min_lookback: int,
        num_val_windows: int,
        num_test_windows: int,
        training_start: str,
        training_kwargs: dict,
    ) -> None:
        """
        Persist training metadata (config and derived info) alongside the model.

        Parameters
        ----------
        time_series : U2TimeSeriesPredictionDataset
            Dataset used for training, providing frequency and added columns.
        min_lookback : int
            Minimum lookback used to form the backtest split.
        num_val_windows : int
            Number of backtest cutoffs used for validation.
        num_test_windows : int
            Number of backtest cutoffs used for evaluation.
        training_start : str
            UTC timestamp string when training began.
        training_kwargs : dict
            Additional keyword arguments passed to `predictor.fit()`.

        Returns
        -------
        None
            Writes a JSON metadata file under the model directory and updates
            `self.metadata`.

        """
        predictor = self._require_predictor()
        if predictor.freq is None:
            raise RuntimeError("Predictor has no frequency set")
        os.makedirs(f"{self.model_path}/u2", exist_ok=True)

        lag_columns = time_series.added_columns.get(FeatureName.LAG, [])
        lags = [int(col.split("_")[1]) for col in lag_columns if col.startswith("lag_")]
        # There are other "roll_" columns, but they all have the same ending numbers as "roll_mean_"
        windows = [int(col.split("_")[2]) for col in lag_columns if col.startswith("roll_mean_")]

        # Include horizon so metadata threshold matches the one used during training validation
        min_timeseries_length = get_min_timeseries_length(
            lags=lags or [0], min_lookback=min_lookback, horizon=predictor.prediction_length
        )

        metadata = ModelMetadata(
            version=__version__,
            frequency=predictor.freq,
            horizon=predictor.prediction_length,
            min_timeseries_length=min_timeseries_length,
            added_columns={FeatureName(k): v for k, v in time_series.added_columns.items()},
            lags=lags,
            windows=windows,
            training=TrainingMetadata(
                start_utc=training_start,
                end_utc=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                min_lookback=min_lookback,
                num_val_windows=num_val_windows,
                num_test_windows=num_test_windows,
                kwargs=training_kwargs,
            ),
        )

        Path(f"{self.model_path}/{METADATA_FILENAME}").write_text(metadata.model_dump_json(indent=4))
        self.metadata = metadata

    def new(
        self,
        target_column: str,
        known_covariates_names: list[str],
        prediction_length: int,
        freq: str,
        eval_metric: str,
        quantile_levels: list[float] | None = None,
    ) -> None:
        """
        Create a new predictor instance at `model_path` with the given settings.

        Parameters
        ----------
        target_column : str
            Name of the target column in the training data.
        known_covariates_names : list[str]
            Feature column names considered known at prediction time.
        prediction_length : int
            Forecast horizon in steps.
        freq : str
            Frequency string (e.g., "h", "D").
        eval_metric : str
            AutoGluon evaluation metric to optimize during training.
        quantile_levels : list[float], optional
            List of quantile levels for quantile regression models (e.g., [0.1, 0.5, 0.9]). Defaults to None.

        Returns
        -------
        None
            Initializes `self.predictor` and unloads any previous predictor.

        """
        self.unload()
        self.predictor = TimeSeriesPredictor(
            path=self.model_path,
            target=target_column,
            known_covariates_names=known_covariates_names,
            prediction_length=prediction_length,
            freq=freq,
            eval_metric=eval_metric,
            quantile_levels=quantile_levels,
        )

    def load(self) -> None:
        """
        Load the AutoGluon TimeSeriesPredictor from disk if not already loaded.

        This method:
        - Acquires a lock to ensure thread safety.
        - Loads the predictor from `self.model_path`.
        - Attempts to persist the predictor to optimize repeated use.
        - Logs progress and warnings.

        Returns
        -------
        None

        """
        with self.lock:
            if self.predictor is not None:
                logger.info("Predictor already loaded")
                return
            logger.info(f"Loading predictor from {self.model_path}")
            self.predictor = TimeSeriesPredictor.load(self.model_path, require_version_match=self.require_version_match)
            self.metadata = ModelMetadata.model_validate_json(
                Path(f"{self.model_path}/{METADATA_FILENAME}").read_text()
            )
            try:
                self.predictor.persist()
            except (AttributeError, NotImplementedError, RuntimeError) as e:
                logger.warning("persist() failed or not supported: %s", e)
            logger.info("Predictor loaded and persisted")

    def unload(self) -> None:
        """
        Unload the currently loaded predictor and release associated resources.

        This method:
        - Acquires a lock for thread safety.
        - Calls `unpersist()` on the predictor when available.
        - Sets `self.predictor` to None.
        - Logs progress and errors.

        Returns
        -------
        None

        """
        with self.lock:
            if self.predictor is None:
                return
            try:
                self.predictor.unpersist()
            except (AttributeError, NotImplementedError, RuntimeError) as e:
                logger.warning("unpersist() failed: %s", e)
            self.predictor = None
            logger.info("Predictor unloaded")

    def reload(self, new_path: str | None = None) -> None:
        """
        Reload the predictor from disk, optionally using a new model path.

        This method:
        - Acquires a lock to ensure serialized access.
        - Updates the model path if provided.
        - Unloads the currently loaded predictor.
        - Loads the predictor again from the updated path.

        Parameters
        ----------
        new_path : str, optional
            New path to load the predictor from. If None, the existing path is reused.

        Returns
        -------
        None

        """
        with self.lock:
            if new_path:
                self.model_path = new_path
            self.unload()
            self.load()

    def train(
        self,
        time_series: U2TimeSeriesPredictionDataset,
        min_lookback: int,
        num_val_windows: int,
        num_test_windows: int,
        save_leaderboards: bool = False,
        training_kwargs: dict | None = None,
    ) -> None:
        """
        Fit the predictor using the provided dataset and training configuration.

        The dataset should already include engineered features. Training performs a
        backtest split using `min_lookback + horizon * num_test_windows` as the
        prediction_length for the split, then calls `predictor.fit()` on the
        training portion. Optionally computes and saves averaged leaderboards.

        Parameters
        ----------
        time_series : U2TimeSeriesPredictionDataset
            Dataset containing the engineered training DataFrame.
        min_lookback : int
            Minimum lookback used to form the backtest split.
        num_val_windows : int
            Number of backtesting windows used for validation.
        num_test_windows : int
            Number of backtesting windows used for evaluation.
        save_leaderboards : bool, optional
            Whether to compute and save averaged leaderboards. Defaults to False.
        training_kwargs : dict, optional
            Additional keyword arguments forwarded to `predictor.fit()`.

        Returns
        -------
        None

        """
        predictor = self._require_predictor()
        if training_kwargs is None:
            training_kwargs = {"presets": "chronos2"}

        if time_series.freq != predictor.freq:
            logger.warning(
                f"Input data has frequency of {time_series.freq} while the model is set with a frequency of {predictor.freq}.\n"
                "The data will be resampled to match, make sure it is intended"
            )
        df = TimeSeriesDataFrame(time_series.df.copy(), timestamp_column=TIME_SERIES_TIMESTAMP_COLUMN_NAME)

        train_data, test_data = df.train_test_split(
            prediction_length=min_lookback + predictor.prediction_length * num_test_windows
        )
        training_start = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        predictor.fit(
            train_data=train_data, num_val_windows=num_val_windows, refit_every_n_windows=None, **training_kwargs
        )

        self._save_metadata(
            time_series, min_lookback, num_val_windows, num_test_windows, training_start, training_kwargs
        )
        if save_leaderboards:
            self._save_leaderboards(num_test_windows, test_data)

    def predict(self, time_series: U2TimeSeriesPredictionDataset, **kwargs) -> TimeSeriesDataFrame:
        """
        Generate forecasts using the loaded TimeSeriesPredictor.

        This method:
        - Ensures the predictor is loaded.
        - Uses the dataset's `.df` (historical data) and `.future_df`
          (forecast targets or covariates) as AutoGluon inputs.
        - Passes any additional keyword arguments directly to
          `TimeSeriesPredictor.predict()`.

        Parameters
        ----------
        time_series : U2TimeSeriesPredictionDataset
            Dataset wrapper containing historical data (`df`) and
            optional future data (`future_df`) in DataFrame format.
        **kwargs
            Additional keyword arguments forwarded to the AutoGluon predictor.

        Returns
        -------
        pandas.DataFrame
            Forecast results produced by the underlying predictor.

        Raises
        ------
        RuntimeError
            If the predictor has not been loaded.

        """
        with self.lock:
            predictor = self._require_predictor()
            df = TimeSeriesDataFrame(time_series.df, timestamp_column=TIME_SERIES_TIMESTAMP_COLUMN_NAME)
            future_df = (
                TimeSeriesDataFrame(time_series.future_df, timestamp_column=TIME_SERIES_TIMESTAMP_COLUMN_NAME)
                if time_series.future_df is not None
                else None
            )

            return predictor.predict(data=df, known_covariates=future_df, **kwargs)
