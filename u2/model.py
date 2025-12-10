import threading

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from .dataset import U2TimeSeriesPredictionDataset
from .logging_config import logger


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
        self.predictor: TimeSeriesPredictor = None
        self.lock = threading.RLock()

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
            try:
                self.predictor.persist()
            except Exception as e:
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
            except Exception as e:
                logger.warning("unpersist() failed: %s", e)
            self.predictor = None
            logger.info("Predictor unloaded")

    def reload(self, new_path: str = None) -> None:
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
            if self.predictor is None:
                raise RuntimeError("Predictor not loaded")

            return self.predictor.predict(time_series.df, time_series.future_df, **kwargs)
