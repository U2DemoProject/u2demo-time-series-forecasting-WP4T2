"""U2Demo time-series forecasting library."""

from u2demo_time_series_forecasting._version import __version__
from u2demo_time_series_forecasting.data_models import Forecast, ForecastInput, ForecastTrainingInput
from u2demo_time_series_forecasting.features.names import FeatureName
from u2demo_time_series_forecasting.metadata import ModelMetadata, TrainingMetadata
from u2demo_time_series_forecasting.model import U2TimeSeriesForecastingModel
from u2demo_time_series_forecasting.predict import predict
from u2demo_time_series_forecasting.train import train

__all__ = [
    "FeatureName",
    "Forecast",
    "ForecastInput",
    "ForecastTrainingInput",
    "ModelMetadata",
    "TrainingMetadata",
    "U2TimeSeriesForecastingModel",
    "__version__",
    "predict",
    "train",
]
