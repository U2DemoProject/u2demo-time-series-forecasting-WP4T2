"""U2 time-series prediction library."""

from u2._version import __version__
from u2.data_models import Forecast, ForecastInput, ForecastTrainingInput
from u2.features.names import FeatureName
from u2.metadata import ModelMetadata, TrainingMetadata
from u2.model import U2TimeSeriesPredictionModel
from u2.predict import predict
from u2.train import train

__all__ = [
    "FeatureName",
    "Forecast",
    "ForecastInput",
    "ForecastTrainingInput",
    "ModelMetadata",
    "TrainingMetadata",
    "U2TimeSeriesPredictionModel",
    "__version__",
    "predict",
    "train",
]
