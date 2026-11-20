"""U2Demo time-series forecasting library."""

from u2demo_time_series_forecasting._version import __version__
from u2demo_time_series_forecasting.data_models import Forecast, ForecastInput, ForecastTrainingInput
from u2demo_time_series_forecasting.data_models_v2 import (
    ForecastInputV2,
    ForecastTrainingInputV2,
    ForecastTypeV2,
    ForecastV2,
    ObservedProperty,
    forecast_to_v2,
)
from u2demo_time_series_forecasting.features.names import FeatureName
from u2demo_time_series_forecasting.metadata import ModelMetadata, TrainingMetadata
from u2demo_time_series_forecasting.model import U2TimeSeriesForecastingModel
from u2demo_time_series_forecasting.predict import predict, predict_from_payload, predict_v2
from u2demo_time_series_forecasting.train import train, train_from_payload, train_v2
from u2demo_time_series_forecasting.versioning import PayloadVersion, detect_payload_version

__all__ = [
    "FeatureName",
    "Forecast",
    "ForecastInput",
    "ForecastInputV2",
    "ForecastTrainingInput",
    "ForecastTrainingInputV2",
    "ForecastTypeV2",
    "ForecastV2",
    "ModelMetadata",
    "ObservedProperty",
    "PayloadVersion",
    "TrainingMetadata",
    "U2TimeSeriesForecastingModel",
    "__version__",
    "detect_payload_version",
    "forecast_to_v2",
    "predict",
    "predict_from_payload",
    "predict_v2",
    "train",
    "train_from_payload",
    "train_v2",
]
