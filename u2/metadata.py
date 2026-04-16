"""Pydantic models persisted alongside trained predictors as ``metadata.json``."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from u2.features.names import FeatureName


class TrainingMetadata(BaseModel):
    """Captures when and how a model was trained."""

    model_config = ConfigDict(extra="ignore")

    start_utc: str
    end_utc: str
    min_lookback: int
    num_val_windows: int
    num_test_windows: int
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ModelMetadata(BaseModel):
    """Persisted companion to a trained AutoGluon predictor."""

    model_config = ConfigDict(extra="ignore")

    version: str
    frequency: str
    horizon: int
    min_timeseries_length: int
    added_columns: dict[FeatureName, list[str]] = Field(default_factory=dict)
    lags: list[int] = Field(default_factory=list)
    windows: list[int] = Field(default_factory=list)
    training: TrainingMetadata
