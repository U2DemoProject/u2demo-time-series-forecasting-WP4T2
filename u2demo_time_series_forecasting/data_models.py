from datetime import datetime
from typing import Annotated, Literal

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, StringConstraints

from u2demo_time_series_forecasting.features.names import FeatureName

type QuantileKey = Annotated[str, StringConstraints(pattern=r"^(0(\.\d+)?|1(\.0+)?)$")]
type QuantileLevel = Annotated[float, Field(ge=0, le=1)]


class TimeValue(BaseModel):
    """Generic timestamp-value pair aligned with SAREF (saref:hasTimestamp, saref:hasValue)."""

    time: datetime = Field(..., description="Timestamp of the measurement")
    value: float = Field(..., description="Measured or scheduled numeric value")


class ForecastedTimeValue(TimeValue):
    """Extends TimeValue with probabilistic forecast information."""

    quantiles: dict[QuantileKey, float] | None = Field(
        default=None,
        description="Quantile forecasts at this timestep. Keys are quantile levels (0-1), values are the forecasted value at that quantile.",
    )


type ForecastType = Literal["deterministic", "stochastic"]


class TimeParameters(BaseModel):
    """Time-related configuration parameters for forecasting."""

    timestep_minutes: int | None = Field(default=None, ge=1, description="Length of a single time step in minutes.")
    horizon_hours: int = Field(default=24, ge=1, description="Planning horizon length in hours.")
    scheduling_time: datetime | None = Field(default=None, description="Time at which the forecasting is executed.")


class TrainingSplit(BaseModel):
    """Validation/test windows configuration."""

    num_val_windows: int = Field(default=1, ge=1, description="Number of windows of data used for validation.")
    num_test_windows: int = Field(default=1, ge=1, description="Number of windows of data used for testing.")


class ModelConfig(BaseModel):
    """Configuration parameters for the training process."""

    model_config = ConfigDict(extra="allow")

    model_id: str = Field(..., description="Unique identifier for this model instance.")
    model_preset: str | None = Field(default="chronos2", description="AutoGluon preset (e.g. 'chronos2', 'fast_training', 'high_quality').")
    model_version: str | None = Field(default=None, description="Version identifier for the model (e.g., '1.0.0').")
    training_split: TrainingSplit = Field(
        default_factory=TrainingSplit, description="Train/validation/test split configuration."
    )
    lookback_steps: int | None = Field(
        default=None, ge=1, description="Number of past timesteps used as input features."
    )
    eval_metric: str | None = Field(
        default=None, description="Evaluation metric to optimize (e.g., 'RMSE', 'MAE', 'WQL')."
    )
    quantile_levels: list[QuantileLevel] | None = Field(
        default=None, description="Quantile levels to predict (for quantile regression models)."
    )
    excluded_model_types: list[str] | None = Field(
        default=None, description="Model types to exclude from training (e.g., ['DeepAR', 'PatchTST'])."
    )


class BaseSeries(BaseModel):
    """Common identified time series metadata."""

    model_config = ConfigDict(extra="allow")

    series_id: str = Field(..., description="Unique identifier for this series.")
    variable_name: str = Field(..., description="Name of the variable to forecast.")
    asset_id: str | None = Field(default=None, description="Identifier of the asset this series belongs to.")
    asset_type: str | None = Field(default=None, description="Type of the asset (e.g., 'PV', 'WindTurbine', 'Load').")
    unit: str | None = Field(default=None, description="Unit of measurement.")
    unitIri: AnyUrl | None = Field(default=None, description="QUDT URI for the unit.")  # noqa: N815


class TrainingTargetSeries(BaseSeries):
    """Target series shape used by ForecastTrainingInputDM."""

    training_values: list[TimeValue] = Field(
        ..., min_length=2, description="Full historical ground truth time series used for training."
    )


class ForecastTargetSeries(BaseSeries):
    """Target series shape used by ForecastInputDM."""

    historical_values: list[TimeValue] = Field(
        ..., min_length=1, description="Past observations of the target variable used as input for forecast."
    )


class BaseExogenousSeries(BaseModel):
    """Common exogenous time series metadata."""

    model_config = ConfigDict(extra="allow")

    series_id: str = Field(..., description="Unique identifier for this exogenous series.")
    variable_name: str = Field(..., description="Name of the exogenous variable.")
    source: str | None = Field(default=None, description="Data source or provider.")
    unit: str | None = Field(default=None, description="Unit of measurement.")
    unitIri: AnyUrl | None = Field(default=None, description="QUDT URI for the unit.")  # noqa: N815


class TrainingExogenousSeries(BaseExogenousSeries):
    """Exogenous series shape used by ForecastTrainingInputDM."""

    is_known_covariate: bool = Field(
        default=False,
        description="Indicates if this exogenous series is a known covariate (i.e., fully known for the forecast horizon).",
    )
    training_values: list[TimeValue] | None = Field(
        ..., min_length=2, description="Full historical ground truth time series used for training."
    )


class ForecastExogenousSeries(BaseExogenousSeries):
    """Exogenous series shape used by ForecastInputDM."""

    historical_values: list[TimeValue] = Field(
        ..., min_length=1, description="Historical values aligned with target_series for forecast."
    )
    future_values: list[TimeValue] | None = Field(
        default=None, description="Known or forecasted future values over the forecast horizon."
    )


class Location(BaseModel):
    """Geographic location aligned with WGS84."""

    latitude: float = Field(ge=-90, le=90, description="Latitude in decimal degrees (WGS84).")
    longitude: float = Field(ge=-180, le=180, description="Longitude in decimal degrees (WGS84).")


class ForecastTrainingInput(BaseModel):
    """Input data model for training a forecasting model."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    forecast_type: ForecastType = Field(..., description="Type of forecast model to train.")
    time_parameters: TimeParameters = Field(
        ..., description="Target forecast resolution and horizon the model should learn."
    )
    training_model_config: ModelConfig = Field(
        ...,
        alias="model_config",
        serialization_alias="model_config",
        description="Configuration parameters for the training process.",
    )
    target_series: TrainingTargetSeries = Field(
        ..., description="The target variable with ground truth data for training."
    )
    location: Location | None = Field(default=None, description="Geographic location of the asset.")
    exogenous_series: list[TrainingExogenousSeries] | None = Field(
        default=None, description="Optional additional time series of external variables."
    )
    features: list[FeatureName] = Field(
        default_factory=list,
        description="Feature-engineering steps to apply during training.",
    )
    num_weekly_lags: int = Field(
        default=0,
        ge=0,
        description="Number of weekly lag/rolling-stat features to generate when the lag feature is enabled.",
    )


class ForecastInput(BaseModel):
    """Input data model for a forecasting request."""

    model_config = ConfigDict(extra="allow")

    forecast_type: ForecastType = Field(..., description="Desired forecast output type.")
    time_parameters: TimeParameters = Field(..., description="Forecast horizon and time resolution parameters.")
    model_id: str | None = Field(default=None, description="Unique identifier for the model to use for forecast.")
    model_version: str | None = Field(default=None, description="Version identifier for the model (e.g., '1.0.0').")
    target_series: ForecastTargetSeries = Field(..., description="The primary variable to be forecasted.")
    location: Location | None = Field(
        default=None, description="Geographic location of the asset. Useful for weather-dependent forecasts."
    )
    exogenous_series: list[ForecastExogenousSeries] | None = Field(
        default=None, description="Optional additional time series of external variables."
    )


class ForecastSeries(BaseModel):
    """A forecast time series for a specific variable/asset combination."""

    model_config = ConfigDict(extra="allow")

    series_id: str = Field(..., description="Unique identifier for this forecast series.")
    variable_name: str = Field(..., description="Name of the forecasted variable.")
    asset_id: str | None = Field(default=None, description="Identifier of the asset this forecast relates to.")
    asset_type: str | None = Field(default=None, description="Type of the asset (e.g., 'PV', 'WindTurbine', 'Load').")
    unit: str | None = Field(default=None, description="Unit of measurement.")
    unitIri: AnyUrl | None = Field(default=None, description="QUDT URI for the unit.")  # noqa: N815
    values: list[ForecastedTimeValue] = Field(..., min_length=1, description="Time series of forecasted values.")


class Forecast(BaseModel):
    """Forecast data model with identified time series and optional probabilistic information."""

    model_config = ConfigDict(extra="allow")

    forecast_type: ForecastType = Field(..., description="Type of forecast.")
    forecast_date: datetime = Field(..., description="Generation time of the forecast.")
    time_parameters: TimeParameters | None = Field(
        default=None, description="Time resolution and horizon parameters for this forecast."
    )
    forecast_series: ForecastSeries = Field(..., description="Identified forecast time series.")
