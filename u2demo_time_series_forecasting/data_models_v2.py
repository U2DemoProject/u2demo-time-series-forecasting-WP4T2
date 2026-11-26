"""Pydantic models for the training and forecast payloads in the U2Demo data models v2 vocabulary.

``ForecastV2`` is the forecast output: the ``ForecastPayloadDM`` projection of the v2 data models
(``v2.0.0/api/components/forecast`` in https://github.com/U2DemoProject/U2DemoDataModels). The v2 data models do not
define the training and forecast requests; ``ForecastTrainingInputV2`` and ``ForecastInputV2`` are this library's own
contract and use the v2 names, so that a request and the forecast it produces describe a series the same way.

v2 requests run through the v1 pipeline: ``to_v1()`` converts a request to its v1 equivalent and ``forecast_to_v2``
converts the resulting v1 ``Forecast`` to the v2 payload.
"""

import math
import re
from datetime import UTC, datetime, tzinfo
from enum import StrEnum
from typing import Annotated, Self
from uuid import uuid4

from pydantic import (
    AfterValidator,
    AnyUrl,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    model_validator,
)

from u2demo_time_series_forecasting.data_models import (
    Forecast,
    ForecastExogenousSeries,
    ForecastInput,
    ForecastTargetSeries,
    ForecastTrainingInput,
    ForecastType,
    Location,
    ModelConfig,
    QuantileKey,
    QuantileLevel,
    TimeParameters,
    TimeValue,
    TrainingExogenousSeries,
    TrainingSplit,
    TrainingTargetSeries,
)
from u2demo_time_series_forecasting.features.names import FeatureName

QUDT_UNIT_NAMESPACE = "http://qudt.org/vocab/unit/"

# Unit symbols with their QUDT unit IRI (https://qudt.org/vocab/unit/), used to derive the missing one of
# ``unit_symbol`` / ``unit`` on a request series.
UNIT_SYMBOL_TO_QUDT: dict[str, str] = {
    "W": "unit:W",
    "kW": "unit:KiloW",
    "MW": "unit:MegaW",
    "Wh": "unit:W-HR",
    "kWh": "unit:KiloW-HR",
    "MWh": "unit:MegaW-HR",
    "°C": "unit:DEG_C",
    "%": "unit:PERCENT",
    "EUR/MWh": "unit:CCY_EUR-PER-MegaW-HR",
}
QUDT_TO_UNIT_SYMBOL: dict[str, str] = {iri: symbol for symbol, iri in UNIT_SYMBOL_TO_QUDT.items()}
_KNOWN_UNIT_SYMBOLS = ", ".join(UNIT_SYMBOL_TO_QUDT)

# Common ``loss_function`` values and the AutoGluon ``eval_metric`` they stand for. Other values are passed
# to AutoGluon upper-cased.
_LOSS_FUNCTION_TO_EVAL_METRIC: dict[str, str] = {
    "mse": "MSE",
    "mae": "MAE",
    "pinball": "WQL",
    "rmse": "RMSE",
    "mape": "MAPE",
    "mase": "MASE",
}

# Preferences passed on to ModelConfig when the request sets them, so that a preference left out takes the
# ModelConfig default exactly as in a v1 request.
_PREFERENCES_FOR_MODEL_CONFIG = {
    "model_preset",
    "model_version",
    "lookback_steps",
    "quantile_levels",
    "excluded_model_types",
}


def _check_qudt_unit(unit: str) -> str:
    if re.fullmatch(r"unit:\S+", unit) is None:
        raise ValueError(
            f"'{unit}' is not a QUDT unit IRI such as 'unit:KiloW'. Unit symbols such as 'kW' go in unit_symbol."
        )
    return unit


type QudtUnit = Annotated[str, AfterValidator(_check_qudt_unit)]


class ObservedProperty(StrEnum):
    """Observable property vocabulary of the v2 data models (``ForecastSeriesDM.observedProperty``)."""

    ACTIVE_POWER = "ActivePower"
    REACTIVE_POWER = "ReactivePower"
    APPARENT_POWER = "ApparentPower"
    CHARGE_POWER = "ChargePower"
    DISCHARGE_POWER = "DischargePower"
    DC_POWER = "DCPower"
    MPP_POWER = "MPPPower"
    POWER_FACTOR = "PowerFactor"
    CHARGE_ENERGY = "ChargeEnergy"
    DISCHARGE_ENERGY = "DischargeEnergy"
    ENERGY_YIELD = "EnergyYield"
    MPP_ENERGY = "MPPEnergy"
    IMPORT_ENERGY = "ImportEnergy"
    EXPORT_ENERGY = "ExportEnergy"
    VOLTAGE = "Voltage"
    DC_VOLTAGE = "DCVoltage"
    CURRENT = "Current"
    DC_CURRENT = "DCCurrent"
    FREQUENCY = "Frequency"
    BESS_TEMPERATURE = "BESSTemperature"
    SUPPLY_TEMPERATURE = "SupplyTemperature"
    RETURN_TEMPERATURE = "ReturnTemperature"
    DHW_TEMPERATURE = "DHWTemperature"
    INDOOR_TEMPERATURE = "IndoorTemperature"
    OUTDOOR_TEMPERATURE = "OutdoorTemperature"
    STATE_OF_CHARGE = "StateOfCharge"
    STATE_OF_HEALTH = "StateOfHealth"
    PHASE_A_ACTIVE_POWER = "PhaseAActivePower"
    PHASE_B_ACTIVE_POWER = "PhaseBActivePower"
    PHASE_C_ACTIVE_POWER = "PhaseCActivePower"
    PHASE_A_VOLTAGE = "PhaseAVoltage"
    PHASE_B_VOLTAGE = "PhaseBVoltage"
    PHASE_C_VOLTAGE = "PhaseCVoltage"
    PHASE_A_CURRENT = "PhaseACurrent"
    PHASE_B_CURRENT = "PhaseBCurrent"
    PHASE_C_CURRENT = "PhaseCCurrent"
    PHASE_A_POWER_FACTOR = "PhaseAPowerFactor"
    PHASE_B_POWER_FACTOR = "PhaseBPowerFactor"
    PHASE_C_POWER_FACTOR = "PhaseCPowerFactor"
    PHASE_A_FREQUENCY = "PhaseAFrequency"
    PHASE_B_FREQUENCY = "PhaseBFrequency"
    PHASE_C_FREQUENCY = "PhaseCFrequency"
    HUMIDITY = "Humidity"
    IRRADIANCE = "Irradiance"
    CLEARED_PRICE = "ClearedPrice"
    CLEARED_VOLUME = "ClearedVolume"
    ORDER_PRICE = "OrderPrice"
    ORDER_VOLUME = "OrderVolume"
    INJECTION_PRICE = "InjectionPrice"
    OFFTAKE_PRICE = "OfftakePrice"
    INTERNAL_PRICE = "InternalPrice"


class ForecastTypeV2(StrEnum):
    """Forecast type of the v2 data models (``UserForecastPreferencesDM.forecast_type``)."""

    DETERMINISTIC = "DETERMINISTIC"
    STOCHASTIC = "STOCHASTIC"


_FORECAST_TYPE_TO_V1: dict[ForecastTypeV2, ForecastType] = {
    ForecastTypeV2.DETERMINISTIC: "deterministic",
    ForecastTypeV2.STOCHASTIC: "stochastic",
}


class TimeValueV2(BaseModel):
    """Timestamp-value pair of a request series. The unit is declared once, on the series."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(
        ..., description="Timestamp of the value (ISO 8601). Read as UTC when it has no offset."
    )
    value: float = Field(..., description="Measured value, in the unit of the series.")


class TimeParametersV2(TimeParameters):
    """
    Forecast resolution and horizon, with the field names of ``TimeParametersDM``.

    The identifiers of the platform record are accepted, so that the record can be passed as is, and are not used.
    """

    model_config = ConfigDict(extra="forbid")

    timeParametersId: str | None = Field(default=None, description="Identifier of the record. Not used.")
    hasActor: str | None = Field(
        default=None, description="Identifier of the actor the parameters belong to. Not used."
    )
    relatedToOptimizationInstance: str | None = Field(
        default=None, description="Identifier of the run the parameters apply to. Not used."
    )


class LocationV2(Location):
    """Geographic location (WGS84), with the ``latitude`` / ``longitude`` names of ``SiteDM`` and ``CommunityDM``."""

    model_config = ConfigDict(extra="forbid")


class TrainingSplitV2(BaseModel):
    """Number of validation and test windows. A count given here takes precedence over the matching ratio."""

    model_config = ConfigDict(extra="forbid")

    num_val_windows: int | None = Field(default=None, ge=1, description="Number of windows used for validation.")
    num_test_windows: int | None = Field(default=None, ge=1, description="Number of windows used for testing.")


class UserForecastPreferencesV2(BaseModel):
    """
    Training configuration, with the field names of ``UserForecastPreferencesDM``.

    ``model_id``, ``training_split`` and ``excluded_model_types`` are not in ``UserForecastPreferencesDM``. The
    identifiers of the platform record are accepted, so that the record can be passed as is, and are not used.
    """

    model_config = ConfigDict(extra="forbid")

    ufpId: str | None = Field(default=None, description="Identifier of the record. Not used.")
    forecast_type: ForecastTypeV2 = Field(..., description="Type of forecast model to train.")
    model_id: str = Field(..., description="Unique identifier for this model instance.")
    model_preset: str | None = Field(
        default=ModelConfig.model_fields["model_preset"].default,
        description="AutoGluon preset (e.g. 'chronos2', 'fast_training', 'high_quality').",
    )
    model_version: str | None = Field(default=None, description="Version identifier for the model (e.g. '1.0.0').")
    lookback_steps: int | None = Field(
        default=None, ge=1, description="Number of past timesteps used as input features."
    )
    loss_function: str | None = Field(
        default=None,
        description="Loss function to optimise (e.g. 'mse', 'mae', 'pinball'), passed to AutoGluon as eval_metric.",
    )
    quantile_levels: list[QuantileLevel] | None = Field(
        default=None, description="Quantile levels to predict (for quantile regression models)."
    )
    training_ratio: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Fraction of data used for training. Training uses the data before the validation and test windows.",
    )
    validation_ratio: float | None = Field(
        default=None, ge=0, le=1, description="Fraction of data used for validation, converted to num_val_windows."
    )
    test_ratio: float | None = Field(
        default=None, ge=0, le=1, description="Fraction of data used for testing, converted to num_test_windows."
    )
    training_split: TrainingSplitV2 = Field(
        default_factory=TrainingSplitV2, description="Number of validation and test windows."
    )
    excluded_model_types: list[str] | None = Field(
        default=None, description="Model types to exclude from training (e.g., ['DeepAR', 'PatchTST'])."
    )
    hasActor: str | None = Field(
        default=None, description="Identifier of the actor the preferences belong to. Not used."
    )
    relatedToOptimizationInstance: str | None = Field(
        default=None, description="Identifier of the run the preferences were applied in. Not used."
    )

    @model_validator(mode="after")
    def _check_ratios(self) -> Self:
        total = math.fsum(r for r in (self.training_ratio, self.validation_ratio, self.test_ratio) if r is not None)
        if total > 1 and not math.isclose(total, 1):
            raise ValueError(f"training_ratio, validation_ratio and test_ratio add up to {total:g}, more than 1.")
        return self


def _derive_missing_unit(data: object) -> object:
    """Fill in the missing one of ``unit`` and ``unit_symbol`` of raw series data from ``UNIT_SYMBOL_TO_QUDT``."""
    if not isinstance(data, dict):
        return data
    unit, unit_symbol = data.get("unit"), data.get("unit_symbol")
    if unit is None and isinstance(unit_symbol, str) and unit_symbol in UNIT_SYMBOL_TO_QUDT:
        return {**data, "unit": UNIT_SYMBOL_TO_QUDT[unit_symbol]}
    if unit_symbol is None and isinstance(unit, str) and unit in QUDT_TO_UNIT_SYMBOL:
        return {**data, "unit_symbol": QUDT_TO_UNIT_SYMBOL[unit]}
    return data


def _check_unit_matches_symbol(unit: str | None, unit_symbol: str | None) -> None:
    """Reject a ``unit`` and a ``unit_symbol`` that the unit table maps to two different units."""
    if unit_symbol is None or unit not in QUDT_TO_UNIT_SYMBOL:
        return
    expected = UNIT_SYMBOL_TO_QUDT.get(unit_symbol)
    if expected is not None and expected != unit:
        raise ValueError(f"unit '{unit}' does not match unit_symbol '{unit_symbol}' ('{expected}').")


class BaseTargetSeriesV2(BaseModel):
    """
    Target series of a v2 request, described as the ``ForecastSeriesDM`` it is forecast into.

    ``unit_symbol`` and ``unit`` describe the same unit. When one of them is missing, it is derived from the other
    with ``UNIT_SYMBOL_TO_QUDT``; the target is rejected when ``unit`` cannot be resolved.
    """

    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(..., description="Unique identifier for this series.")
    observedProperty: ObservedProperty = Field(..., description="Quantity to forecast.")
    hasFeatureOfInterest: str = Field(
        ...,
        min_length=1,
        description=(
            "Identifier of the subject whose property is forecast: the asset id, or the facility id for a "
            "facility-wide quantity such as the outdoor temperature."
        ),
    )
    asset_type: str | None = Field(
        default=None,
        description="Type of the asset, when the feature of interest is an asset. Not part of the v2 forecast.",
    )
    unit_symbol: str | None = Field(default=None, description="Human-readable unit symbol (e.g. 'kW', 'kWh', '°C').")
    unit: QudtUnit = Field(
        ...,
        description=f"QUDT unit IRI (e.g. 'unit:KiloW-HR'). Can be omitted when unit_symbol is one of: {_KNOWN_UNIT_SYMBOLS}.",
    )

    @model_validator(mode="before")
    @classmethod
    def _derive_unit(cls, data: object) -> object:
        data = _derive_missing_unit(data)
        if isinstance(data, dict) and data.get("unit") is None and isinstance(data.get("unit_symbol"), str):
            raise ValueError(
                f"unit_symbol '{data['unit_symbol']}' is not one of {_KNOWN_UNIT_SYMBOLS}: set unit to its QUDT unit IRI."
            )
        return data

    @model_validator(mode="after")
    def _check_unit(self) -> Self:
        _check_unit_matches_symbol(self.unit, self.unit_symbol)
        return self


class TrainingTargetSeriesV2(BaseTargetSeriesV2):
    """Target series of a v2 training request."""

    training_values: list[TimeValueV2] = Field(
        ..., min_length=2, description="Full historical ground truth time series used for training."
    )

    def to_v1(self) -> TrainingTargetSeries:
        """
        Convert to the v1 ``TrainingTargetSeries``.

        Returns
        -------
        TrainingTargetSeries
            The series with ``observedProperty`` as ``variable_name`` and ``hasFeatureOfInterest`` as ``asset_id``.

        """
        return TrainingTargetSeries(
            series_id=self.series_id,
            variable_name=self.observedProperty.value,
            asset_id=self.hasFeatureOfInterest,
            asset_type=self.asset_type,
            unit=self.unit_symbol,
            unitIri=_v1_unit_iri(self.unit),
            training_values=_v1_values(self.training_values),
        )


class ForecastTargetSeriesV2(BaseTargetSeriesV2):
    """Target series of a v2 forecast request."""

    historical_values: list[TimeValueV2] = Field(
        ..., min_length=1, description="Past observations of the target variable used as input for forecast."
    )

    def to_v1(self) -> ForecastTargetSeries:
        """
        Convert to the v1 ``ForecastTargetSeries``.

        Returns
        -------
        ForecastTargetSeries
            The series with ``observedProperty`` as ``variable_name`` and ``hasFeatureOfInterest`` as ``asset_id``.

        """
        return ForecastTargetSeries(
            series_id=self.series_id,
            variable_name=self.observedProperty.value,
            asset_id=self.hasFeatureOfInterest,
            asset_type=self.asset_type,
            unit=self.unit_symbol,
            unitIri=_v1_unit_iri(self.unit),
            historical_values=_v1_values(self.historical_values),
        )


class BaseExogenousSeriesV2(BaseModel):
    """
    Exogenous series of a v2 request.

    ``variable_name`` is the column name of the series in the model and defaults to ``observedProperty``. The trained
    model stores it as a covariate name, so it must be the same at training and at forecast time. ``unit_symbol``
    and ``unit`` are optional; when one of them is given, the other is derived with ``UNIT_SYMBOL_TO_QUDT``.
    """

    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(..., description="Unique identifier for this exogenous series.")
    observedProperty: ObservedProperty | None = Field(default=None, description="Quantity of the series.")
    variable_name: str = Field(
        ...,
        description="Column name of the series in the model. Defaults to observedProperty; one of the two is required.",
    )
    source: str | None = Field(default=None, description="Data source or provider.")
    unit_symbol: str | None = Field(default=None, description="Human-readable unit symbol (e.g. 'kW', 'kWh', '°C').")
    unit: QudtUnit | None = Field(default=None, description="QUDT unit IRI (e.g. 'unit:DEG_C').")

    @model_validator(mode="before")
    @classmethod
    def _derive_unit(cls, data: object) -> object:
        return _derive_missing_unit(data)

    @model_validator(mode="after")
    def _check_unit(self) -> Self:
        _check_unit_matches_symbol(self.unit, self.unit_symbol)
        return self

    @model_validator(mode="before")
    @classmethod
    def _default_variable_name(cls, data: object) -> object:
        if not isinstance(data, dict) or data.get("variable_name") is not None:
            return data
        if data.get("observedProperty") is None:
            raise ValueError("An exogenous series needs a variable_name or an observedProperty.")
        return {**data, "variable_name": str(data["observedProperty"])}


class TrainingExogenousSeriesV2(BaseExogenousSeriesV2):
    """Exogenous series of a v2 training request."""

    is_known_covariate: bool = Field(
        default=False,
        description="Indicates if this exogenous series is a known covariate (i.e., fully known for the forecast horizon).",
    )
    training_values: list[TimeValueV2] = Field(
        ..., min_length=2, description="Historical values aligned with the target training values."
    )

    def to_v1(self) -> TrainingExogenousSeries:
        """
        Convert to the v1 ``TrainingExogenousSeries``.

        Returns
        -------
        TrainingExogenousSeries
            The series, named by its ``variable_name``.

        """
        return TrainingExogenousSeries(
            series_id=self.series_id,
            variable_name=self.variable_name,
            source=self.source,
            unit=self.unit_symbol,
            unitIri=_v1_unit_iri(self.unit),
            is_known_covariate=self.is_known_covariate,
            training_values=_v1_values(self.training_values),
        )


class ForecastExogenousSeriesV2(BaseExogenousSeriesV2):
    """Exogenous series of a v2 forecast request."""

    historical_values: list[TimeValueV2] = Field(
        ..., min_length=1, description="Historical values aligned with target_series for forecast."
    )
    future_values: list[TimeValueV2] | None = Field(
        default=None, description="Known or forecasted future values over the forecast horizon."
    )

    def to_v1(self) -> ForecastExogenousSeries:
        """
        Convert to the v1 ``ForecastExogenousSeries``.

        Returns
        -------
        ForecastExogenousSeries
            The series, named by its ``variable_name``.

        """
        return ForecastExogenousSeries(
            series_id=self.series_id,
            variable_name=self.variable_name,
            source=self.source,
            unit=self.unit_symbol,
            unitIri=_v1_unit_iri(self.unit),
            historical_values=_v1_values(self.historical_values),
            future_values=_v1_values(self.future_values) if self.future_values is not None else None,
        )


class ForecastTrainingInputV2(BaseModel):
    """Input data model for training a forecasting model, in the v2 vocabulary."""

    model_config = ConfigDict(extra="forbid")

    hasTimeParameters: TimeParametersV2 = Field(
        ..., description="Target forecast resolution and horizon the model should learn."
    )
    hasUserForecastPreferences: UserForecastPreferencesV2 = Field(
        ..., description="Configuration parameters for the training process."
    )
    target_series: TrainingTargetSeriesV2 = Field(
        ..., description="The target variable with ground truth data for training."
    )
    location: LocationV2 | None = Field(default=None, description="Geographic location of the asset.")
    exogenous_series: list[TrainingExogenousSeriesV2] | None = Field(
        default=None, description="Optional additional time series of external variables."
    )
    features: list[FeatureName] = Field(
        default_factory=list, description="Feature-engineering steps to apply during training."
    )
    num_weekly_lags: int = Field(
        default=0,
        ge=0,
        description="Number of weekly lag/rolling-stat features to generate when the lag feature is enabled.",
    )

    @model_validator(mode="after")
    def _check_training_split(self) -> Self:
        self._training_split()
        return self

    def _training_split(self) -> TrainingSplit:
        preferences = self.hasUserForecastPreferences
        windows = {
            "num_val_windows": self._windows(preferences.training_split.num_val_windows, preferences.validation_ratio),
            "num_test_windows": self._windows(preferences.training_split.num_test_windows, preferences.test_ratio),
        }
        return TrainingSplit.model_validate({name: count for name, count in windows.items() if count is not None})

    def _windows(self, windows: int | None, ratio: float | None) -> int | None:
        if windows is not None or ratio is None:
            return windows
        timestep_minutes = self.hasTimeParameters.timestep_minutes
        if timestep_minutes is None:
            raise ValueError(
                "hasTimeParameters.timestep_minutes is required to convert validation_ratio and test_ratio to windows."
            )
        horizon_steps = max(1, self.hasTimeParameters.horizon_hours * 60 // timestep_minutes)
        # Rounded before flooring so that float error (0.35 * 1440 / 24 = 20.999999999999996) does not drop a window
        return max(1, math.floor(round(ratio * len(self.target_series.training_values) / horizon_steps, 9)))

    def to_v1(self) -> ForecastTrainingInput:
        """
        Convert to the v1 ``ForecastTrainingInput`` run by ``train``.

        Preferences set in the request are passed on to ``ModelConfig``, the others take its defaults, as in a v1
        request. ``loss_function`` becomes ``eval_metric``. A window count missing from ``training_split`` is
        derived from the matching ratio as ``max(1, floor(ratio * n_steps / horizon_steps))``, where ``n_steps`` is
        the number of training values and ``horizon_steps`` the horizon in timesteps. Without a count or a ratio, the
        ``TrainingSplit`` default applies. Timestamps keep their wall-clock time and drop their offset, as the v1
        pipeline does with them.

        Returns
        -------
        ForecastTrainingInput
            The same training request in the v1 data model.

        """
        preferences = self.hasUserForecastPreferences
        config = preferences.model_dump(include=_PREFERENCES_FOR_MODEL_CONFIG, exclude_unset=True)
        if "loss_function" in preferences.model_fields_set:
            config["eval_metric"] = _eval_metric(preferences.loss_function)
        return ForecastTrainingInput(
            forecast_type=_FORECAST_TYPE_TO_V1[preferences.forecast_type],
            time_parameters=_v1_time_parameters(self.hasTimeParameters),
            model_config=ModelConfig(model_id=preferences.model_id, training_split=self._training_split(), **config),
            target_series=self.target_series.to_v1(),
            location=_v1_location(self.location),
            exogenous_series=[series.to_v1() for series in self.exogenous_series] if self.exogenous_series else None,
            features=self.features,
            num_weekly_lags=self.num_weekly_lags,
        )


class ForecastInputV2(BaseModel):
    """Input data model for a forecasting request, in the v2 vocabulary."""

    model_config = ConfigDict(extra="forbid")

    forecastId: str | None = Field(
        default=None, description="Identifier of the forecast to produce. A UUID4 is generated when omitted."
    )
    forecast_type: ForecastTypeV2 = Field(..., description="Desired forecast output type.")
    hasTimeParameters: TimeParametersV2 = Field(..., description="Forecast horizon and time resolution parameters.")
    model_id: str | None = Field(default=None, description="Unique identifier for the model to use for forecast.")
    model_version: str | None = Field(default=None, description="Version identifier for the model (e.g., '1.0.0').")
    target_series: ForecastTargetSeriesV2 = Field(..., description="The primary variable to be forecasted.")
    location: LocationV2 | None = Field(
        default=None, description="Geographic location of the asset. Useful for weather-dependent forecasts."
    )
    exogenous_series: list[ForecastExogenousSeriesV2] | None = Field(
        default=None, description="Optional additional time series of external variables."
    )

    def to_v1(self) -> ForecastInput:
        """
        Convert to the v1 ``ForecastInput`` run by ``predict``.

        Timestamps keep their wall-clock time and drop their offset, as the v1 pipeline does with them.

        Returns
        -------
        ForecastInput
            The same forecast request in the v1 data model.

        """
        return ForecastInput(
            forecast_type=_FORECAST_TYPE_TO_V1[self.forecast_type],
            time_parameters=_v1_time_parameters(self.hasTimeParameters),
            model_id=self.model_id,
            model_version=self.model_version,
            target_series=self.target_series.to_v1(),
            location=_v1_location(self.location),
            exogenous_series=[series.to_v1() for series in self.exogenous_series] if self.exogenous_series else None,
        )


def _is_none(value: object) -> bool:
    return value is None


class QuantityValueV2(BaseModel):
    """Forecasted number and its unit (``QuantityValueDM``)."""

    model_config = ConfigDict(extra="forbid")

    numericValue: FiniteFloat = Field(..., description="Forecasted value.")
    unit: str = Field(..., description="QUDT unit IRI of the value (e.g. 'unit:KiloW').")


class ForecastedValueV2(BaseModel):
    """One forecasted value (``ForecastedValuePayloadDM``)."""

    model_config = ConfigDict(extra="forbid")

    timestamp: AwareDatetime = Field(..., description="Timestep the value refers to.")
    hasResult: QuantityValueV2 = Field(..., description="Forecasted value and its unit.")
    quantiles: dict[QuantileKey, FiniteFloat] | None = Field(
        default=None,
        exclude_if=_is_none,
        description="Quantile forecasts at this timestep, keyed by quantile level. Present for a stochastic forecast.",
    )


class ForecastSeriesV2(BaseModel):
    """One forecasted series (``ForecastSeriesPayloadDM``)."""

    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(..., description="Unique identifier of this series within the forecast.")
    observedProperty: ObservedProperty = Field(..., description="The quantity forecast.")
    hasFeatureOfInterest: str = Field(
        ..., min_length=1, description="Identifier of the subject whose property is forecast."
    )
    unit_symbol: str | None = Field(
        default=None, exclude_if=_is_none, description="Human-readable unit symbol (e.g. 'kW')."
    )
    unit: str = Field(..., description="QUDT unit IRI (e.g. 'unit:KiloW').")
    model_id: str = Field(..., description="Trained model that produced the series, as '<model_id>_<model_version>'.")
    hasValues: list[ForecastedValueV2] = Field(..., min_length=1, description="The forecasted values, in time order.")


class ForecastV2(BaseModel):
    """
    Forecast in the v2 data models (``ForecastPayloadDM``), as consumed by the optimisation engines.

    The model describes the forecasts this library produces and is stricter than the schema: ``forecast_date``,
    ``forecast_type``, the series ``unit`` and ``model_id`` are required, timestamps need a UTC offset and values must
    be finite. Optional fields without a value are left out of every dump, since the schema rejects ``null``: any
    ``model_dump_json()`` of a ``ForecastV2`` is a valid ``ForecastPayloadDM``.
    """

    model_config = ConfigDict(extra="forbid")

    forecastId: str = Field(..., description="Unique identifier of the forecast.")
    forecast_date: AwareDatetime = Field(..., description="Time at which the forecast was produced.")
    forecast_type: ForecastTypeV2 = Field(..., description="Type of forecast.")
    hasForecastSeries: list[ForecastSeriesV2] = Field(..., min_length=1, description="The forecasted series.")


def forecast_to_v2(forecast: Forecast, request: ForecastInputV2) -> ForecastV2:
    """
    Convert a v1 ``Forecast`` to the v2 forecast payload.

    The series identity and units come from the v2 request the forecast answers. ``model_id`` is
    ``<model_id>_<model_version>``, the directory name of the trained model. Quantiles are kept for a stochastic
    forecast only. Value timestamps without an offset take the offset of the last historical value of the request,
    or UTC when it has none; a ``forecast_date`` without an offset is read as UTC.

    Parameters
    ----------
    forecast : Forecast
        The forecast produced by ``predict`` for ``request.to_v1()``.
    request : ForecastInputV2
        The v2 forecast request.

    Returns
    -------
    ForecastV2
        The forecast payload, with the ``forecastId`` of the request or a new UUID4.

    """
    target = request.target_series
    forecast_type = ForecastTypeV2(forecast.forecast_type.upper())
    keep_quantiles = forecast_type is ForecastTypeV2.STOCHASTIC
    request_tzinfo = target.historical_values[-1].timestamp.tzinfo or UTC
    values = [
        ForecastedValueV2(
            timestamp=_with_tzinfo(value.time, request_tzinfo),
            hasResult=QuantityValueV2(numericValue=value.value, unit=target.unit),
            quantiles=value.quantiles if keep_quantiles else None,
        )
        for value in forecast.forecast_series.values
    ]
    series = ForecastSeriesV2(
        series_id=forecast.forecast_series.series_id,
        observedProperty=target.observedProperty,
        hasFeatureOfInterest=target.hasFeatureOfInterest,
        unit_symbol=target.unit_symbol,
        unit=target.unit,
        model_id=f"{request.model_id}_{request.model_version}",
        hasValues=values,
    )
    return ForecastV2(
        forecastId=request.forecastId if request.forecastId is not None else str(uuid4()),
        forecast_date=_with_tzinfo(forecast.forecast_date, UTC),
        forecast_type=forecast_type,
        hasForecastSeries=[series],
    )


def _with_tzinfo(timestamp: datetime, default: tzinfo) -> datetime:
    return timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=default)


def _eval_metric(loss_function: str | None) -> str | None:
    if loss_function is None:
        return None
    return _LOSS_FUNCTION_TO_EVAL_METRIC.get(loss_function.lower(), loss_function.upper())


def _v1_unit_iri(unit: str | None) -> AnyUrl | None:
    return AnyUrl(QUDT_UNIT_NAMESPACE + unit.removeprefix("unit:")) if unit is not None else None


def _v1_values(values: list[TimeValueV2]) -> list[TimeValue]:
    return [TimeValue(time=value.timestamp.replace(tzinfo=None), value=value.value) for value in values]


def _v1_time_parameters(time_parameters: TimeParametersV2) -> TimeParameters:
    return TimeParameters.model_validate(time_parameters.model_dump(include=set(TimeParameters.model_fields)))


def _v1_location(location: LocationV2 | None) -> Location | None:
    return Location.model_validate(location.model_dump()) if location is not None else None
