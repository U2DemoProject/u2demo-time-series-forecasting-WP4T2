"""Data model version (v1 or v2) of raw training and forecast payloads."""

from typing import Any, Literal, get_args

from u2demo_time_series_forecasting.core.logger import logger
from u2demo_time_series_forecasting.data_models import ForecastInput, ForecastTrainingInput, ForecastType
from u2demo_time_series_forecasting.data_models_v2 import ForecastInputV2, ForecastTrainingInputV2, ForecastTypeV2

type PayloadVersion = Literal["v1", "v2"]

PAYLOAD_VERSIONS: tuple[PayloadVersion, ...] = get_args(PayloadVersion.__value__)

_V1_KEYS = ("time_parameters", "model_config", "training_model_config")
_V2_KEYS = ("hasTimeParameters", "hasUserForecastPreferences", "forecastId")
_V1_TARGET_SERIES_KEYS = ("variable_name", "asset_id", "unitIri")
_V2_TARGET_SERIES_KEYS = ("observedProperty", "hasFeatureOfInterest")
_V1_FORECAST_TYPES: tuple[str, ...] = get_args(ForecastType.__value__)
_V2_FORECAST_TYPES: tuple[str, ...] = tuple(ForecastTypeV2)


def detect_payload_version(payload: dict[str, Any]) -> PayloadVersion:
    """
    Detect the data model version of a raw training or forecast payload.

    A payload is v2 when it has a v2-only top-level key (``hasTimeParameters``, ``hasUserForecastPreferences``,
    ``forecastId``), an upper-case ``forecast_type`` (``DETERMINISTIC``, ``STOCHASTIC``), or a ``target_series``
    with ``observedProperty`` or ``hasFeatureOfInterest``. Any other payload is v1. As v1 models accept unknown
    fields, a v1 payload carrying a field with one of these v2 names is rejected as mixed: validate it with an
    explicit version instead.

    Parameters
    ----------
    payload : dict[str, Any]
        The decoded JSON payload.

    Returns
    -------
    PayloadVersion
        ``"v1"`` or ``"v2"``.

    Raises
    ------
    ValueError
        If the payload is not a JSON object, or if it has both v2 fields and v1 fields: ``time_parameters``,
        ``model_config``, ``training_model_config``, a lower-case ``forecast_type``, or ``variable_name``,
        ``asset_id`` or ``unitIri`` in ``target_series``.

    """
    if not isinstance(payload, dict):
        raise ValueError(f"A payload must be a JSON object, not {type(payload).__name__}.")
    v1_fields, v2_fields = _version_fields(payload)
    if v1_fields and v2_fields:
        raise ValueError(
            f"The payload mixes v1 fields ({', '.join(v1_fields)}) and v2 fields ({', '.join(v2_fields)}). "
            "Use the fields of one version, or set the version explicitly."
        )
    version: PayloadVersion = "v2" if v2_fields else "v1"
    logger.info(f"Detected payload version {version}")
    return version


def parse_training_payload(
    payload: dict[str, Any], version: PayloadVersion | None = None
) -> ForecastTrainingInput | ForecastTrainingInputV2:
    """
    Validate a raw training payload against the data model of its version.

    Parameters
    ----------
    payload : dict[str, Any]
        The decoded JSON payload.
    version : {"v1", "v2"}, optional
        Data model version of the payload. Detected with ``detect_payload_version`` when omitted.

    Returns
    -------
    ForecastTrainingInput or ForecastTrainingInputV2
        The validated training request.

    Raises
    ------
    ValueError
        If ``version`` is unknown, or if it is omitted and the payload mixes v1 and v2 fields.
    pydantic.ValidationError
        If the payload does not match the data model of the version.

    """
    if _resolve_version(payload, version) == "v2":
        return ForecastTrainingInputV2.model_validate(payload)
    return ForecastTrainingInput.model_validate(payload)


def parse_forecast_payload(
    payload: dict[str, Any], version: PayloadVersion | None = None
) -> ForecastInput | ForecastInputV2:
    """
    Validate a raw forecast payload against the data model of its version.

    Parameters
    ----------
    payload : dict[str, Any]
        The decoded JSON payload.
    version : {"v1", "v2"}, optional
        Data model version of the payload. Detected with ``detect_payload_version`` when omitted.

    Returns
    -------
    ForecastInput or ForecastInputV2
        The validated forecast request.

    Raises
    ------
    ValueError
        If ``version`` is unknown, or if it is omitted and the payload mixes v1 and v2 fields.
    pydantic.ValidationError
        If the payload does not match the data model of the version.

    """
    if _resolve_version(payload, version) == "v2":
        return ForecastInputV2.model_validate(payload)
    return ForecastInput.model_validate(payload)


def _resolve_version(payload: dict[str, Any], version: PayloadVersion | None) -> PayloadVersion:
    if version is None:
        return detect_payload_version(payload)
    if version not in PAYLOAD_VERSIONS:
        raise ValueError(f"Unknown payload version '{version}'. Expected one of: {', '.join(PAYLOAD_VERSIONS)}.")
    return version


def _version_fields(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    """List the fields of ``payload`` that only exist in v1 and those that only exist in v2."""
    target_series = payload.get("target_series")
    if not isinstance(target_series, dict):
        target_series = {}
    v1_fields = [key for key in _V1_KEYS if key in payload]
    v1_fields += [f"target_series.{key}" for key in _V1_TARGET_SERIES_KEYS if key in target_series]
    v2_fields = [key for key in _V2_KEYS if key in payload]
    v2_fields += [f"target_series.{key}" for key in _V2_TARGET_SERIES_KEYS if key in target_series]
    forecast_type = payload.get("forecast_type")
    if forecast_type in _V1_FORECAST_TYPES:
        v1_fields.append(f"forecast_type={forecast_type}")
    elif forecast_type in _V2_FORECAST_TYPES:
        v2_fields.append(f"forecast_type={forecast_type}")
    return v1_fields, v2_fields
