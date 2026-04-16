from enum import StrEnum


class FeatureName(StrEnum):
    """Canonical names for feature-engineering steps."""

    TEMPORAL = "temporal"
    LAG = "lag"
    HOLIDAY = "holiday"
    WEATHER = "weather"
