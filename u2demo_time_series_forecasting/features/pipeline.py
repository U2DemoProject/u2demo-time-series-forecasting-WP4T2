from collections.abc import Sequence

from u2demo_time_series_forecasting.dataset import U2TimeSeriesPredictionDataset
from u2demo_time_series_forecasting.features.names import FeatureName


def apply_features(
    time_series: U2TimeSeriesPredictionDataset,
    features: Sequence[FeatureName | str],
    *,
    lags: Sequence[int] | None = None,
    windows: Sequence[int] | None = None,
) -> None:
    """
    Run the shared feature-engineering pipeline on ``time_series`` in place.

    ``lags`` and ``windows`` are only required when :class:`FeatureName.LAG` is requested.
    """
    selected = {FeatureName(f) for f in features}

    if FeatureName.TEMPORAL in selected:
        time_series.add_temporal()
    if FeatureName.LAG in selected:
        if lags is None or windows is None:
            raise ValueError("lag feature requires lags and windows")
        time_series.add_lags(list(lags), list(windows))
    if FeatureName.HOLIDAY in selected:
        time_series.add_holidays()
    if FeatureName.WEATHER in selected:
        time_series.add_weather()
