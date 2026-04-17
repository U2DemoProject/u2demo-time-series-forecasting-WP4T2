"""Container and feature-engineering pipeline for time-series forecasting data."""

from collections.abc import Iterable
from typing import Self

import pandas as pd
from joblib import Memory

from u2demo_time_series_forecasting.constants.columns import (
    TIME_SERIES_GROUP_COLUMN_NAME,
    TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)
from u2demo_time_series_forecasting.constants.paths import CACHE_DIR
from u2demo_time_series_forecasting.core.logger import logger
from u2demo_time_series_forecasting.data_models import (
    ForecastExogenousSeries,
    ForecastInput,
    ForecastTrainingInput,
    TrainingExogenousSeries,
)
from u2demo_time_series_forecasting.features.holidays import merge_holidays_with_dataframe
from u2demo_time_series_forecasting.features.lag import LagFeatureGenerator
from u2demo_time_series_forecasting.features.temporal import add_encoded_time_features
from u2demo_time_series_forecasting.features.weather import enrich_weather

memory = Memory(location=CACHE_DIR, verbose=0)


def _infer_frequency(timestamps: pd.Series | pd.Index, expected_minutes: int | None) -> tuple[str, int, pd.Timedelta]:
    """
    Infer the frequency of a timestamp column and validate it against an expectation.

    Parameters
    ----------
    timestamps : pandas.Series or pandas.Index
        Datetime-like timestamps assumed to be regularly spaced.
    expected_minutes : int | None
        Expected step size in minutes (from the input data model). When provided, an
        error is raised if it disagrees with the inferred frequency.

    Returns
    -------
    tuple[str, int, pandas.Timedelta]
        ``(freq, freq_minutes, freq_timedelta)`` derived from ``timestamps``.

    Raises
    ------
    ValueError
        If the timestamps are irregular (``pd.infer_freq`` returns ``None``) or the
        inferred frequency does not match ``expected_minutes``.

    """
    # infer_freq returns None for irregular timestamps; catch early with a clear message
    freq = pd.infer_freq(pd.DatetimeIndex(timestamps))
    if freq is None:
        raise ValueError(
            "Could not infer a consistent frequency from the timestamps. Check for irregular spacing in the input data."
        )
    offset = pd.tseries.frequencies.to_offset(freq=freq)
    if not isinstance(
        offset, pd.offsets.Tick
    ):  # pragma: no cover - infer_freq never yields non-Tick for supported h/D inputs
        raise ValueError(f"Unsupported non-tick frequency: {freq!r}. Only hourly and daily frequencies are supported.")
    freq_minutes = int(pd.Timedelta(offset).total_seconds() / 60)
    if expected_minutes is not None and expected_minutes != freq_minutes:
        raise ValueError(
            f"Inferred time series frequency `{freq}` (`{freq_minutes}min`) does not match "
            f"expected frequency `{expected_minutes}min` based on time parameters."
        )
    return freq, freq_minutes, pd.to_timedelta(freq_minutes, unit="min")


def _normalize_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the housekeeping shared by training and forecasting inputs.

    Adds the (single) group column, drops timezone information from timestamps,
    and validates that the resulting frame is non-empty and free of NaNs.
    """
    if df.empty:
        raise ValueError("The input data is empty.")
    if df.isna().to_numpy().any():
        raise ValueError("The input data contains NaN values.")
    df[TIME_SERIES_GROUP_COLUMN_NAME] = 0
    df[TIME_SERIES_TIMESTAMP_COLUMN_NAME] = df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].dt.tz_localize(None)
    return df


def _check_unique_variable_names(
    exogenous_series: Iterable[TrainingExogenousSeries] | Iterable[ForecastExogenousSeries],
) -> None:
    """Reject duplicate exogenous variable names early to prevent silent column duplication."""
    variable_names = [exo.variable_name for exo in exogenous_series]
    duplicates = sorted({n for n in variable_names if variable_names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate exogenous variable names detected: {duplicates}. "
            "Each exogenous series must have a unique variable_name."
        )


def _stack_exogenous_columns(named_series: list[tuple[str, list[dict]]]) -> pd.DataFrame:
    """Stack one column per ``(variable_name, value_rows)`` pair, indexed by timestamp."""
    columns = [
        pd.DataFrame(rows).rename(columns={"value": variable_name}).set_index(TIME_SERIES_TIMESTAMP_COLUMN_NAME)
        for variable_name, rows in named_series
        if rows
    ]
    return pd.concat(columns, axis=1).reset_index()


def _build_training_dataframe(data: ForecastTrainingInput) -> pd.DataFrame:
    """Project a training input data model into a flat DataFrame ready for feature engineering."""
    target_df = pd.DataFrame([value.model_dump() for value in data.target_series.training_values])
    if data.exogenous_series:
        _check_unique_variable_names(data.exogenous_series)
        named_series = [
            (exo.variable_name, [v.model_dump() for v in exo.training_values])
            for exo in data.exogenous_series
            if exo.training_values is not None
        ]
        exogenous_df = _stack_exogenous_columns(named_series)
        target_df = pd.merge(target_df, exogenous_df, how="left", on=TIME_SERIES_TIMESTAMP_COLUMN_NAME)
    return _normalize_input_dataframe(target_df)


def _build_forecast_exogenous_dataframe(
    exogenous_series: Iterable[ForecastExogenousSeries],
) -> pd.DataFrame:
    """Stack forecast-side exogenous columns, including future values when present."""
    named_series = [
        (
            exo.variable_name,
            [v.model_dump() for v in exo.historical_values]
            + ([v.model_dump() for v in exo.future_values] if exo.future_values is not None else []),
        )
        for exo in exogenous_series
    ]
    return _stack_exogenous_columns(named_series)


def _build_forecast_dataframe(data: ForecastInput) -> pd.DataFrame:
    """Project a forecast input data model into a flat DataFrame containing only historical rows."""
    target_df = pd.DataFrame([value.model_dump() for value in data.target_series.historical_values])
    return _normalize_input_dataframe(target_df)


def _extract_latlon(data: ForecastInput | ForecastTrainingInput) -> tuple[float, float] | None:
    return (data.location.latitude, data.location.longitude) if data.location else None


class U2TimeSeriesForecastingDataset:
    """
    Container and feature-engineering pipeline for U2Demo time-series forecasting models.

    A dataset always wraps a non-empty DataFrame and the frequency metadata derived
    from its timestamp column. Construct one via ``from_training_input``,
    ``from_forecast_input``, or ``from_dataframe`` rather than calling the
    constructor directly.

    Attributes
    ----------
    df : pandas.DataFrame
        The full time-series dataset (historical + optionally extended rows).
    future_df : pandas.DataFrame or None
        The final ``horizon`` rows extracted by ``split_past_and_future_dfs``.
    added_columns : dict[str, list[str]]
        Maps feature category names to lists of newly added column names.
    freq : str
        Inferred pandas frequency string (e.g. ``"h"``, ``"D"``).
    freq_minutes : int
        Step size in minutes corresponding to ``freq``.
    freq_timedelta : pandas.Timedelta
        Timedelta equivalent of the inferred frequency.
    final_timestamp : pandas.Timestamp or None
        The latest historical timestamp; populated only when constructed from a
        forecast input where past/future need to be distinguished.
    latlon : tuple[float, float] or None
        Latitude and longitude coordinates for geocoding and weather features.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        freq: str,
        freq_minutes: int,
        freq_timedelta: pd.Timedelta,
        *,
        latlon: tuple[float, float] | None = None,
        final_timestamp: pd.Timestamp | None = None,
    ) -> None:
        self.df: pd.DataFrame = df
        self.future_df: pd.DataFrame | None = None
        self.added_columns: dict[str, list[str]] = {}
        self.freq: str = freq
        self.freq_minutes: int = freq_minutes
        self.freq_timedelta: pd.Timedelta = freq_timedelta
        self.latlon: tuple[float, float] | None = latlon
        self.final_timestamp: pd.Timestamp | None = final_timestamp

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        latlon: tuple[float, float] | None = None,
        final_timestamp: pd.Timestamp | None = None,
    ) -> Self:
        """
        Build a dataset from an already-shaped DataFrame, inferring frequency from its timestamps.

        Intended for tests and ad-hoc use where a full input data model is overkill.
        The DataFrame must already contain the canonical ``timestamp`` and ``item_id``
        columns and be free of NaNs. Frequency is inferred from the first group's
        timestamps so multi-group frames with overlapping timestamps still work.
        """
        first_group = next(iter(df.groupby(TIME_SERIES_GROUP_COLUMN_NAME)))[1]
        freq, freq_minutes, freq_timedelta = _infer_frequency(
            first_group[TIME_SERIES_TIMESTAMP_COLUMN_NAME], expected_minutes=None
        )
        return cls(
            df=df,
            freq=freq,
            freq_minutes=freq_minutes,
            freq_timedelta=freq_timedelta,
            latlon=latlon,
            final_timestamp=final_timestamp,
        )

    @classmethod
    def from_training_input(cls, data: ForecastTrainingInput) -> Self:
        """
        Create a dataset from a validated ``ForecastTrainingInput`` data model.

        Raises
        ------
        ValueError
            If the input is empty, contains NaN values, has duplicate exogenous
            variable names, or carries timestamps with an irregular frequency.

        """
        df = _build_training_dataframe(data)
        freq, freq_minutes, freq_timedelta = _infer_frequency(
            df[TIME_SERIES_TIMESTAMP_COLUMN_NAME],
            data.time_parameters.timestep_minutes,
        )
        return cls(
            df=df,
            freq=freq,
            freq_minutes=freq_minutes,
            freq_timedelta=freq_timedelta,
            latlon=_extract_latlon(data),
        )

    @classmethod
    def from_forecast_input(cls, data: ForecastInput) -> Self:
        """
        Create a dataset from a validated ``ForecastInput`` data model.

        The dataset is extended by the requested forecast horizon and any provided
        exogenous covariates are merged in (including their future values).

        Raises
        ------
        ValueError
            If the input is empty, contains NaN values, or carries timestamps with
            an irregular frequency.

        """
        target_df = _build_forecast_dataframe(data)
        timestamp_col = target_df[TIME_SERIES_TIMESTAMP_COLUMN_NAME]
        freq, freq_minutes, freq_timedelta = _infer_frequency(timestamp_col, data.time_parameters.timestep_minutes)
        final_timestamp = timestamp_col.max()

        dataset = cls(
            df=target_df,
            freq=freq,
            freq_minutes=freq_minutes,
            freq_timedelta=freq_timedelta,
            latlon=_extract_latlon(data),
            final_timestamp=final_timestamp,
        )

        # Ensure at least 1 step when horizon < freq (e.g. 12h horizon with daily data)
        dataset.extend_by_horizon(max(1, data.time_parameters.horizon_hours * 60 // freq_minutes))

        if data.exogenous_series:
            exogenous_df = _build_forecast_exogenous_dataframe(data.exogenous_series)
            dataset.df = pd.merge(dataset.df, exogenous_df, how="left", on=TIME_SERIES_TIMESTAMP_COLUMN_NAME)

        if dataset.df.empty:
            raise ValueError("The input data is empty.")

        return dataset

    def drop_short_timeseries(self, min_timeseries_length: int) -> None:
        """
        Drop time-series groups that don't have enough rows for forecasting or training.

        Parameters
        ----------
        min_timeseries_length : int
            Minimum number of rows required per time-series. Groups with fewer rows
            than this threshold are dropped.

        Raises
        ------
        ValueError
            If no time-series remain after filtering.

        """
        counts = self.df[TIME_SERIES_GROUP_COLUMN_NAME].value_counts()
        dropped = [idx for idx, cnt in counts.items() if cnt < min_timeseries_length]
        if dropped:
            logger.warning(
                f"The following groups were dropped because they have too few rows to predict on (<{min_timeseries_length}):"
            )
            for d in dropped:
                logger.warning(f"\t{d}")
            self.df = self.df[~self.df[TIME_SERIES_GROUP_COLUMN_NAME].isin(dropped)]
        if self.df.empty:
            error = (
                f"No timeseries in the input has enough rows to predict/train on "
                f"(must have >= {min_timeseries_length} observations)."
            )
            logger.error(error)
            raise ValueError(error)

    def extend_by_horizon(self, horizon: int) -> None:
        """
        Append ``horizon`` empty rows per group at the inferred frequency.

        Parameters
        ----------
        horizon : int
            Number of future time steps to append per group. Each appended row is
            blank (NaN) for every column except the group identifier and timestamp.

        """
        frames: list[pd.DataFrame] = []
        for item_id, subdf in self.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME):
            last_ts = subdf[TIME_SERIES_TIMESTAMP_COLUMN_NAME].iloc[-1]
            new_timestamps = [last_ts + self.freq_timedelta * (i + 1) for i in range(horizon)]
            blank = pd.DataFrame(
                {
                    TIME_SERIES_GROUP_COLUMN_NAME: item_id,
                    TIME_SERIES_TIMESTAMP_COLUMN_NAME: new_timestamps,
                }
            )
            frames.append(pd.concat([subdf, blank], ignore_index=True))
        self.df = pd.concat(frames, ignore_index=True)

    def split_past_and_future_dfs(self, horizon: int) -> None:
        """
        Separate the final ``horizon`` rows of each group into ``future_df``.

        Parameters
        ----------
        horizon : int
            Number of most recent rows per group to designate as future data.
            ``horizon == 0`` is a no-op on ``self.df`` and produces an empty
            ``future_df``; this is an explicit guard because ``-0 == 0`` in Python
            would otherwise make ``iloc[:-0]`` silently return an empty slice.

        """
        self.df = self.df.sort_values([TIME_SERIES_GROUP_COLUMN_NAME, TIME_SERIES_TIMESTAMP_COLUMN_NAME])
        remaining_frames: list[pd.DataFrame] = []
        future_frames: list[pd.DataFrame] = []

        for _item_id, subdf in self.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME, group_keys=False):
            future_part = subdf.tail(horizon)
            if horizon == 0:
                remaining_part = subdf
            elif len(subdf) > horizon:
                remaining_part = subdf.iloc[:-horizon]
            else:
                remaining_part = subdf.iloc[0:0]

            future_frames.append(future_part)
            remaining_frames.append(remaining_part)

        self.df = pd.concat(remaining_frames, ignore_index=True)
        self.future_df = pd.concat(future_frames, ignore_index=True)

    def add_temporal(self) -> None:
        """
        Add encoded temporal features (day-of-week, day-of-year spline, sin/cos hour and month).

        Calling this method twice is idempotent: previously-added temporal columns
        are dropped before the new ones are computed.
        """
        # Drop previously-added temporal columns so calling add_temporal() twice is idempotent
        if "temporal" in self.added_columns:
            self.df = self.df.drop(columns=self.added_columns["temporal"], errors="ignore")
        self.df, self.added_columns["temporal"] = add_encoded_time_features(self.df)

    def add_lags(self, lags: list[int], windows: list[int]) -> None:
        """
        Add lagged target values and rolling statistical features.

        Parameters
        ----------
        lags : list[int]
            Lag offsets to compute (e.g. ``[1, 2, 24]``).
        windows : list[int]
            Window lengths for rolling statistics (mean, std, min, max).

        """
        lag_gen = LagFeatureGenerator(self.df, TIME_SERIES_GROUP_COLUMN_NAME)
        # Up to now lags have only ever been calculated on the value column
        target_columns = [TIME_SERIES_PREDICT_VALUE_COLUMN_NAME]
        for col in target_columns:
            lag_gen.create_lag_features(col, lags=lags)
            lag_gen.create_rolling_stat_features(col, windows=windows)
        self.df, self.added_columns["lag"] = lag_gen.finalize()

    def add_holidays(self) -> None:
        """
        Merge holiday indicators into the dataset using ``self.latlon``.

        Raises
        ------
        AttributeError
            If ``self.latlon`` is not set.

        """
        if self.latlon is None:
            raise AttributeError(
                "Location information is required to add `holiday` features but is missing in the dataset."
            )
        self.df, self.added_columns["holiday"] = merge_holidays_with_dataframe(
            self.df, self.latlon, TIME_SERIES_TIMESTAMP_COLUMN_NAME
        )

    def add_weather(self) -> None:
        """
        Enrich the dataset with weather features aligned to each timestamp and location.

        Weather is fetched relative to ``self.final_timestamp`` (the boundary between
        observed and forecast intervals) and resampled to the dataset's frequency.

        Raises
        ------
        AttributeError
            If ``self.latlon`` is not set.

        """
        if self.latlon is None:
            raise AttributeError(
                "Location information is required to add `weather` features but is missing in the dataset."
            )
        self.df, self.added_columns["weather"] = enrich_weather(
            self.df,
            TIME_SERIES_TIMESTAMP_COLUMN_NAME,
            self.latlon,
            self.final_timestamp,
            self.freq,
        )
