from typing import Self

import pandas as pd
from joblib import Memory

from u2.data_models import ForecastInput, ForecastTrainingInput
from u2.defaults import (
    CACHE_DIR,
    TIME_SERIES_GROUP_COLUMN_NAME,
    TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)
from u2.features.holidays import merge_holidays_with_dataframe
from u2.features.lag import LagFeatureGenerator
from u2.features.temporal import add_encoded_time_features
from u2.features.weather import enrich_weather
from u2.logging_config import logger

memory = Memory(location=CACHE_DIR, verbose=0)


class U2TimeSeriesPredictionDataset:
    """
    Container and feature-engineering pipeline for time-series data used in
    U2 forecasting models.

    This class:
    - Loads raw time series data
    - Validates and normalizes column names
    - Ensures consistent frequency and aligned final timestamps
    - Splits past and future data for forecasting
    - Applies temporal, lag, holiday, and weather feature transformations
    - Extends each series for a specified prediction horizon

    Attributes
    ----------
    df : pandas.DataFrame or None
        The full time-series dataset (historical + optionally extended rows).
    future_df : pandas.DataFrame or None
        The final `horizon` rows extracted for prediction.
    added_columns : dict
        Maps feature category names to lists of newly added column names.
    freq : str or None
        Inferred frequency across groups ("h" or "D").
    freq_timedelta : datetime.timedelta or None
        Timedelta equivalent of the inferred frequency.
    final_timestamp : pandas.Timestamp or None
        The latest timestamp across all groups in the dataset.
    latlon : tuple[float, float] or None
        Latitude and longitude coordinates for geocoding and weather features.

    """

    def __init__(self):
        self.df = None
        self.future_df = None
        self.added_columns = {}
        self.freq = None
        self.freq_timedelta = None
        self.freq_minutes = None
        self.final_timestamp = None
        self.latlon = None

    @classmethod
    def from_training_input(cls, data: ForecastTrainingInput) -> Self:
        """
        Create a dataset instance from a structured data model.

        Parameters
        ----------
        data : ForecastTrainingInput
            A structured data model containing time-series information, including
            target series, exogenous series, and time parameters.

        Returns
        -------
        U2TimeSeriesPredictionDataset
            A dataset instance initialized from the provided data model.

        Raises
        ------
        ValueError
            If the input data model is invalid or cannot be converted to a DataFrame.

        """
        obj = cls()
        target_df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.training_values], orient="columns"
        )
        if data.exogenous_series:
            exogenous_df = pd.concat(
                [
                    pd.DataFrame.from_dict(
                        data=[value.model_dump() for value in exog_series.training_values], orient="columns"
                    )
                    .rename(columns={"value": exog_series.variable_name})
                    .set_index(TIME_SERIES_TIMESTAMP_COLUMN_NAME)
                    for exog_series in data.exogenous_series
                ],
                axis=1,
            ).reset_index()

            obj.df = pd.merge(target_df, exogenous_df, how="left", on=TIME_SERIES_TIMESTAMP_COLUMN_NAME)
        else:
            obj.df = target_df

        if obj.df.empty:
            error = "The input data is empty."
            raise ValueError(error)

        if obj.df.isna().any().any():
            error = "The input data contains NaN values."
            raise ValueError(error)

        obj.df[TIME_SERIES_GROUP_COLUMN_NAME] = 0

        obj.df[TIME_SERIES_TIMESTAMP_COLUMN_NAME] = obj.df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].dt.tz_localize(None)

        freq = pd.infer_freq(obj.df[TIME_SERIES_TIMESTAMP_COLUMN_NAME])
        freq_minutes = int(pd.Timedelta(pd.tseries.frequencies.to_offset(freq=freq)).total_seconds() / 60)

        if data.time_parameters.timestep_minutes is not None and data.time_parameters.timestep_minutes != freq_minutes:
            error = f"Inferred time series frequency `{freq}` (`{freq_minutes}min`) does not match expected frequency `{data.time_parameters.timestep_minutes}min` based on time parameters."
            raise ValueError(error)
        obj.freq = freq
        obj.freq_minutes = freq_minutes
        obj.freq_timedelta = pd.to_timedelta(freq_minutes, unit="min")

        obj.latlon = (data.location.latitude, data.location.longitude) if data.location else None

        return obj

    @classmethod
    def from_forecast_input(cls, data: ForecastInput) -> Self:
        """
        Create a dataset instance from a structured data model.

        Parameters
        ----------
        data : ForecastInput
            A structured data model containing time-series information, including
            target series, exogenous series, and time parameters.

        Returns
        -------
        U2TimeSeriesPredictionDataset
            A dataset instance initialized from the provided data model.

        Raises
        ------
        ValueError
            If the input data model is invalid or cannot be converted to a DataFrame.

        """
        obj = cls()
        obj.df = pd.DataFrame.from_dict(
            data=[value.model_dump() for value in data.target_series.historical_values], orient="columns"
        )
        obj.df[TIME_SERIES_GROUP_COLUMN_NAME] = 0
        obj.df[TIME_SERIES_TIMESTAMP_COLUMN_NAME] = obj.df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].dt.tz_localize(None)

        freq = pd.infer_freq(obj.df[TIME_SERIES_TIMESTAMP_COLUMN_NAME])
        freq_minutes = int(pd.Timedelta(pd.tseries.frequencies.to_offset(freq=freq)).total_seconds() / 60)

        if data.time_parameters.timestep_minutes is not None and data.time_parameters.timestep_minutes != freq_minutes:
            error = f"Inferred time series frequency `{freq}` (`{freq_minutes}min`) does not match expected frequency `{data.time_parameters.timestep_minutes}min` based on time parameters."
            raise ValueError(error)

        obj.freq = freq
        obj.freq_minutes = freq_minutes
        obj.freq_timedelta = pd.to_timedelta(freq_minutes, unit="min")

        obj.final_timestamp = obj.df[TIME_SERIES_TIMESTAMP_COLUMN_NAME].max()
        obj.latlon = (data.location.latitude, data.location.longitude) if data.location else None

        if obj.df.isna().any().any():
            error = "The input data contains NaN values."
            raise ValueError(error)

        obj.extend_by_horizon(data.time_parameters.horizon_hours * 60 // freq_minutes)

        if data.exogenous_series:
            exogenous_df = pd.concat(
                [
                    pd.DataFrame.from_dict(
                        data=[value.model_dump() for value in exog_series.historical_values]
                        + (
                            [value.model_dump() for value in exog_series.future_values]
                            if exog_series.future_values is not None
                            else []
                        ),
                        orient="columns",
                    )
                    .rename(columns={"value": exog_series.variable_name})
                    .set_index(TIME_SERIES_TIMESTAMP_COLUMN_NAME)
                    for exog_series in data.exogenous_series
                ],
                axis=1,
            ).reset_index()

            obj.df = pd.merge(obj.df, exogenous_df, how="left", on=TIME_SERIES_TIMESTAMP_COLUMN_NAME)

        if obj.df.empty:
            error = "The input data is empty."
            raise ValueError(error)

        return obj

    def drop_short_timeseries(self, min_timeseries_length: int) -> None:
        """
        Drop time-series from the dataset that do not have enough rows for prediction/training.

        For each group in the specified column:
        - Counts the number of rows per time-series
        - Identifies time-series with fewer rows than `min_timeseries_length`
        - Logs a warning listing the dropped time-series
        - Removes rows belonging to these short time-series from `self.df`
        - Raises a ValueError if no time-series remain after filtering

        Parameters
        ----------
        min_timeseries_length : int
            Minimum number of rows required per time-series to keep it; time-series with fewer
            rows than `min_timeseries_length` are dropped.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no time-series has enough rows to predict on.

        """
        counts = self.df[TIME_SERIES_GROUP_COLUMN_NAME].value_counts()
        dropped = counts[counts < min_timeseries_length].index.tolist()
        if dropped:
            logger.warning(
                f"The following groups were dropped because they have too few rows to predict on (<{min_timeseries_length}):"
            )
            for d in dropped:
                logger.warning(f"\t{d}")
            self.df = self.df[self.df[TIME_SERIES_GROUP_COLUMN_NAME].map(counts) >= min_timeseries_length]
        if self.df.empty:
            error = f"No timeseries in the input has enough rows to predict/train on (must have >= {min_timeseries_length} observations)."
            logger.error(error)
            raise ValueError(error)

    def extend_by_horizon(self, horizon: int) -> None:
        """
        Extend each time-series group by appending blank rows for the prediction horizon.

        For each group:
        - Determines the last timestamp
        - Generates `horizon` new timestamps at the inferred frequency
        - Preserves the group's location value
        - Creates empty rows (NaN for non-key columns)
        - Concatenates the extended group back into the dataset

        Parameters
        ----------
        horizon : int
            Number of future time steps to append per group.

        Returns
        -------
        None

        """
        frames = []
        for item_id, subdf in self.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME):
            last_ts = subdf[TIME_SERIES_TIMESTAMP_COLUMN_NAME].iloc[-1]

            # Create horizon new timestamps
            new_timestamps = [last_ts + self.freq_timedelta * (i + 1) for i in range(horizon)]

            blank = {
                TIME_SERIES_GROUP_COLUMN_NAME: item_id,
                TIME_SERIES_TIMESTAMP_COLUMN_NAME: new_timestamps,
            }

            blank = pd.DataFrame(blank)

            frames.append(pd.concat([subdf, blank], ignore_index=True))

        self.df = pd.concat(frames, ignore_index=True)

    def split_past_and_future_dfs(self, horizon: int) -> None:
        """
        Separate the final `horizon` rows of each group into `future_df`.

        For each `item_id` group:
          - Extract the last `horizon` rows as future data
          - Remove them from the main dataset
          - Remaining rows become the historical dataset

        Parameters
        ----------
        horizon : int
            Number of most recent rows per group to designate as future data.

        Returns
        -------
        None
            Updates `self.df` and `self.future_df` in place.

        """
        self.df = self.df.sort_values([TIME_SERIES_GROUP_COLUMN_NAME, TIME_SERIES_TIMESTAMP_COLUMN_NAME])
        remaining_frames = []
        future_frames = []

        for _item_id, subdf in self.df.groupby(TIME_SERIES_GROUP_COLUMN_NAME, group_keys=False):
            # Split this group's data
            future_part = subdf.tail(horizon)
            remaining_part = subdf.iloc[:-horizon] if len(subdf) > horizon else subdf.iloc[0:0]

            future_frames.append(future_part)
            remaining_frames.append(remaining_part)

        # Combine results
        remaining_df = pd.concat(remaining_frames, ignore_index=True)
        future_df = pd.concat(future_frames, ignore_index=True)

        self.df, self.future_df = remaining_df, future_df

    def add_temporal(self) -> None:
        """
        Add encoded temporal features (e.g., hour, day, month, cyclical encodings)
        to the dataset using `add_encoded_time_features`.

        Returns
        -------
        None
            Updates `self.df` and records added column names.

        """
        self.df, self.added_columns["temporal"] = add_encoded_time_features(self.df)

    def add_lags(self, lags: list[int], windows: list[int]) -> None:
        """
        Add lagged target values and rolling statistical features.

        This method:
        - Creates lag features for the forecasting target column
        - Creates rolling mean/median/std windows over past values
        - Finalizes the dataset with newly engineered lag features

        Parameters
        ----------
        lags : list[int], optional
            Lag offsets to compute (e.g., [1, 2, 24]).
        windows : list[int], optional
            Window lengths for rolling statistics.

        Returns
        -------
        None
            Updates `self.df` and records added column names.

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
        Merge holiday indicators into the dataset for all locations.

        Uses `merge_holidays_with_dataframe` to add binary/encoded holiday features

        Parameters
        ----------
        latlon : tuple[float, float]
            Latitude and longitude coordinates used for geocoding the places in the dataset.

        Returns
        -------
        None
            Updates `self.df` and records added column names.

        """
        if self.latlon is None:
            error = "Location information is required to add `holiday` features but is missing in the dataset."
            raise AttributeError(error)
        self.df, self.added_columns["holiday"] = merge_holidays_with_dataframe(
            self.df, self.latlon, TIME_SERIES_TIMESTAMP_COLUMN_NAME
        )

    def add_weather(self) -> None:
        """
        Enrich the dataset with weather features aligned to each timestamp and location.

        Weather is fetched or merged relative to:
        - The timestamp immediately after the final observed value
        - The dataset's frequency (hourly or daily)

        Parameters
        ----------
        None

        Returns
        -------
        None
            Updates `self.df` and records added column names.

        """
        if self.latlon is None:
            error = "Location information is required to add `weather` features but is missing in the dataset."
            raise AttributeError(error)

        self.df, self.added_columns["weather"] = enrich_weather(
            self.df,
            TIME_SERIES_TIMESTAMP_COLUMN_NAME,
            self.latlon,
            self.final_timestamp,
            self.freq,
        )
