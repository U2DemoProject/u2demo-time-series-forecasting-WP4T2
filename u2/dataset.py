import datetime as dt
import os
from typing import Self

import pandas as pd
from joblib import Memory

from .defaults import (
    DEFAULT_LAG_WINDOWS,
    DEFAULT_LAGS,
    DEFAULT_LOCATION_COLUMN_NAME,
    DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME,
    DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
    DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME,
)
from .features.holidays import merge_holidays_for_multiple_places
from .features.lag import LagFeatureGenerator
from .features.temporal import add_encoded_time_features
from .features.weather import enrich_weather
from .logging_config import logger

memory = Memory(location=os.getenv("CACHE_DIR", ".cache"), verbose=0)


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

    """

    def __init__(self):
        self.df = None
        self.future_df = None
        self.added_columns = {}
        self.freq = None
        self.freq_timedelta = None
        self.final_timestamp = None

    @classmethod
    def from_csv(
        cls,
        time_series_csv: str,
        min_num_rows_per_group: int,
        location_col_name: str = DEFAULT_LOCATION_COLUMN_NAME,
        timestamp_col_name: str = DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME,
        predict_value_col_name: str = DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME,
        group_col_name: str = DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME,
        *args,
        **kwargs,
    ) -> Self:
        """
        Load a multigroup time-series dataset from CSV and validate its structure.

        This method:
        - Reads the CSV into a DataFrame
        - Ensures required columns (timestamp, value, location) exist
        - Creates a default group column if missing
        - Drops groups with fewer than `min_num_rows_per_group` rows
        - Verifies that all groups share the same final timestamp
        - Ensures all groups have a consistent inferred frequency ("h" or "D")
        - Normalizes column names to U2 defaults
        - Sorts rows for proper sequential processing

        Parameters
        ----------
        time_series_csv : str
            Path to the CSV file.
        min_num_rows_per_group : int
            Minimum number of rows required per group to be included.
        location_col_name : str, optional
            Column name for the location identifier.
        timestamp_col_name : str, optional
            Column name for timestamps.
        predict_value_col_name : str, optional
            Column name for the target value.
        group_col_name : str, optional
            Column name for the grouping (item_id).
        *args, **kwargs :
            Additional arguments passed to `pandas.read_csv`.

        Returns
        -------
        U2TimeSeriesPredictionDataset
            A fully initialized dataset object.

        Raises
        ------
        AttributeError
            If required columns are missing.
        ValueError
            If all groups are dropped or group frequencies/final timestamps differ.

        """
        obj = cls()
        obj.df = pd.read_csv(time_series_csv, *args, **kwargs)

        # Force timestamp column to datetime type
        obj.df[timestamp_col_name] = pd.to_datetime(obj.df[timestamp_col_name])

        if (
            timestamp_col_name not in obj.df.columns
            or predict_value_col_name not in obj.df.columns
            or location_col_name not in obj.df.columns
        ):
            error = f"One of `{timestamp_col_name}` and/or `{predict_value_col_name}` and/or `{location_col_name}` columns are missing`."
            raise AttributeError(error)

        if group_col_name not in obj.df.columns:
            logger.warning(
                f"`{group_col_name}` column is not present in time series CSV `{time_series_csv}`, assuming all values belong to a single group."
            )
            obj.df[group_col_name] = 0

        # Drop groups with too few rows
        counts = obj.df[group_col_name].value_counts()
        dropped = counts[counts < min_num_rows_per_group].index.tolist()
        if dropped:
            logger.warning(
                f"The following groups were dropped because they have too few rows to predict on (<{min_num_rows_per_group}):"
            )
            for d in dropped:
                logger.warning(f"\t{d}")
            obj.df = obj.df[obj.df[group_col_name].map(counts) >= min_num_rows_per_group]
        if obj.df.empty:
            error = "No group in the input CSV has enough rows to predict on."
            logger.error(error)
            raise ValueError(error)

        # All groups must have the same final timestamp
        final_timestamps = set(obj.df.groupby(group_col_name)[timestamp_col_name].max().to_list())
        if len(final_timestamps) > 1:
            error = "Not all groups have the same final timestamp. Found values:"
            for t in set(final_timestamps):
                error += f"\n\t{t}"
            logger.error(error)
            raise ValueError(error)
        obj.final_timestamp = next(iter(final_timestamps))

        # Get the frequency
        # It should ALWAYS be "h" or "D", because the model is only ever trained on those two
        # It must also be the same for all groups
        group_freqs = (
            obj.df.sort_values([group_col_name, timestamp_col_name])
            .groupby(group_col_name)[timestamp_col_name]
            .apply(pd.infer_freq)
        )
        unique_freqs = group_freqs.dropna().unique()
        if len(unique_freqs) != 1:
            error = f"Inconsistent inferred frequencies across groups: {group_freqs}"
            logger.error(error)
            raise ValueError(error)
        obj.freq = unique_freqs[0]
        obj.freq_timedelta = {"h": dt.timedelta(hours=1), "D": dt.timedelta(days=1)}[obj.freq]

        # Rename columns if necessary
        if timestamp_col_name != DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME:
            obj.df = obj.df.rename(columns={timestamp_col_name: DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME})
        if predict_value_col_name != DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME:
            obj.df = obj.df.rename(columns={predict_value_col_name: DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME})
        if group_col_name != DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME:
            obj.df = obj.df.rename(columns={group_col_name: DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME})
        if location_col_name != DEFAULT_LOCATION_COLUMN_NAME:
            obj.df = obj.df.rename(columns={location_col_name: DEFAULT_LOCATION_COLUMN_NAME})

        obj.df = obj.df.sort_values([DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME, DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME])

        return obj

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
        for item_id, subdf in self.df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME):
            last_ts = subdf[DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME].iloc[-1]

            # Each group keeps its own location
            location_value = subdf[DEFAULT_LOCATION_COLUMN_NAME].iloc[0]

            # Create horizon new timestamps
            new_timestamps = [last_ts + self.freq_timedelta * (i + 1) for i in range(horizon)]

            blank = pd.DataFrame(
                {
                    DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME: item_id,
                    DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME: new_timestamps,
                    DEFAULT_LOCATION_COLUMN_NAME: location_value,
                }
            )

            # Other columns should remain blank → add them as NaN automatically
            missing_cols = [c for c in self.df.columns if c not in blank.columns]
            for col in missing_cols:
                blank[col] = pd.NA

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
        self.df = self.df.sort_values(
            [DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME, DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME]
        )
        remaining_frames = []
        future_frames = []

        for _item_id, subdf in self.df.groupby(DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME, group_keys=False):
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

    def add_lags(self, lags: list[int] = DEFAULT_LAGS, windows: list[int] = DEFAULT_LAG_WINDOWS) -> None:
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
        lag_gen = LagFeatureGenerator(self.df, DEFAULT_TIME_SERIES_GROUP_COLUMN_NAME)
        # Up to now lags have only ever been calculated on the value column
        target_columns = [DEFAULT_TIME_SERIES_PREDICT_VALUE_COLUMN_NAME]
        for col in target_columns:
            lag_gen.create_lag_features(col, lags=lags)
            lag_gen.create_rolling_stat_features(col, windows=windows)
        self.df, self.added_columns["lag"] = lag_gen.finalize()

    def add_holidays(self) -> None:
        """
        Merge holiday indicators into the dataset for all locations.

        Uses `merge_holidays_for_multiple_places` to:
        - Identify holidays for each location
        - Add binary/encoded holiday features

        Returns
        -------
        None
            Updates `self.df` and records added column names.

        """
        self.df, self.added_columns["holiday"] = merge_holidays_for_multiple_places(
            self.df, DEFAULT_LOCATION_COLUMN_NAME, DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME
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
        present_time = self.final_timestamp + self.freq_timedelta
        self.df, self.added_columns["weather"] = enrich_weather(
            self.df, DEFAULT_TIME_SERIES_TIMESTAMP_COLUMN_NAME, DEFAULT_LOCATION_COLUMN_NAME, present_time, self.freq
        )
