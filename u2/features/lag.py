import pandas as pd


class LagFeatureGenerator:
    """
    Generate lag and rolling statistical features for grouped time-series data.

    This class operates on a copy of the provided DataFrame and tracks all newly
    added feature columns. It supports generating:
    - Lag features (e.g., t-1, t-24, ...)
    - Rolling statistics (mean, std, min, max) based on shifted windows

    Parameters
    ----------
    df : pandas.DataFrame
        Input time-series DataFrame.
    group_name : str
        Column name indicating the timeseries grouping (e.g., item_id).

    Attributes
    ----------
    df : pandas.DataFrame
        Working copy of the input DataFrame with added features.
    group_name : str
        Grouping column for computing lag/rolling features.
    _max_shift : int
        Maximum number of rows shifted for lag/rolling features; used for trimming.
    added_columns : list[str]
        Names of all feature columns added during processing.

    """

    def __init__(self, df: pd.DataFrame, group_name: str) -> None:
        """
        Initialize the feature generator with a DataFrame and group column.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame containing time-series data.
        group_name : str
            Name of the column identifying independent time-series groups.

        """
        self.df = df.copy()
        self.group_name = group_name
        self._max_shift = 0
        self.added_columns = []

    def create_lag_features(self, column: str, lags: list[int]) -> pd.DataFrame:
        """
        Create lag features for a given column within each timeseries group.

        For each value in `lags`, a new column `lag_<lag>` is created containing
        values shifted downward by that lag offset.

        Parameters
        ----------
        column : str
            Name of the column for which lag features should be computed.
        lags : list[int]
            List of integer lag steps (e.g., [1, 24, 48]).

        Returns
        -------
        pandas.DataFrame
            The updated DataFrame with newly added lag features.

        Notes
        -----
        Updates the internal `_max_shift` to ensure proper trimming in `finalize()`.

        """
        for lag in lags:
            lag_column_name = f"lag_{lag}"
            self.df[lag_column_name] = self.df.groupby(self.group_name)[column].shift(lag)
            self.added_columns.append(lag_column_name)

        self._max_shift = max(self._max_shift, int(max(lags)))

    def create_rolling_stat_features(self, column: str, windows: list, statistics: list[str] = None) -> pd.DataFrame:
        """
        Create rolling statistical features (mean, std, min, max) for a column.

        Rolling features are computed on shifted data (t-1, t-2, ..., t-w) to avoid
        data leakage. For each window size and each requested statistic, a new
        feature named `roll_<stat>_<window>` is produced.

        Parameters
        ----------
        column : str
            Column name on which to compute rolling statistics.
        windows : list[int]
            List of rolling window sizes.
        statistics : list[str], optional
            List of statistics to compute. Defaults to ["mean", "std", "min", "max"].

        Returns
        -------
        pandas.DataFrame
            The updated DataFrame with added rolling feature columns.

        Raises
        ------
        ValueError
            If unsupported statistics are requested.

        Notes
        -----
        - Only implemented statistics ("mean", "std", "min", "max") are allowed.
        - Updates `_max_shift` based on the largest window size.

        """
        implemented_statistics = ["mean", "std", "min", "max"]

        if not statistics:
            statistics = implemented_statistics

        unsupported_statistics = set(statistics).difference(implemented_statistics)
        if unsupported_statistics:
            raise ValueError(
                f"Unsupported statistics: {', '.join(unsupported_statistics)}. Supported: {implemented_statistics}"
            )
        if windows != []:
            for w in windows:
                grouped = self.df.groupby(self.group_name)[column]
                roll = grouped.shift(1).rolling(window=w)
                for stat in statistics:
                    if hasattr(roll, stat):
                        col_name = f"roll_{stat}_{w}"
                        self.df[col_name] = getattr(roll, stat)()
                        self.added_columns.append(col_name)

            self._max_shift = max(self._max_shift, int(max(windows)))

    def finalize(self) -> tuple[pd.DataFrame, list[str]]:
        """
        Finalize and return the processed DataFrame and added feature column names.

        This method trims the first `_max_shift` rows of each group to ensure that
        all lag and rolling feature values are valid (i.e., no NaNs introduced by
        shifting or rolling).

        Returns
        -------
        tuple[pandas.DataFrame, list[str]]
            A tuple containing:
            - The trimmed DataFrame with all generated features
            - A list of the added feature column names

        """
        return (
            self.df.groupby(self.group_name)
            .apply(lambda g: g.iloc[self._max_shift :], include_groups=True)
            .reset_index(drop=True)
        ), self.added_columns
