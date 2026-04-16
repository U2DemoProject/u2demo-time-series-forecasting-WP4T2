import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, SplineTransformer

from u2.constants.columns import TIME_SERIES_TIMESTAMP_COLUMN_NAME


def sin_transformer(period: int) -> FunctionTransformer:
    """
    Create a sine-based cyclical feature transformer.

    This transformer encodes cyclic variables (e.g., hour of day, month of year)
    using the function:

        sin( x / period * 2π )

    Parameters
    ----------
    period : int
        The period of the cycle (e.g., 24 for hours, 12 for months).

    Returns
    -------
    FunctionTransformer
        A transformer applying sine encoding element-wise to the input.

    Notes
    -----
    Based on cyclical feature encoding recommended by skforecast:
    https://skforecast.org/0.10.0/faq/cyclical-features-time-series.html#sine-cosine-encoding

    """
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period: int) -> FunctionTransformer:
    """
    Create a cosine-based cyclical feature transformer.

    This transformer encodes cyclic variables using the function:

        cos( x / period * 2π )

    Parameters
    ----------
    period : int
        The period of the cycle (e.g., 24 for hours, 365 for days).

    Returns
    -------
    FunctionTransformer
        A transformer applying cosine encoding element-wise to the input.

    Notes
    -----
    Based on cyclical feature encoding recommended by skforecast:
    https://skforecast.org/0.10.0/faq/cyclical-features-time-series.html#sine-cosine-encoding

    """
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def spline_transformer(period: int, degree: int = 3, extrapolation: str = "periodic") -> SplineTransformer:
    """
    Create a B-spline transformer for cyclical feature encoding.

    B-splines offer more flexibility than sine/cosine encodings and can model
    irregular cyclic patterns (e.g., day-of-year variations). Periodic
    extrapolation ensures smooth wrapping at cycle boundaries.

    Parameters
    ----------
    period : int
        Number of internal intervals (knots - 1), approximating the cycle length.
    degree : int, optional
        Degree of spline basis functions. Defaults to 3 (cubic).
    extrapolation : str, optional
        Extrapolation strategy outside knot range. Defaults to "periodic".

    Returns
    -------
    SplineTransformer
        A configured spline transformer with pandas output enabled.

    Notes
    -----
    Based on skforecast's guide on B-spline cyclical encoding:
    https://skforecast.org/0.10.0/faq/cyclical-features-time-series.html#b-splines-functions

    """
    return SplineTransformer(
        degree=degree, n_knots=period + 1, knots="uniform", extrapolation=extrapolation, include_bias=True
    ).set_output(transform="pandas")


def add_sin_cos_feature(df: pd.DataFrame, values: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    """
    Add sine and cosine encoded columns for a cyclical variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    values : pandas.Series
        Integer-valued cyclical variable (e.g., hours, months).
    period : int
        Period of the cycle.
    prefix : str
        Prefix for naming generated columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two new columns:
        - <prefix>_sin
        - <prefix>_cos

    """
    df = df.copy()
    df[f"{prefix}_sin"] = np.sin(values / period * 2 * np.pi)
    df[f"{prefix}_cos"] = np.cos(values / period * 2 * np.pi)
    return df


def add_day_of_week_onehot(df: pd.DataFrame, datetime_col: str = "timestamp") -> pd.DataFrame:
    """
    Add one-hot encoded day-of-week indicator columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a datetime column.
    datetime_col : str, optional
        Name of the datetime column. Defaults to "timestamp".

    Returns
    -------
    pandas.DataFrame
        DataFrame with seven new one-hot columns:
        day_of_week_0, ..., day_of_week_6

    """
    df = df.copy()
    # Use Categorical with all 7 days so get_dummies always produces day_of_week_0..6
    df["day_of_week"] = pd.Categorical(df[datetime_col].dt.dayofweek, categories=list(range(7)))
    return pd.get_dummies(df, columns=["day_of_week"], prefix="day_of_week")


def add_day_of_year_spline(df: pd.DataFrame, period: int, datetime_col: str = "timestamp") -> pd.DataFrame:
    """
    Add B-spline encoded day-of-year features.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the datetime column.
    period : int
        Number of internal spline intervals (periodicity of the cycle).
    datetime_col : str, optional
        Name of the datetime column. Defaults to "timestamp".

    Returns
    -------
    pandas.DataFrame
        DataFrame with spline basis columns named:
        day_of_year_spline_<i>

    """
    df = df.copy()
    doy = df[datetime_col].dt.dayofyear.to_numpy().reshape(-1, 1)
    transformer = spline_transformer(period=period)
    doy_spline = transformer.fit_transform(doy)
    doy_spline.columns = [f"day_of_year_spline_{i}" for i in range(doy_spline.shape[1])]
    return pd.concat([df, doy_spline], axis=1)


def add_encoded_time_features(
    df: pd.DataFrame, datetime_col: str = TIME_SERIES_TIMESTAMP_COLUMN_NAME
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add multiple time-derived features to a DataFrame.

    Features added include:
    - One-hot day-of-week indicators
    - B-spline day-of-year basis functions
    - Sine/cosine encodings for month
    - Sine/cosine encodings for hour (only if frequency ≤ 1 hour)

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a datetime column.
    datetime_col : str, optional
        Name of the datetime column. Defaults to TIME_SERIES_TIMESTAMP_COLUMN_NAME.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        The transformed DataFrame and a list of column names added.

    """
    new_columns = []
    feature_steps = [
        lambda df: add_day_of_week_onehot(df, datetime_col),
        lambda df: add_day_of_year_spline(df, period=12, datetime_col=datetime_col),
        lambda df: add_sin_cos_feature(df, df[datetime_col].dt.month, period=12, prefix="month"),
    ]

    if df[datetime_col].diff().mode()[0] <= pd.Timedelta("1 hour"):
        feature_steps.append(lambda df: add_sin_cos_feature(df, df[datetime_col].dt.hour, period=24, prefix="hour"))

    for step in feature_steps:
        old_cols = set(df.columns)
        df = step(df)
        added_cols = list(set(df.columns) - old_cols)
        new_columns.extend(added_cols)

    return df, new_columns
