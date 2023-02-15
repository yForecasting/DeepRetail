import pandas as pd


def pivoted_df(df, target_frequency, agg_func=None, fill_values=True):
    """
    Converts a transaction df to a pivoted df.
    Each row is a unique id and columns are the dates.
    Missing values are filled with zeros by default.
    Time series can be resampled to different frequencies.

    Args:
        df (pd.DataFrame): A transaction DataFrame with columns 'date', 'y', and 'unique_id'.
        target_frequency (str): Target frequency for resampling. Ex: 'D' for daily, 'W' for weekly.
        agg_func (str): The aggregation function. Options: 'sum', 'constant', None. Default: None.
        fill_values (bool): Whether or not to fill missing values with zeros. Default: True.

    Returns:
        pd.DataFrame: A pivoted DataFrame.

    Examples:
        >>> df = pd.DataFrame({'date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-01', '2022-01-02', '2022-01-03'],
        ...                    'y': [1, 2, 3, 4, 5, 6],
        ...                    'unique_id': ['A', 'A', 'A', 'B', 'B', 'B']})
        >>> pivoted_df(df, 'D', 'sum')
                    2022-01-01  2022-01-02  2022-01-03
        unique_id
        A                   1           2           3
        B                   4           5           6
    """

    # Ensure dates are on the right formatI
    df["date"] = pd.to_datetime(df["date"])

    # Pivots on the original frequency
    pivot_df = pd.pivot_table(
        df, index="unique_id", columns="date", values="y", aggfunc="first"
    )

    # Drop values with full nans
    pivot_df = pivot_df.dropna(axis=0, how="all")

    # Resamples with the given function
    # for sales data
    if agg_func == "sum":
        pivot_df = pivot_df.resample(target_frequency, axis=1).sum()
    # for stock data
    elif agg_func == "constant":
        pivot_df = pivot_df.resample(target_frequency, axis=1).last()

    # Fills missing values
    if fill_values:
        pivot_df = pivot_df.reindex(
            columns=pd.date_range(
                pivot_df.columns.min(), pivot_df.columns.max(), freq=target_frequency
            )
        )
    return pivot_df


