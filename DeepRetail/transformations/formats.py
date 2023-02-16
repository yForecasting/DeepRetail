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
        >>> df = pd.DataFrame({'date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-01', '2022-01-02',
                            '2022-01-03'],
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


def transaction_df(df, drop_zeros=False):
    """
    Converts a pivoted df to a transaction df. A transaction df has 3 columns:
    - unique_id: Sales location of each time series.
    - date: The date.
    - y: The value for the time series.

    Args:
        df (pd.DataFrame): The pivoted DataFrame with time series as rows and dates as columns.
        drop_zeros (bool): Whether or not to drop periods with zero sales. Default: False.

    Returns:
        pd.DataFrame: A transaction DataFrame.

    Examples:
        >>> df = pd.DataFrame({'unique_id': ['A', 'A', 'B', 'B'], '2022-01-01': [1, 2, 0, 4],
                '2022-01-02': [0, 5, 6, 0]})
        >>> transaction_df(df)
        unique_id        date  y
        0         A  2022-01-01  1
        1         A  2022-01-01  2
        2         B  2022-01-02  6
        3         B  2022-01-01  4
        >>> transaction_df(df, drop_zeros=True)
        unique_id
    """

    # resets the index
    trans_df = df.reset_index(names="unique_id")

    # Melts
    trans_df = pd.melt(trans_df, id_vars="unique_id", value_name="y", var_name="date")

    # Filters zeros if keep_zeros is set to True
    if drop_zeros:
        trans_df = trans_df[trans_df["y"] != 0]

    return trans_df


def sktime_forecast_format(df, format="transaction"):
    """Converts a dataframe to the format required by sktime for forecasting.

    Args:
        df (pd.DataFrame): The dataframe in either pivot or transcation format
        format (str, optional): The format. Defaults to 'transaction'. Options: 'transaction', 'pivot'.

    Returns:
        pd.DataFrame: The converted dataframe.
    """

    # if we have the transaction format
    if format == "transaction":
        # rename and pivot
        df = df.rename(columns={"date": "Period"})
        df = pd.pivot_table(
            df,
            index="Period",
            columns="unique_id",
            values="y",
            aggfunc="first",
        )

        # Drop the name on the columns
        df.columns.name = None
    else:
        # Droping the name of the index
        df.index.name = None

        # Transpose and rename
        df.T.rename_axis("Period").head()

    return df
