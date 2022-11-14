# Functions converting pivoted to transcation dataframes!


import pandas as pd
from collections import Counter
import numpy as np


def pivoted_df(df, target_frequency, agg_func=None, fill_values=True):
    """Converts a transaction df to a pivoted df.
    Each row is a unique id and columns are the dates.
    Missing values are filled with zeros by default.
    Time series can be resampled to different frequencies

    Args:
        df (pd.DataFrame): A transaction dataframe. Requires the following columns:
                          date: The dates
                          y: Sales of the given date
                          unique_id: IDs for each sales location
        target_frequency (str): Targeted frequency. Resample through aggregations
        agg_func (str): The aggregations function. Supports: sum, constant, None.
                        Usage: sum for sales date and constant for stock data.
                        Default is None for no aggregations
        fill_values (bool, optional): If to add zeros on mising values.
                                    Missing values are dates with no sales.
                                    Defaults to True.
    """

    # Ensure dates are on the right format
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


def transaction_df(df, keep_zeros=False):
    """Converts a pivoted df to a transaction df. A transaction df has 3 columns:
        - unique_id: Sales location of each time series.
        - date: The date
        - y: The value for the time series

    Args:
        df (pd.DataFrame): The pivoted Dataframe.
                        Each row is a time series and columns are the dates.
        keep_zeros (bool, optional): If to keep periods with zero sales.
                                    Defaults to False.
    """

    # resets the index
    trans_df = df.reset_index(names="unique_id")

    # Melts
    trans_df = pd.melt(trans_df, id_vars="unique_id", value_name="y", var_name="date")

    # Filters zeros if keep_zeros is set to True
    if keep_zeros:
        trans_df = trans_df[trans_df["y"] != 0]

    return trans_df


def fix_duplicate_cols(df):
    """Fixes an issue with duplicate columns on chunk breas

    Args:
        df (pd.DataFrame): The dataframe to make the fixes
    """

    # Get the duplicate columns
    duplicate_cols = [item for item, count in Counter(df.columns).items() if count > 1]

    # Replace for every dup col
    for col in duplicate_cols:
        # Sum the duplicates
        temp = df[col]
        temp = temp.sum(axis=1)

        # drop the old columns
        df = df.drop(col, axis=1)
        # Add the sum as the new col
        df[col] = temp

    # Convert to dt
    df.columns = pd.to_datetime(df.columns)
    df.columns = sorted(df.columns)  # sort
    return df


# Removes some items
def remove_gifts(p, q):
    """A function to compare values on case 6

    Args:
        p (float): Value 1 to compare
        q (float): value 2 to compare

    Returns:
        int: A flag 1 or 2 on weather to keep or not a specific column
    """
    if (q > 0) & (p == 0):
        return 1
    else:
        return 0


def forecast_format(df, format="transaction"):
    """Converts a df to the required format for sktime's AutoETS

    Args:
        df (pd.DataFrame): The dataframe in either pivot or transcation format
        format (str, optional): The format. Defaults to 'transaction'.

    Returns:
        pd.DataFrame: The converted dataframe.
    """

    # if we have the transaction format
    if format == "transaction":

        # rename and pivot
        df = df.rename(columns={"date": "Period"})
        df = pd.pivot_table(
            df, index="Period", columns="unique_id", values="y", aggfunc="first"
        )

        # Drop the name on the columns
        df.columns.name = None

    else:
        # Droping the name of the index
        df.index.name = None

        # Transpose and rename
        df.T.rename_axis("Period").head()

    return df
