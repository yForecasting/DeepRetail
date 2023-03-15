import pandas as pd
import numpy as np
import gc
import datetime
# import re


def read_case_0(read_filepath, calendar_filepath):
    """Reads data for case 0

    Args:
        read_filepath (str): Existing location of the data file.
        calendar_filepath (str): Existing location of the calendar file.
            Required for reading.

    Returns:
        pandas.DataFrame: A dataframe with the loaded data.

    Example usage:
    >>> df = read_case_0('data.csv', 'calendar.csv')

    """

    # read the data file and the calendar
    df = pd.read_csv(read_filepath)
    calendar = pd.read_csv(calendar_filepath)

    # Drop some columns
    # Hierarchy is defined as:
    # State -> Store -> Category -> Department -> Item
    to_drop = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    df = df.drop(to_drop, axis=1)

    # Modify the id and set it as index
    df["id"] = ["_".join(d.split("_")[:-1]) for d in df["id"].values]
    df = df.rename(columns={"id": "unique_id"})
    df = df.set_index("unique_id")

    # Prepare the dates from the calendar
    dates = calendar[["d", "date"]]

    # find the total days
    total_days = df.shape[1]
    dates = dates.iloc[:total_days]["date"].values

    # Replace on the columns
    df.columns = dates

    # Convert to datetime
    df.columns = pd.to_datetime(df.columns)

    # drop columns with only zeros
    df = df.loc[~(df == 0).all(axis=1)]

    return df


def read_case_1(read_filepath, write_filepath, frequency, temporary_save):
    """Reads data for case 1

    Args:
        read_filepath (str): Existing loocation of the data file
        write_filepath (str): Location to save the new file
        frequency (str): The selected frequency.
                    Note: Due to size issues, for case 1 only supports W and M
        temporary_save (bool, Optional): If true it saves the dataframe on chunks
                                         Deals with memory breaks.
    """
    # Initialize parameters to ensure stable loading

    chunksize = 10**6
    dict_dtypes = {
        "CENTRALE": "category",
        "FILIAAL": "category",
        "ALDIARTNR": "category",
        "ST": np.float32,
        "VRD": np.float16,
    }

    # Initialize the reading itterator
    tp = pd.read_csv(
        read_filepath,
        iterator=True,
        chunksize=chunksize,
        sep=";",
        dtype=dict_dtypes,
        parse_dates=["DATUM"],
        infer_datetime_format=True,
        decimal=",",
    )

    # Drop stock column for now
    df = pd.concat(tp, ignore_index=True).drop("VRD", axis=1)

    # Name the columns
    cols = ["DC", "Shop", "Item", "date", "y"]
    df.columns = cols

    # Delete the itterator to release some memory
    del tp
    gc.collect()
    # Main loading idea!
    # Process the df in chunks: -> At each chunk sample to the given frequency
    # Then concat!

    # Initialize chunk size based on the frequency
    if frequency == "W":
        chunk_size = 14
    elif frequency == "M":
        chunk_size = 59
    else:
        raise ValueError(
            "Currently supporting only Weekly(W) and Monthly(M) frequencies for case 1"
        )
    # Initialize values for the chunks
    start_date = df["date"].min()
    chunk_period = datetime.timedelta(days=chunk_size)
    temp_date = start_date + chunk_period

    # Initialize the df on the first chunk!
    out_df = df[(df["date"] < temp_date) & (df["date"] > start_date)]
    start_date = temp_date - datetime.timedelta(days=1)

    # Initialize the names on the unique_id
    # Lower level is the product-level
    out_df = out_df.drop(["DC", "Shop"], axis=1)
    out_df = out_df.rename(columns={"Item": "unique_id"})
    # Pivot and resample to the given frequency
    out_df = (
        pd.pivot_table(
            out_df, index="unique_id", columns="date", values="y", aggfunc="sum"
        )
        .resample(frequency, axis=1)
        .sum()
    )
    # Itterate over the other chunks:
    while start_date + chunk_period < df["date"].max():
        # Update the date
        temp_date = start_date + chunk_period

        # Filter on the given period
        temp_df = df[(df["date"] < temp_date) & (df["date"] > start_date)]
        start_date = temp_date - datetime.timedelta(days=1)

        # Update names on the unique_id, drop columns, pivot & resample
        temp_df = temp_df.drop(["DC", "Shop"], axis=1)
        temp_df = temp_df.rename(columns={"Item": "unique_id"})

        temp_df = (
            pd.pivot_table(
                temp_df, index="unique_id", columns="date", values="y", aggfunc="sum"
            )
            .resample(frequency, axis=1)
            .sum()
        )

        # Add to the main df
        out_df = pd.concat([out_df, temp_df], axis=1)

        # Save at each itteration to deal with memory breaks
        if temporary_save:
            out_df.to_csv(write_filepath)
    # Final save
    # out_df = fix_duplicate_cols(out_df)
    return out_df
