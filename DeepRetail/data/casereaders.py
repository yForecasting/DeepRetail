import pandas as pd


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
