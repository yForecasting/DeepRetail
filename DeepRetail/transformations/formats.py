import pandas as pd
from DeepRetail.transformations.decomposition import MSTL


def pivoted_df(df, target_frequency=None, agg_func=None, fill_values=True):
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

    if target_frequency is not None:
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
                    pivot_df.columns.min(),
                    pivot_df.columns.max(),
                    freq=target_frequency,
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
        df (pd.DataFrame): The dataframe in either pivot or transaction format
        format (str, optional): The format. Defaults to 'transaction'. Options: 'transaction', 'pivot'.

    Returns:
        pd.DataFrame: The converted dataframe.
    """

    # if we  have the transaction format
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
        # Dropping the name of the index
        df.index.name = None

        # Transpose and rename
        df = df.T
        df.index.names = ["Period"]

    return df


def statsforecast_forecast_format(
    df, format="transaction", fill_missing=False, fill_value="nan"
):
    """
    Converts a dataframe to the format required for forecasting with statsforecast.

    Args:
        df : pd.DataFrame
            The input data.
        format : str, default='transaction'
            The format of the input data. Can be 'transaction' or 'pivotted'.
        fill_missing : bool, default=False
            Whether to fill missing dates with NaN values. If True, the 'fill_value' argument
            must be specified.
        fill_value : str, int, float, default='nan'
            The value to use for filling missing dates. Default is 'nan'.

    Returns:
        df : pd.DataFrame
            The formatted dataframe.

    """

    # if we have transaction
    if format == "transaction":
        # just rename the date column to ds
        df = df.rename(columns={"date": "ds"})

        # fill missing dates if specified
        if fill_missing:
            df = fill_missing_dates(df, fill_value=fill_value)

    elif format == "pivotted":
        # if we have pivotted
        # we need to convert it to transaction
        df = transaction_df(df, drop_zeros=False)
        # and rename the date column to ds
        df = df.rename(columns={"date": "ds"})
    else:
        raise ValueError(
            "Provide the dataframe either in pivoted or transactional format."
        )

    # Return
    return df


def fill_missing_dates(df, fill_value="nan"):
    """
    Fills missing dates due to no sales.

    Args:
        df : pd.DataFrame
            The input data, expected to have at least 'ds' (date), 'y' (target variable),
            and 'unique_id' columns.
        fill_value: str, int
            The value to use for filling missing dates. Default is 'nan'.

    Returns:
        df : pd.DataFrame
            The formatted DataFrame with a continuous date range for each 'unique_id',
            filling missing dates with NaN values for 'y', and ensuring that the 'ds' column is
            of datetime type. The returned DataFrame is sorted by 'unique_id' and 'ds'.
    """

    # Identify the full date range in the dataset
    min_date, max_date = df["ds"].min(), df["ds"].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

    # Create a MultiIndex with all combinations of 'unique_id' and 'all_dates'
    unique_ids = df["unique_id"].unique()
    multi_index = pd.MultiIndex.from_product(
        [unique_ids, all_dates], names=["unique_id", "ds"]
    )

    # Reindex the DataFrame to include missing dates
    df_reindexed = df.set_index(["unique_id", "ds"]).reindex(multi_index).reset_index()

    # Sort by 'unique_id' and 'ds'
    df_reindexed = df_reindexed.sort_values(by=["unique_id", "ds"])

    if fill_value != "nan":
        df_reindexed = df_reindexed.fillna(fill_value)

    return df_reindexed


def extract_hierarchical_structure(
    df, current_format, correct_format, splitter, add_total=True, format="transaction"
):
    """
    Extract the hierarchical structure from the unique id of a dataframe.
    Returns a new dataframe with the hierarchical structure as columns.

    Args:
        df (pd.DataFrame): The dataframe with the unique_id column.
        current_format (list): Current names of the hierarchical levels in the unique_id
        correct_format (list): Names of the hierarchical levels on their correct order.
            Bottom levels are first, later levels last.
        splitter (str): The splitter used to split the unique_id column.
        add_total (bool): Whether or not to add a column for the total hiearhical level.
            Default is True and the name is "T".
        format (str): The format of the dataframe.
            Default is 'transaction'. Accepts transaction and pivoted.

    Returns:
        pd.DataFrame: The dataframe with the hierarchical structure as columns.

    Examples:
        >>> df = pd.DataFrame({'date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-01', '2022-01-02',
                            '2022-01-03'],
        ...                    'y': [1, 2, 3, 4, 5, 6],
        ...                    'unique_id': ['Hobbies_001_CA_1', 'Food_003_LA_1',
                        'Hobbies_002_CA_2', 'Food_002_CA_1', 'Food_001_LA_1', 'Hobbies_005_CA_2']})
        >>> extract_hierarchical_structure(df,
                                        ['category','item_num, 'state', 'store'],
                                        ['item_num', 'category', 'store', 'state'],
                                        '_',
                                        rue)

                                item_num            category         store     state total
        Hobbies_001_CA_1      T_CA_1_Hobbies_001  T_CA_1_Hobbies     T_CA_1    T_CA    T
        Food_003_LA_1         T_LA_1_Food_003     T_LA_1_Food        T_LA_1    T_LA    T
        Hobbies_002_CA_2      T_CA_2_Hobbies_002  T_CA_2_Hobbies     T_CA_2    T_CA    T
        Food_002_CA_1         T_CA_1_Food_001     T_CA_1_Food        T_CA_1    T_CA    T
        Food_001_LA_1         T_LA_1_Food_001     T_LA_1_Food        T_LA_1    T_LA    T
        Hobbies_005_CA_2      T_CA_2_Hobbies_005  T_CA_2_Hobbies     T_CA_2    T_CA    T

    """

    # Convert to the right format
    if format == "pivoted":
        # Convert to transaction format
        df = transaction_df(df)
    elif format == "transaction":
        pass
    else:
        raise ValueError("Format not recognized")

    # Drop duplicates on unique_id
    df = df.drop_duplicates(subset=["unique_id"])

    # drop the date and the y columns
    # We focus only on the format
    df = df.drop(columns=["date", "y"])

    # Split the unique_id column on the splitter
    df["temp"] = df["unique_id"].str.split(splitter)

    # set as index the unique_id
    df = df.set_index("unique_id")

    # Create a new column for every item on the list of the temp column
    df = df.join(pd.DataFrame(df["temp"].tolist(), index=df.index)).drop(
        columns=["temp"]
    )

    # Rename the columns based on the current format
    df.columns = current_format

    # Reorder the columns based on the correct format
    df = df[correct_format]

    # Step 8:
    # If total is True, add a total column
    if add_total:
        df["total"] = "T"

    # Starting from the end, for each column add the previous column as a sufix
    cols = df.columns

    # Itterate over the reversed columns and skipping the column with the total
    for i in range(len(df.columns) - 2, -1, -1):
        df.loc[:, cols[i]] = df.loc[:, cols[i + 1]] + "_" + df.loc[:, cols[i]]

    return df


def build_cross_sectional_df(df, hierarchical_df, format="pivoted"):
    """
    Extends a dataframe to include all hierarchical levels.
    Performs aggregations given the hierarchical dataframe.

    Args:
        df (pd.DataFrame): The original dataframe
        hierarchical_df (pd.DataFrame): A dataframe with the hierarchical structure.
            Its generated using the extract_hierarchical_structure function.

    Returns:
        pd.DataFrame: A new pivoted df that includes new time series for every hierarchical level.

    """

    # Convert to the right format
    if format == "pivoted":
        # Convert to transaction format
        df = transaction_df(df)
    elif format == "transaction":
        pass
    else:
        raise ValueError("Format not recognized")

    # Take the levels
    levels = hierarchical_df.columns

    # Merge with the hierarchical_df
    df = df.merge(hierarchical_df, left_on="unique_id", right_index=True, how="left")

    # Initialize a dataframe
    new_pivoted_df = pd.DataFrame()

    # Initialize the names of the columns
    new_cols = ["unique_id", "date", "y"]

    # Itterate over the levels
    for level in levels:
        # Groupby the level and sum
        temp_level = df.groupby([level, "date"]).agg({"y": "sum"}).reset_index()

        # Change the columns
        temp_level.columns = new_cols

        # pivot
        temp_level = pivoted_df(temp_level)

        # Concat with the new pivoted df
        new_pivoted_df = pd.concat([new_pivoted_df, temp_level], axis=0)

    return new_pivoted_df


def hierarchical_to_transaction(df, h_format, sort_by=True, format="pivoted"):
    """
    Extends the given dataframe to include the hierarchical format.

    Args:
        df (pd.DataFrame):
            The original dataframe.
        h_format (pd.DataFrame):
            The hierarchical schema.
            It is extracted using the extract_hierarchical_structure function
        sort_by (bool, optional):
            If True, the dataframe is sorted by the hierarchical format.
            Defaults to True.
            Lower levels are sorted first.
        format (str, optional):
            The format of the original dataframe.
            It can be either 'pivoted' or 'transactional'.
            Defaults to 'pivoted'.

    Returns:
        df (pd.DataFrame):
            The extended dataframe in transactional format.

    """

    # create a categorical data type with the true order
    # Used for sorting the dataframe
    cat_dtype = pd.api.types.CategoricalDtype(
        categories=h_format.columns.values, ordered=True
    )

    # Prepare the hierarchical format
    # Collapse the hierarchical format
    h_format = h_format.stack().reset_index()

    # Keep only level_1 and 0
    h_format = h_format[["level_1", 0]]

    # drop Duplicates
    h_format = h_format.drop_duplicates(subset=[0])

    # rename
    h_format = h_format.rename(
        columns={"level_1": "cross_sectional_level", 0: "unique_id"}
    )

    # Prepare the original df
    df = transaction_df(df) if format == "pivoted" else df

    # Merge on unique_id
    df = df.merge(h_format, on="unique_id", how="left")

    # Sort by level if true
    if sort_by:
        # convert the "levels" column to the categorical data type
        df["cross_sectional_level"] = df["cross_sectional_level"].astype(cat_dtype)

        # sort the dataframe based on the "levels" column
        df = df.sort_values(by="cross_sectional_level")

    # Return
    return df


def get_reminder(df, periods):
    """
    Decompose a time series and return the residuals.

    Args:
        df (pd.DataFrame):
            A DataFrame containing the time series data
        periods (list):
            A list of periods to use for the decomposition (e.g. [7, 30, 365])

    Returns:
        residuals (pd.DataFrame):
            A DataFrame containing the residuals of the decomposition
    """
    # Extract the values of the DataFrame
    vals = df.values

    # Use the MSTL function to decompose the time series and extract the residuals
    res_only = [MSTL(ts, periods=periods).fit().resid for ts in vals]

    return res_only


def MinMaxScaler_custom(x, feature_range=(0, 1)):
    """
    Performs MinMaxScaling on a numpy array.
    Follows the documentation from sklearn.

    Args:
        x (np.array):
            The array to be scaled.
        feature_range (tuple):
            The range of the output.

    Returns:
        np.array:
            The scaled array.

    References:
         https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """

    # Extract some values
    x_min, x_max = x.min(), x.max()
    feature_min, feature_max = feature_range

    # Perform scalling
    X_std = (x - x_min) / (x_max - x_min)
    X_scaled = X_std * (feature_max - feature_min) + feature_min

    return X_scaled


def StandardScaler_custom(x):
    """
    Performs StandardScaling on a numpy array.
    Follows the documentation from sklearn.

    Args:
        x (np.array):
            The array to be scaled.

    Returns:
        np.array:
            The scaled array.

    References:
         https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    """

    # Extract some values
    x_mean, x_std = x.mean(), x.std()

    # Perform scalling
    X_scaled = (x - x_mean) / x_std

    return X_scaled
