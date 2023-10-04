from pandas.tseries.frequencies import to_offset
import numpy as np
import pandas as pd


def get_numeric_frequency(freq):
    """
    Return the frequency of a time series in numeric format.

    The function returns the frequency of a time series in numeric format. This is useful when working with
    forecasting libraries that require the frequency to be a number instead of a string.

    If frequency has multiple seasonalities, for example Daily and Hourly, returns a list with all periods.

    Args:
        freq (str): A string specifying the frequency of the time series.
        Valid values are:
        'Y' (yearly), 'A' (annually), 'Q' (quarterly), 'M' (monthly), 'W' (weekly), 'D' (daily), or 'H' (hourly).

    Returns:
        int: The frequency of the time series in numeric format if frequency has only one seasonalities.
        list: A list with all periods if frequency has multiple seasonalities.

    References:
        - https://otexts.com/fpp3/tsibbles.html

    Example:
        >>> get_numeric_frequency('M')
        1

        >>> get_numeric_frequency('W')
        13

        >>> get_numeric_frequency('D')
        365
    """

    keys = ["Y", "A", "Q", "M", "W", "D", "H"]
    vals = [1, 1, 4, 12, 52, [7, 30, 364], [24, 168, 720, 8760]]

    freq_dictionary = dict(zip(keys, vals))

    # Getting the period and the frequency
    period = to_offset(freq).n

    # Taking the first letter of the frequency in case we have MS for month start etc
    freq = to_offset(freq).name[0]

    # Initializing the dictionary
    numeric_freq = freq_dictionary[freq]

    # Dividing with the period:
    # For example if I have a 2M frequency:
    # Then instead of 12 months we have 6 examina
    numeric_freq = (
        int(freq_dictionary[freq] / period)
        if isinstance(numeric_freq, int)
        else [int(i / period) for i in numeric_freq]
    )

    return numeric_freq


# A functions that extends input features to the extended format.
def add_missing_values(input_features, input_transformations=None):
    """
    Fills the features and transformations dictionaries with default values.
    Default values are Nones and nans.

    Args:
        input_features (dict):
            A dictionary containing the features to be used in the model.
        input_transformations (dict):
            A dictionary containing the transformations to be used in the model.

    Returns:
        input_features: (dict):
            A dictionary containing the features to be used in the model.
        input_transformations: (dict):
            A dictionary containing the transformations to be used in the model.
    """

    # Default values for features and transformations dictionaries
    # Initialize a dictionary for transformations if it is none
    input_transformations = {} if input_transformations is None else input_transformations

    features = {
        "lags": None,
        "rolling_features": None,
        "rolling_lags": None,
        "seasonal_features": None,
        "fourier_terms": None,
        "positional_features": False,
        "time_series_id": False,
        "level_information": None,
    }
    transformations = {
        "stationarity": False,
        "logarithm": False,
        "normalize": None,
        "custom_no_reverse_1": None,
        "custom_no_reverse_2": None,
        "custom_no_reverse_3": None,
        "custom_reverse_1": [None, None],
        "custom_reverse_2": [None, None],
        "custom_reverse_3": [None, None],
    }

    # Check if each key in default features exists in input_features,
    # if not, add it with value equal to None
    for key in features.keys():
        if key not in input_features.keys():
            input_features[key] = features[key]

    # Check if each key in default transformations exists in input_transformations,
    # if not, add it with value equal to None
    for key in transformations.keys():
        if key not in input_transformations.keys():
            input_transformations[key] = transformations[key]

    return input_features, input_transformations


def augment(ts, window_size):
    """
    Augments the time series data by creating windows of size `window_size` days.
    If the length of the series is less than the window size, it pads the series with zeros.

    Args:
        ts (np.array):
            time series data
        window_size (int):
            size of the windows in days

    Returns:
        view (np.array):
            augmented time series data
    """
    total_length = len(ts)

    # If the length of the series is less than the window size, add padding with NaN values
    if total_length < window_size:
        zeros_to_add = window_size - total_length
        # Pad the series with NaN values
        view = np.pad(ts, pad_width=(zeros_to_add, 0), constant_values=np.nan)
        # Reshape the series to a 2D array
        view = view.reshape(1, -1)
    else:
        # Use the windowed function
        view = window(ts, window_size)

    return view


def window(a, window_size):
    """
    Create windows of size `window_size` from the 1D array `a`.

    Args:
        a (np.array):
            1D array
        window_size (int):
            size of the windows

    Returns:
        view (np.array):
            2D array of windows
    """
    # Convert window size to int
    w = int(window_size)
    # Calculate the shape of the windowed array
    sh = (a.size - w + 1, w)
    # Calculate the strides for the windowed array
    st = a.strides * 2
    # Create the windowed array using as_strided method
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::1]  # The step size is 1, i.e. no overlapping

    # Discard windows with all zeros
    view = view[~np.all(view == 0, axis=1)]

    return view


def create_lags(df, lags):
    """
    Creates the lagged dataframe for all time series on the input dataframe.

    Args:
        df (pd.DataFrame):
            A dataframe containing the time series data.
        lags (list):
            A list containing the lags to be used.

    Returns:
        lags_df (pd.DataFrame):
            A dataframe containing the lagged time series data.


    """

    lags_df = df.apply(lambda x: augment(x.values, lags).squeeze(), axis=1).to_frame(name="lag_windows")

    return lags_df


def construct_single_rolling_feature(df, rolling_aggregation, original_lags, rolling_windows, rolling_lags=1):
    # Check if rolling_window is integer and convert to list
    rolling_windows = [rolling_windows] if isinstance(rolling_windows, int) else rolling_windows

    # In case rolling_lags has a single value, repeat it for the number of rolling windows
    rolling_lags = np.repeat(rolling_lags, len(rolling_windows)) if isinstance(rolling_lags, int) else rolling_lags

    # Initialize a dataframe to include all rolling features
    rolling_df = pd.DataFrame()

    # for every rolling window itterate
    for window, temp_lag in zip(rolling_windows, rolling_lags):
        # Create the name of the rolling feature
        name = f"rolling_{rolling_aggregation}_{window}"
        # construct the rolling features for the time series
        temp_df = df.rolling(window, axis=1).agg(rolling_aggregation)
        # Slice them into lags -> using the starting number of lags as the original lags
        # Also drop the lag_windows column
        temp_df = create_lags(temp_df, original_lags)
        # Keep only the specified amount and round the subwindow
        # temp_df['lag_windows'] = [subwindows[:, -temp_lag:] for subwindows in temp_df['lag_windows'].values]
        temp_df["lag_windows"] = [subwindows[:, -temp_lag:].round(3) for subwindows in temp_df["lag_windows"].values]
        # rename
        temp_df = temp_df.rename(columns={"lag_windows": name})

        # Append to the main df
        rolling_df = pd.concat([rolling_df, temp_df], axis=1)

    # return]
    return rolling_df


def split_lag_targets(df, test_size=1):
    """
    Splits the lagged dataframe into targets and lagged values.

    Args:
        df (pd.DataFrame):
            A dataframe containing the lagged time series data.
        test_size (int):
            The number of windows to be used for testing.

    Returns:
        df (pd.DataFrame):
            A dataframe containing the lagged time series data with the targets and lagged values.

    """

    # dimension of targets: (windows, 1)
    # dimension of lags: (windows, lags)

    # Fix an issue for when we have just a single lag
    if len(df["lag_windows"].values[0].shape) == 1:
        # reshape all the lag windows
        df["lag_windows"] = df["lag_windows"].apply(lambda x: x.reshape(1, -1))

    # targets are the last value for each window
    if test_size == 1:
        # df["targets"] = [subwindows[:, -1].reshape(-1, 1) for subwindows in df["lag_windows"].values]
        df["targets"] = [
            subwindows[:, -1].reshape(-1, 1)
            if len(subwindows.shape) == 2
            else subwindows.reshape(1, -1)[:, -1].reshape(-1, 1)
            for subwindows in df["lag_windows"].values
        ]
    else:
        # df["targets"] = [subwindows[:, -test_size:] for subwindows in df["lag_windows"].values]
        df["targets"] = [
            subwindows[:, -test_size:].reshape(-1, 1)
            if len(subwindows.shape) == 2
            else subwindows.reshape(1, -1)[:, -test_size:].reshape(-1, 1)
            for subwindows in df["lag_windows"].values
        ]
    # lagged values are all values until the last one
    # df["lagged_values"] = [subwindows[:, :-test_size] for subwindows in df["lag_windows"].values]
    df["lagged_values"] = [
        subwindows[:, :-test_size]
        if len(subwindows.shape) == 2
        else subwindows.reshape(1, -1)[:, :-test_size].reshape(-1, 1)
        for subwindows in df["lag_windows"].values
    ]

    return df


def standard_scaler_custom(df, mode="train"):
    """
    A custom standard scaler normalization method.
    Normalized the lagged windows

    Args:
        df (pd.DataFrame):
            A dataframe containing the lagged time series data.
        mode (str):
            A string indicating the mode of the normalization.
            If mode is 'train' then the normalization is performed on the lagged windows and the targets.
            If mode is 'test' then the normalization is performed only on the lagged windows.

    Returns:
        df (pd.DataFrame):
            A dataframe containing the lagged time series data with the normalized lagged windows and targets.

    """

    # Take the mean and the std of each subwindow
    df["mus"] = [
        np.array([np.mean(subwindow) for subwindow in windows]).reshape(-1, 1).tolist()
        for windows in df["lagged_values"].values
    ]
    df["stds"] = [
        np.array([np.std(subwindow) for subwindow in windows]).reshape(-1, 1).tolist()
        for windows in df["lagged_values"].values
    ]

    # Normalize the lagged values by substracting the mean and dividing with the std of every window.
    # If std is zero or nan skip the division.
    df["normalized_lagged_values"] = [
        np.array(
            [(subwindow - mu) / std if std[0] > 0 else subwindow - mu for subwindow, mu, std in zip(windows, mus, stds)]
        ).tolist()
        for windows, mus, stds in zip(df["lagged_values"].values, df["mus"].values, df["stds"].values)
    ]

    # If we have rolling features
    rolling_columns = [col for col in df.columns if "rolling" in col]
    if len(rolling_columns) > 0:
        # Normalize these as well
        for rolling_col in rolling_columns:
            new_rolling_col = "normalized_" + rolling_col
            df[new_rolling_col] = [
                np.array(
                    [
                        (subwindow - mu) / std if std[0] > 0 else subwindow - mu
                        for subwindow, mu, std in zip(windows, mus, stds)
                    ]
                ).tolist()
                for windows, mus, stds in zip(df[rolling_col].values, df["mus"].values, df["stds"].values)
            ]
            # drop the old rolling column
            df = df.drop(columns=rolling_col)
    # Normalize the targets in the same way
    if mode == "train":
        df["normalized_targets"] = [
            np.array(
                [(target - mu) / std if std[0] > 0 else target - mu for target, mu, std in zip(targets, mus, stds)]
            )
            .reshape(-1, 1)
            .tolist()
            for targets, mus, stds in zip(df["targets"].values, df["mus"].values, df["stds"].values)
        ]

    else:
        # Squezze the mus and stds columns
        df["mus"] = df["mus"].apply(lambda x: x[0][0])
        df["stds"] = df["stds"].apply(lambda x: x[0][0])

    return df
