import numpy as np
import pandas as pd


def get_factors(freq):
    """
    Returns all factors for a given frequency.
    Factors are used to construct the temporal levels.

    Args:
        freq (int, list): frequency or list of frequencies
                List of frequencies is used for multi-frequency data like daily, hourly.

    Returns:
        factors (np.array): array of factors

    """
    # If we have a single frequency
    if isinstance(freq, int):
        factors = np.array([i for i in range(1, freq + 1) if freq % i == 0])

    # If we have a list of frequencies
    # get the union of all factors
    elif isinstance(freq, list):
        factors = np.array(sorted(list(set().union(*[get_factors(i) for i in freq]))))

    return factors


def convert_offset_to_lower_freq(offset):
    """
    Converts a pandas offset to its equivalent lower frequency
    For example 12M to 1Y, 7D to 1W, 24H to 1D.
    Conversions also take into account the number of periods.

    Args:
        offset (str): The offset to convert

    Returns:
        str: The converted offset
    """

    # split the offset into the number and the frequency
    if len(offset) == 1:
        total_periods, freq = 1, offset
    else:
        total_periods, freq = int(offset[:-1]), offset[-1]
    # Loop for every case, H, D, W, M, Q
    # Start from the lowest frequency
    if freq == "H":
        # if we have a factor of 364*24
        if total_periods % 364 * 24 == 0:
            new_freq = str(int(total_periods) // (364 * 24)) + "Y"
        # if we have a factor of 90*24:
        elif int(total_periods) % (90 * 24) == 0:
            new_freq = str(int(total_periods) // (90 * 24)) + "Q"
        # if we have a factor of 30*24:
        elif int(total_periods) % (30 * 24) == 0:
            new_freq = str(int(total_periods) // (30 * 24)) + "M"
        # if we have a factor of 7*24:
        elif int(total_periods) % (7 * 24) == 0:
            new_freq = str(int(total_periods) // (7 * 24)) + "W"
        # if we have a factor of 24:
        if int(total_periods) % 24 == 0:
            new_freq = str(int(total_periods) // 24) + "D"
        else:
            new_freq = offset

    elif freq == "D":
        # if we have a factor of 364:
        if int(total_periods) % 364 == 0:
            new_freq = str(int(total_periods) // 364) + "Y"
        # if we have a factor of 90:
        elif int(total_periods) % 90 == 0:
            new_freq = str(int(total_periods) // 90) + "Q"
        # if we have a factor of 30:
        elif int(total_periods) % 30 == 0:
            new_freq = str(int(total_periods) // 30) + "M"
        # if we have a factor of 7:
        elif int(total_periods) % 7 == 0:
            new_freq = str(int(total_periods) // 7) + "W"
        else:
            new_freq = offset

    elif freq == "W":
        # if we have a factor of 52:
        if int(total_periods) % 52 == 0:
            new_freq = str(int(total_periods) // 52) + "Y"
        # if we have a factor of 13:
        elif int(total_periods) % 13 == 0:
            new_freq = str(int(total_periods) // 13) + "Q"
        # if we have a factor of 4:
        elif int(total_periods) % 4 == 0:
            new_freq = str(int(total_periods) // 4) + "M"
        else:
            new_freq = offset

    elif freq == "M":
        # if we have a factor of 12:
        if int(total_periods) % 12 == 0:
            new_freq = str(int(total_periods) // 12) + "Y"
        # if we have a factor of 3:
        elif int(total_periods) % 3 == 0:
            new_freq = str(int(total_periods) // 3) + "Q"
        else:
            new_freq = offset

    elif freq == "Q":
        # if we have a factor of 4:
        if int(total_periods) % 4 == 0:
            new_freq = str(int(total_periods) // 4) + "Y"
        else:
            new_freq = offset

    elif freq == "Y":
        new_freq = offset

    return new_freq


def compute_resampled_frequencies(factors, bottom_freq):
    """
    Computes the resampled frequencies. It also converts to a lower equivalent frequency
    For example 12M to 1Y, 7D to 1W, 24H to 1D

    Args:
        factors (list): a list of factors for the temporal levels

    Returns:
        resample_factors (list): a list of resampled frequencies
    """

    # Converts to pandas frequencies
    resample_factors = [str(i) + bottom_freq for i in factors]

    # Converts to lower frequencies
    resample_factors = [convert_offset_to_lower_freq(i) for i in resample_factors]

    return resample_factors


def compute_matrix_S_temporal(factors):
    """
    Computes the summation matrix S for temporal levels

    Args:
        factors (list): a list of factors for the temporal levels

    Returns:
        numpy.ndarray: a numpy array representing the summation matrix S

    """

    # get the total number of factors
    total_factors = len(factors)
    # get the highest frequency
    max_freq = max(factors)

    # initialize a list of numpy arrays for every factor
    S_thief = [
        np.zeros((max_freq // factors[k], max_freq)) for k in range(total_factors)
    ]

    # loop through the factors
    for k in range(total_factors):
        # loop through the frequencies
        for i in range(max_freq // factors[k]):
            # populate the S_thief matrix
            S_thief[k][
                i, factors[k] * i : factors[k] + factors[k] * i  # noqa: E203
            ] = 1

    # reverse the order of the stacked levels
    S_thief = S_thief[::-1]

    # stack
    S_thief = np.vstack(S_thief)

    return S_thief


def resample_temporal_level(df, factor, bottom_freq, resampled_freq):
    """
    Resamples a dataframe to a given factor

    Args:
        df (pandas.DataFrame): a dataframe to be resampled
        factor (int): the factor to resample to
        bottom_freq (str): the frequency of the bottom level
        resampled_freq (str): the frequency of the resampled level


    Returns:
        pandas.DataFrame: a resampled dataframe
    """

    # Take the length of the original dataframe
    total_obs = df.shape[1]

    # check if the number of observations is divisible by the factor
    if total_obs % factor != 0:
        # if not, drop observations from the beginning
        df = df.iloc[:, total_obs % factor :]  # noqa: E203

    resample_df = df.resample(
        str(factor) + bottom_freq, closed="left", label="left", axis=1
    ).sum()

    # change the frequency of the columns to the resampled_freq
    # resample_df.columns = pd.to_datetime(resample_df.columns).to_period(resampled_freq)

    return resample_df


def split_reconciled(out, frequencies):
    """
    Split the `out` list into a list of sublists.
    Sublists contain the specified number of elements determined by the corresponding element in the `frequencies` list.

    Args:
    - out (List[Any]): The list to split into sublists.
    - frequencies (List[int]): A list of integers representing the number of elements in each sublist.
            The length of this list determines the number of sublists that will be returned.

    Returns:
    - List[List[Any]]: A list of sublists of `out`,
    """
    splits = []  # Initialize an empty list to store the sublists

    for i, frequency in enumerate(frequencies):
        # Calculate the start and end indices for the current sublist
        start = sum(frequencies[:i])
        end = start + frequency
        # Extract the current sublist from `out` and append it to the `splits` list
        split = out[start:end]
        splits.append(split)

    return splits


def reverse_order(out, frequencies):
    """
    Reverse the order of the elements in the `out` list, and return the resulting list.

    Args:
    - out (List[Any]): The list to reverse.
    - frequencies (List[int]): A list of integers representing the number of elements in each sublist of `out`.

    Returns:
    - List[Any]: The reversed version of the `out` list.
    """

    # Split the `out` list into sublists using the `split_reconciled` function
    split = split_reconciled(out, frequencies)

    # Reverse the order of the sublists in the `split` list
    flip = split[::-1]

    # Flatten the list of sublists into a single list and return it
    return [x for sublist in flip for x in sublist]


def get_w_matrix_structural(frequencies, total_ts):
    """
    Get the W matrix for structural scaling

    Args:
        frequencies (list): a list of frequencies for the temporal levels
        total_ts (int): the total number of time series

    Returns:
        numpy.ndarray: a numpy array representing the W matrix
    """
    # computes the reconciliation matrix for structural scaling

    # Get the factors
    m = np.flip(frequencies)

    # Get nsum
    nsum = np.flip(np.repeat(m, frequencies))

    # Get the weights
    weights = [1 / weight for weight in nsum]

    # Then convert to diagonal
    W_inv = np.diag(weights)

    # Repeat the matrix for each time series
    W_inv = np.dstack([W_inv] * total_ts)

    # Move the time series axis to the front
    W_inv = np.moveaxis(W_inv, -1, 0)

    return W_inv


def compute_y_tilde(y_hat, Smat, Wmat):
    """
    Compute the reconciled y_tilde through matrix multiplications
    Have a fallback for the case where the matrix is not invertible.

    Args:
        y_hat (numpy.ndarray): a numpy array representing the predicted y
        Smat (numpy.ndarray): a numpy array representing the summation matrix S
        Wmat (numpy.ndarray): a numpy array representing the weight W matrix

    Returns:
        numpy.ndarray: a numpy array representing the reconciled y_tilde

    """
    # Does matrix multiplication to compute y_tilde

    # The full format of the matrix is like that
    # S * (S_T * W_inv * S)^-1 S_T * W_inv * pred

    # First we inverse W
    W_inv = np.linalg.pinv(Wmat)

    # Then get the A = S_T * W_inv * S
    A = Smat.T @ W_inv @ Smat

    # Inverse A
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        # A fallback in case the matrix is not invertible due to det = 0
        # This is the pseudo inverse of A
        # https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
        A_inv = np.linalg.pinv(A)

    # Now we have: S * A_inv * S_T * W_inv * pred
    # First take the S* A_inv * S_T
    B = Smat @ A_inv @ Smat.T

    # Now we have: B * W_inv * pred
    # First take the B * W_inv
    C = B @ W_inv

    # Now we have: C * pred
    y_tilde = C @ y_hat

    return y_tilde


def get_w_matrix_mse(res_df):
    """
    Get the W matrix for MSE scaling

    Args:
        res_df (pandas.DataFrame): a pandas DataFrame containing the residuals

    Returns:
        numpy.ndarray: a numpy array representing the W matrix

    """
    # Get the unique_id for all time series
    unique_ts = res_df["unique_id"].unique()

    # Loop over the unique time series to construct unique W matrices
    for i, ts in enumerate(unique_ts):
        # Filter the df
        temp_df = res_df[res_df["unique_id"] == ts]

        # Get them on the right order
        # sort temp_df descending based on temporal_level first and ascending based on fh
        temp_df = temp_df.sort_values(
            by=["temporal_level", "cv"], ascending=[False, True]
        )

        # take the values of the mse
        temp_mse = temp_df["residual_squarred"].values
        # get the weights for each level
        temp_weights = 1 / temp_mse
        # Convert to diagonal matrix
        temp_W = np.diag(temp_weights)
        # Ensure no infs or nans.
        # If exist convert to 0
        temp_weights[np.isinf(temp_weights)] = 0
        temp_weights[np.isnan(temp_weights)] = 0
        # Initialize matrix if it does not exist
        if i == 0:
            # add an extra dimension to the matrix
            W_inv = temp_W[np.newaxis, :, :]
        else:
            # add the matrix to the stack
            W_inv = np.concatenate((W_inv, temp_W[np.newaxis, :, :]), axis=0)

    return W_inv


def compute_matrix_S_cross_sectional(df):
    """
    Estimates the S matrix for cross-sectional reconcliation.

    Args:
        df (pandas.DataFrame): a pandas DataFrame with the hierarchical structure.
            Note: Generated using the extract_hierarchical_structure function.

    Returns:
        pandas.DataFrame: a pandas DataFrame representing the S matrix.

    """

    # Switch the index with the lowest level of the hierarchy
    # Take the new index
    new_index = df.columns[0]
    df = df.reset_index(drop=True).set_index(df[new_index].copy())

    # Get the total levels -> the column names
    # But reversed -> start from the top to the bottom
    total_levels = df.columns.values[::-1]

    # Initialize a dataframe to concat the S matrix
    S = pd.DataFrame()

    # Itterate over column names and stack them to create the S matrix
    for level in total_levels:
        # Get dummies and transpose
        temp_level = pd.get_dummies(df[level]).T

        # Concat with the S mat
        S = pd.concat([S, temp_level], axis=0)

    # Sort columns on S
    # This is the most time consuming step
    S = S.sort_index(axis=1)

    return S
