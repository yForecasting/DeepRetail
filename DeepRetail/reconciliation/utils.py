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
    Converts a pandas offset to its equivelant lower frequency
    For example 12M to 1Y, 7D to 1W, 24H to 1D.
    Convertions also take into account the number of periods.

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
    Computes the resampled frequencies. It also converts to a lower equivelant frequency
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


def compute_matrix_S(factors):
    """
    Computes the summation matrix s

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
            S_thief[k][i, factors[k] * i : factors[k] + factors[k] * i] = 1

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
        df = df.iloc[:, total_obs % factor :]

    resample_df = df.resample(
        str(factor) + bottom_freq, closed="left", label="left", axis=1
    ).sum()

    # change the frequency of the columns to the resampled_freq
    resample_df.columns = pd.to_datetime(resample_df.columns).to_period(resampled_freq)

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


def get_w_matrix_structural(frequencies):
    # computes the reconciliation matrix for structural scalling

    # Get the factors
    m = np.flip(frequencies)

    # Get nsum
    nsum = np.flip(np.repeat(m, frequencies))

    # Get the weights
    weights = [1 / weight for weight in nsum]

    # Then convert to diagonal
    W_inv = np.diag(weights)

    return W_inv
