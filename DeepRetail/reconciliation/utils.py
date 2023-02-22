import numpy as np


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
            S_thief[k][i, factors[k] * i:factors[k] + factors[k] * i] = 1

    # reverse the order of the stacked levels
    S_thief = S_thief[::-1]

    # stack
    S_thief = np.vstack(S_thief)

    return S_thief
