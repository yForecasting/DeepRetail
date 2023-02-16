from pandas.tseries.frequencies import to_offset


def get_numeric_frequency(freq):
    """
    Return the frequency of a time series in numeric format.

    The function returns the frequency of a time series in numeric format. This is useful when working with
    forecasting libraries that require the frequency to be a number instead of a string.

    Args:
        freq (str): A string specifying the frequency of the time series.
        Valid values are:
        'Y' (yearly), 'A' (annually), 'Q' (quarterly), 'M' (monthly), 'W' (weekly), 'D' (daily), or 'H' (hourly).

    Returns:
        int: The frequency of the time series, converted from a string to a number.

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
    vals = [1, 1, 4, 12, 52, 7, 24]

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
    numeric_freq = int(freq_dictionary[freq] / period)

    return numeric_freq
