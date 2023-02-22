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
