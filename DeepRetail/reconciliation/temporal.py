from DeepRetail.forecasting.utils import get_numeric_frequency
from DeepRetail.reconciliation.utils import (
    get_factors,
    compute_resampled_frequencies,
    compute_matrix_S,
)
import numpy as np


class TemporalReconciler(object):
    def __init__(self):
        pass


class THieF(object):
    def __init__(self, bottom_level_freq, factors=None, top_fh=1):
        # Ensure that either factors or bottom_freq is given
        # Raise an error otherwise
        if factors is None and bottom_level_freq is None:
            raise TypeError("Either factors or bottom_freq should be given")

        # Get the numeric frequency
        self.bottom_level_freq = bottom_level_freq
        self.bottom_level_numeric_freq = get_numeric_frequency(self.bottom_level_freq)

        # Construct all factors if they are not given
        if factors is None:
            factors = get_factors(self.highest_freq)
            self.factors = factors
        else:
            self.factors = factors

        # Initialize extra parameters
        # Build the forecast horizons
        self.fhs = np.flip(factors) * top_fh
        self.frequencies = self.fhs.copy()
        self.m = np.flip(self.frequencies)
        self.max_freq = max(self.frequencies)
        self.total_levels = len(self.factors)

        # array indices for the flatten base forecasts
        self.level_indices = np.cumsum(np.insert(self.fhs, 0, 0))
        # self.level_indices = np.cumsum(self.fhs)

        # Get the resampled frequencies for upscaling the data
        self.resampled_factors = compute_resampled_frequencies(
            self.factors, self.bottom_level_freq
        )

        # Construct the Smat
        self.Smat = compute_matrix_S(self.factors)
