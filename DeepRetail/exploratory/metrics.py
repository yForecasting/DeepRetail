# Includes the metrics

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# a small epsilon to avoid devisions with zero
e = np.finfo(np.float64).eps


def simple_error(actual, predicted, *args):
    # Simple difference
    return actual - predicted


def percentage_error(actual, predicted, *args):
    # % Error
    return simple_error(actual, predicted) / (
        actual + e
    )  # The small e asserts that division is not with 0


def scaled_error(actual, predicted, train, scale_factor=None, lag=1, *args):
    # Scalling the error with the given factor
    # If scalling factor is not given, scales with the naive in-sample mse

    error = simple_error(actual, predicted)
    if scale_factor != None:
        scale_denom = scale_factor
    elif len(train) > 0:
        # Getting the naive predictions
        y_naive = naive_forecasts(train, lag)
        # And the mse
        denom = mse(train[lag:], y_naive)
    else:
        raise ValueError("Provide a scalling factor or the training data ")
    return error / denom


def naive_forecasts(actual, lag=1):
    # Just repeats previous samples
    return actual[:-lag]


def mse(actual, predicted, *args):
    # The mse
    return mean_squared_error(y_true=actual, y_pred=predicted)


def rmse(actual, predicted, *args):
    # for rmse just turn squared to false
    return mean_squared_error(y_true=actual, y_pred=predicted, squared=False)


def mae(actual, predicted, *args):
    return mean_absolute_error(y_true=actual, y_pred=predicted)


def mape(actual, predicted, *args):
    # !!!!!!!Carefull!!!!!!
    # MAPE is biased as it is not symmetric
    # MAPE is not defined for actual = 0
    error = np.abs(percentage_error(actual, predicted))
    return np.mean(error)


def smape(actual, predicted, *args):
    # Symmetric mape
    error = (
        2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + e)
    )
    return np.mean(error)


def mase(actual, predicted, naive_mae, train=None, lag=1, *args):

    # The original MASE as proposed by "Another look at measures of forecast accuracy", Rob J Hyndman
    # Uses the in-sample Naive Forecast for scaling!
    # Train data should be provided

    # computed with naive forecasting for lag = 1
    # can be configured for snaive with lag = freq

    # Getting the MAE of the in-sample naive forecast
    if naive_mae != None:
        scale_denom = naive_mae
    elif train != None:
        scale_denom = mae(train[lag:], naive_forecasts(train, lag))
    else:
        raise ValueError("Provide in-sample mae or the training data ")
    # Getting the mae
    num = mae(actual, predicted)
    return num / scale_denom


def rmsse(actual, predicted, naive_mse=None, train=None, lag=1, *args):
    # Reference:
    # https://github.com/alan-turing-institute/sktime/blob/main/sktime/performance_metrics/forecasting/_functions.py
    # root mean squared scaled error
    # the variant of MASE used as a metric in M5

    # It is unweighted, I can also weight it!
    num = mse(actual, predicted)
    if naive_mse != None:
        denom = naive_mse
    elif len(train) > 0:
        # Getting the naive predictions
        y_naive = naive_forecasts(train, lag)
        # And the mse
        denom = mse(train[lag:], y_naive)
    else:
        raise ValueError("Provide in-sample mse or the training data ")
    error = np.sqrt(num / np.maximum(denom, e))
    return error
