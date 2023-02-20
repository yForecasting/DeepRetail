from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# a small epsilon to avoid devisions with zero
e = np.finfo(np.float64).eps


def simple_error(actual, predicted, *args, **kwargs):
    """
    Calculates the simple error between actual and predicted values.

    Args:
        actual (np.array): The actual values.
        predicted (np.array): The predicted values.

    Returns:
        np.array: The simple error.

    """
    # Simple difference
    return actual - predicted


def bias(actual, predicted, *args, **kwargs):
    """
    Calculates the bias between actual and predicted values.

    Args:
        actual (np.array): The actual values.
        predicted (np.array): The predicted values.

    Returns:
        np.array: The bias.

    """
    return np.mean(simple_error(actual, predicted, *args, **kwargs))


def scaled_error(actual, predicted, naive_mse=None, train=None, lag=1, *args, **kwargs):
    """
    Calculates the scaled error between actual and predicted values.
    Scaling is done by dividing the error with the naive mse.
    Either the naive mse or the training data must be provided.

    Args:
        actual (np.array): The actual values.
        predicted (np.array): The predicted values.
        naive_mse (float): The naive mse. Default: None.
        train (pd.DataFrame): The training data. Default: None.
        lag (int): The lag to use for the naive forecast. Default: 1.

    Returns:
        np.array: The scaled error.
    """

    # Scalling the error with the naive mse

    error = simple_error(actual, predicted, **kwargs)
    if naive_mse is not None:
        denom = naive_mse
    elif len(train) > 0:
        # Getting the naive predictions
        y_naive = naive_forecasts(train, lag)
        # And the mse
        denom = mse(train[lag:], y_naive)
    else:
        raise ValueError("Provide in-sample mse or the training data ")
    return np.mean(error) / denom


def percentage_error(actual, predicted, *args, **kwargs):
    """
    Calculates the percentage error between actual and predicted values.

    Args:
        actual (np.array): The actual values.
        predicted (np.array): The predicted values.

    Returns:
        np.array: The percentage error.
    """

    # % Error
    return simple_error(actual, predicted) / (
        actual + e
    )  # The small e asserts that division is not with 0


def naive_forecasts(actual, lag=1):
    """
    Calculates the naive forecasts for a given lag.

    Args:
        actual (np.array): The actual values.
        lag (int): The lag to use for the naive forecast. Default: 1.

    Returns:
        np.array: The naive forecasts.
    """

    # Just repeats previous samples
    return actual[:-lag]


def mse(actual, predicted, *args, **kwargs):
    """
    Calculates the mean squared error between actual and predicted values.

    Args:
        actual (np.array): The actual values.
        predicted (np.array): The predicted values.

    Returns:
        np.array: The mean squared error.

    """

    # The mse
    return mean_squared_error(y_true=actual, y_pred=predicted)


def rmse(actual, predicted, *args, **kwargs):
    """
    Calculates the root mean squared error between actual and predicted values.

    Args:
        actual (np.array): The actual values.
        predicted (np.array): The predicted values.

    Returns:
        np.array: The root mean squared error.
    """

    # for rmse just turn squared to false
    return mean_squared_error(y_true=actual, y_pred=predicted, squared=False)


def mae(actual, predicted, *args, **kwargs):
    """
    Calculates the mean absolute error between actual and predicted values.

    Args:
        actual (np.array): The actual values.
        predicted (np.array): The predicted values.

    Returns:
        np.array: The mean absolute error.
    """
    return mean_absolute_error(y_true=actual, y_pred=predicted)


def mape(actual, predicted, *args, **kwargs):
    """
    Calculates the mean absolute percentage error between actual and predicted values.

    Args:
        actual (np.array): The actual values.
        predicted (np.array): The predicted values.

    Returns:
        np.array: The mean absolute percentage error.

    """

    # !!!!!!!Carefull!!!!!!
    # MAPE is biased as it is not symmetric
    # MAPE is not defined for actual = 0
    error = np.abs(percentage_error(actual, predicted))
    return np.mean(error)


def smape(actual, predicted, *args, **kwargs):
    """
    Calculates the symmetric mean absolute percentage error between actual and predicted values.

    Args:
        actual (np.array): The actual values.
        predicted (np.array): The predicted values.

    Returns:
        np.array: The symmetric mean absolute percentage error.
    """
    # Symmetric mape
    error = (
        2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + e)
    )
    return np.mean(error)


def wmape(actual, predicted, *args, **kwargs):
    """
    Calculates the weighted mean absolute percentage error between actual and predicted values.

    Args:
        actual (np.array): The actual values.
        predicted (np.array): The predicted values.

    Returns:
        np.array: The weighted mean absolute percentage error.
    """
    # Weighted mape
    error = np.abs(actual - predicted)
    weights = np.abs(actual)
    return np.mean(error / weights)


def mase(actual, predicted, naive_mae, train=None, lag=1, *args, **kwargs):
    """
    Calculates the mean absolute scaled error between actual and predicted values.
    As proposed by "Another look at measures of forecast accuracy", Rob J Hyndman

    Either training data or the naive_mae should be given.
    If training data is given computes the naive_mae internaly

    Args:
        actual (np.array): The actual values.
        predicted (np.array): The predicted values.
        naive_mae (float): The mean absolute error of the naive forecast, used for scaling.
        train (np.array): The training data. Default: None.
        lag (int): The lag to use for the naive forecast. Default: 1.

    Returns:
        np.array: The mean absolute scaled error.

    """
    # Getting the MAE of the in-sample naive forecast
    if naive_mae is not None:
        scale_denom = naive_mae
    elif train is not None:
        scale_denom = mae(train[lag:], naive_forecasts(train, lag))
    else:
        raise ValueError("Provide in-sample mae or the training data ")
    # Getting the mae
    num = mae(actual, predicted)
    return num / scale_denom


def rmsse(actual, predicted, naive_mse=None, train=None, lag=1, *args, **kwargs):
    """
    Calculates the root mean squared scaled error between actual and predicted values.
    The variant of MASE used as a metric in M5.

    Either training data or the naive_mse should be given.
    If training data is given computes the naive_mse internaly

    Reference:
        https://github.com/alan-turing-institute/sktime/blob/main/sktime/performance_metrics/forecasting/_functions.py

    Args:
        actual (np.array): The actual values.
        predicted (np.array): The predicted values.
        naive_mse (float): The mean squared error of the naive forecast, used for scaling.
        train (np.array): The training data. Default: None.
        lag (int): The lag to use for the naive forecast. Default: 1.

    Returns:
        np.array: The root mean squared scaled error.
    """

    # It is unweighted, I can also weight it!
    num = mse(actual, predicted)
    if naive_mse is not None:
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
