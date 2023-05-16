import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from DeepRetail.transformations.formats import (
    get_reminder,
    MinMaxScaler_custom,
    transaction_df,
)
from tsfeatures import tsfeatures, stl_features, entropy, lumpiness


def Residual_CoV(df, periods):
    """Estimates the coefficient of variation of the residuals of a seasonal decomposition
    Currently not supported due to statsmodels dependency.



    Args:
        df (pd.DataFrame): The pivoted dataframe
        periods (list): A list of seasonal periods

    Returns:
        pd.DataFrame: A dataframe with the CoV of the residuals
    """

    # Get the residuals
    residuals = get_reminder(df, periods=periods)

    # Get the std of the residuals
    std_residuals = np.array([np.std(res) for res in residuals])

    # Get the mean of the original series
    mean_original = np.array([df.mean(axis=1).values])

    # Get the CoV values
    CoV = std_residuals / mean_original

    # Map to [0,1] range using min max scaler
    CoV_scaled = MinMaxScaler_custom(CoV)

    # Reverse the scale so bigger => more forecastable
    CoV_scaled = 1 - CoV_scaled

    return CoV_scaled


def get_features(df, seasonal_period, periods):
    """Estimates coefficient of variation, entropy, seasonality and trend using tsfeatures

    Args:
        df (pd.DataFrame): The pivoted dataframe
        seasonal_period (int): The seasonal period (frequency)
        forecastability (str, optional): The forecastability metric to use.
                                        Accepts either 'entropy' or 'CoV'.
                                        CoV is a bit slower. Defaults to 'entropy'.
        periods (list, optional): The periods to use for the decomposition.

    Returns:
        features (pd.DataFrame):
            A dataframe with the estimate features
    """

    # convert to transaction format
    t_df = transaction_df(df)

    # Estimate the features

    features = tsfeatures(t_df, freq=seasonal_period, features=[stl_features, entropy, lumpiness])
    # Estimate CV here
    features["Residual_CoV"] = Residual_CoV(df, periods).squeeze()

    # if plot:
    #    visualize_features(features)

    return features


def get_class(p, v):
    """
    Estimates the intermittency

    """
    # returns the class
    if p < 1.32:
        if v < 0.49:
            out = "Smooth"
        else:
            out = "Erratic"
    else:
        if v < 0.49:
            out = "Intermittent"
        else:
            out = "Lumpy"

    return out


def get_intermittent_class(ts):
    """Returns the intermittent class of a time series

    Args:
        ts (array-type): The time series to check its intermittency

    Returns:
        str: A description of the intermittency
    """

    # Get the indices of the true demand
    nzd = np.where(ts != 0)[0]

    # If we have at least non-zero demand value
    if len(nzd) > 0:

        # Get the total non-zero observations
        # k = len(nzd)

        # Get the actual demand
        z = ts[nzd]  # the demand

        # Get the intervals between non-zero observations
        x = np.diff(nzd)
        x = np.insert(x, 0, nzd[0])

        # Get the average interval -> will be used for classification later
        p = np.mean(x)

        # Get the squared cv
        v = (np.std(z) / np.mean(z)) ** 2

        # classify
        in_class = get_class(p, v)

    # If we have no demand!
    else:
        in_class = "No Demand"

    return in_class


def intermittency_classification(df, plot=True):
    """Classifies the time series of the dataframe based on their intermittency

    Args:
        df (pd.DataFrame): The dataframe with the time series
        plot (bool, optional): If the function returns a barplot with the results.
                                Defaults to True.
    """

    data = df.values

    # Get the classes
    classes = [get_intermittent_class(ts) for ts in data]

    # Counts how many we have on each class
    class_to_dict = dict(Counter(classes))

    print("\n".join("{}: {}".format(k, v) for k, v in class_to_dict.items()))

    if plot:
        # Plotting vs the % of zeros
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)

        ax.bar(range(len(class_to_dict)), class_to_dict.values(), align="center")

        # ticks = range(len(class_to_dict)), list(class_to_dict.keys())
        gray_scale = 0.93

        ticks = np.arange(0, len(class_to_dict), 1)

        ax.set_xticks(ticks=ticks)
        ax.set_xticklabels(labels=class_to_dict.keys())

        ax.set_facecolor((gray_scale, gray_scale, gray_scale))
        ax.grid(linestyle="-", axis="y")
        ax.set_title("Classification of Demand Time Series")

        plt.show()
