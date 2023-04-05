import numpy as np
from DeepRetail.transformations.formats import (
    get_reminder,
    MinMaxScaler_custom,
    transaction_df,
)
from tsfeatures import tsfeatures, stl_features, entropy


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

    features = tsfeatures(t_df, freq=seasonal_period, features=[stl_features, entropy])
    # Estimate CV here
    features["Residual_CoV"] = Residual_CoV(df, periods).squeeze()

    # if plot:
    #    visualize_features(features)

    return features
