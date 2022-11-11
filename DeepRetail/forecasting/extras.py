import ray


@ray.remote
def for_ray(forecaster, horizon, n_windows_cv=None):
    """Uses ray to produce forecasts in parallel

    Args:
        forecaster (StatsForecast object): The forecast object
        horizon (int): Forecasting horizon
        n_windows_cv (int, optional): Number of cross-validation windows.
                                      Defaults to None.

    Returns:
        pd.DataFrame: A dataframe with the forecast results.
    """

    # For no cross-validation
    if n_windows_cv is None:
        forecast_df = forecaster.forecast(h=horizon)

        # For cross-validation
    else:
        forecast_df = forecaster.cross_validation(h=horizon, n_windows=n_windows_cv)

    return forecast_df
