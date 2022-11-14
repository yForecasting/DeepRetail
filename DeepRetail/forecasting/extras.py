import ray
import numpy as np
import pandas as pd


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


def add_fh_cv(res_df):

    # add the number of cv and fh
    cv_vals = sorted(res_df["cutoff"].unique())
    fh_vals = sorted(res_df["date"].unique())

    cv_dict = dict(zip(cv_vals, np.arange(1, len(cv_vals) + 1)))
    fh_dict = dict(zip(fh_vals, np.arange(1, len(fh_vals) + 1)))

    res_df["fh"] = [fh_dict[date] for date in res_df["date"].values]
    res_df["cv"] = [cv_dict[date] for date in res_df["cutoff"].values]

    return res_df


def fit_predict(model, y, y_train, fh, cross_val, name):

    # Fit the model
    model.fit(y_train, fh=fh)

    # Predict
    y_pred = model.update_predict(y, cross_val)

    # Convert to the right format
    y_pred = (
        y_pred.unstack()
        .unstack(level=1)
        .reset_index()
        .rename(columns={"level_0": "cutoff", "Period": "date"})
    )

    # Collapse
    y_pred = pd.melt(
        y_pred,
        id_vars=["date", "cutoff"],
        value_vars=y_pred.columns[2:],
        value_name="y",
        var_name="unique_id",
    )

    # Add the model name
    y_pred["Model"] = name

    # Return
    return y_pred
