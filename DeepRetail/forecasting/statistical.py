import ray
import pandas as pd
import numpy as np

from DeepRetail.forecasting.extras import fit_predict, add_fh_cv
from DeepRetail.preprocessing.converters import transaction_df, forecast_format

from sktime.forecasting.ets import AutoETS
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import (
    SlidingWindowSplitter,
    temporal_train_test_split,
    ForecastingGridSearchCV,
    ExpandingWindowSplitter,
)
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA


class StatisticalForecaster(object):
    def __init__(self, models, seasonal_length, window_length=None, n_jobs=-1):

        self.models = models
        self.seasonal_length = seasonal_length
        self.window_length = window_length
        self.n_jobs = n_jobs

    def fit(
        self,
        df,
        freq,
        observation_threshold=None,
        trailing_zeros_threshold=None,
        total_to_forecast="all",
    ):

        # trailing zeros threshold:
        # how many successive zeros at the end so we discard the time series
        # On observation threshold consider the test set as well
        # (fitted vals + test set)
        # -> Rule of thumb: we need at least 3 times the size of the forecast horizon
        # -> Rule of thumb 2: at least 2-3 full seasonal circles to capture seasonality

        # Generate the model list
        models_to_fit = []
        model_names = []
        # Append to the list
        if "Naive" in self.models:
            models_to_fit.append(NaiveForecaster(strategy="last"))
            model_names.append("Naive")
        if "SNaive" in self.models:
            models_to_fit.append(
                NaiveForecaster(strategy="last", sp=self.seasonal_length)
            )
            model_names.append("Seasonal Naive")
        if "ARIMA" in self.models:
            models_to_fit.append(
                StatsForecastAutoARIMA(sp=self.seasonal_length, n_jobs=self.n_jobs)
            )
            model_names.append("ARIMA")
        if "ETS" in self.models:
            models_to_fit.append(
                AutoETS(auto=True, sp=self.seasonal_length, n_jobs=self.n_jobs)
            )
            model_names.append("ETS")
        # Note -> More models to be added here!

        # Estimate number of non-zero observations and trailing zeros
        obs_count = pd.DataFrame(df.shape[1] - df.isin([0]).sum(axis=1)).rename(
            columns={0: "Total_Observations"}
        )

        if trailing_zeros_threshold is not None:
            obs_count["Trailing_Zeros"] = (
                df.iloc[:, -trailing_zeros_threshold:].isin([0]).sum(axis=1)
            )

        # filter
        if observation_threshold is not None:
            obs_count = obs_count[
                (obs_count["Total_Observations"] > observation_threshold)
            ]
        if trailing_zeros_threshold is not None:
            obs_count = obs_count["Trailing_Zeros"] < trailing_zeros_threshold

        ids = obs_count.reset_index()["unique_id"].unique()
        fc_df = df.loc[ids]

        # Give a summary of the selection
        print(
            f"From a total of {df.shape[0]}, {fc_df.shape[0]}  fullfill the conditions."
        )

        # convert to the right format
        fc_df = transaction_df(fc_df, keep_zeros=False)

        if total_to_forecast != "all":
            # Take a sample
            ids = fc_df["unique_id"].unique()
            sample = np.random.choice(ids, 15)
            fc_df = fc_df[fc_df["unique_id"].isin(sample)]

        # Edit for ETS here
        fc_df = forecast_format(fc_df)

        # Fix an issue with frequencies.
        fc_df = fc_df.asfreq(freq)

        # Complete the fit.
        self.fitted_models = models_to_fit
        self.fc_df = fc_df
        self.model_names = model_names

    def predict(self, h, cv=1):

        # If CV is None then we predict out-of-sample without true values!
        # Still under construction!

        # Prepare the parameters for predicting
        fh = np.arange(1, h + 1, 1)  # sktime forecast horizon
        # Split
        y_train, y_test = temporal_train_test_split(self.fc_df, test_size=h)
        # Convert y_test to the selected format
        y_test = pd.melt(
            y_test.reset_index(),
            id_vars=["Period"],
            value_vars=y_test.columns[1:],
            value_name="True",
            var_name="unique_id",
        ).rename(columns={"Period": "date"})
        # Prepare the cross_val
        cross_val = SlidingWindowSplitter(
            window_length=len(self.fc_df) - h - (cv - 1), fh=fh, step_length=1
        )

        # Make predictions for every model and stack them
        y_pred = pd.concat(
            [
                fit_predict(model, self.fc_df, y_train, fh, cross_val, name)
                for model, name in zip(self.fitted_models, self.model_names)
            ]
        )

        # Add the true values
        y_out = pd.merge(y_pred, y_test, on=["unique_id", "date"])

        # Add the cv and the fh
        y_out = add_fh_cv(y_out)

        return y_out
