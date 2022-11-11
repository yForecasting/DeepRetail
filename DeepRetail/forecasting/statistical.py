import ray
import pandas as pd
from DeepRetail.forecasting.extras import for_ray
from DeepRetail.preprocessing.converters import transaction_df
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, WindowAverage, Naive, ETS, AutoARIMA
import numpy as np


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

        # Append to the list
        if "Naive" in self.models:
            models_to_fit.append(Naive())
        if "SNaive" in self.models:
            models_to_fit.append(SeasonalNaive(season_length=self.seasonal_length))
        if "MovingAverage" in self.models:
            models_to_fit.append(WindowAverage(self.window_length))
        if "ETS" in self.models:
            models_to_fit.append(ETS(season_length=self.seasonal_length))
        if "ARIMA" in self.models:
            models_to_fit.append(AutoARIMA(season_length=self.seasonal_length))

        # Estimate number of non-zero observations and trailing zeros
        obs_count = pd.DataFrame(df.shape[1] - df.isin([0]).sum(axis=1)).rename(
            columns={0: "Total_Observations"}
        )
        obs_count["Trailing_Zeros"] = (
            df.iloc[:, -trailing_zeros_threshold:].isin([0]).sum(axis=1)
        )

        # filter
        if observation_threshold is not None:
            obs_count_f = obs_count[
                (obs_count["Total_Observations"] > observation_threshold)
            ]
        if trailing_zeros_threshold is not None:
            obs_count_f = obs_count_f["Trailing_Zeros"] < trailing_zeros_threshold

        ids = obs_count_f.reset_index()["unique_id"].unique()
        fc_df = df.loc[ids]

        # Give a summary of the selection
        print(
            f"From a total of {df.shape[0]}, {fc_df.shape[0]}  fullfill the conditions for forecasting"
        )

        # convert to the right format for stats forecasts
        # simply renaming
        fc_df = transaction_df(fc_df, keep_zeros=False)

        # Prepare the date column
        fc_df = fc_df.rename(columns={"date": "ds"})
        fc_df["ds"] = pd.to_datetime(fc_df["ds"])

        if total_to_forecast != "all":
            # Take a sample
            ids = fc_df["unique_id"].unique()
            sample = np.random.choice(ids, 15)
            fc_df = fc_df[fc_df["unique_id"].isin(sample)]

        # Define the forecaster
        forecaster = StatsForecast(
            df=fc_df, models=models_to_fit, freq=freq, n_jobs=self.n_jobs
        )
        # Complete the fit
        self.forecaster = forecaster

    def predict(self, fh, cv=None, parallel=True):

        # if we do not have cv
        if cv is None:
            cv = 1  # set it to 1 to deal with some issues

        if parallel:
            # For parallelism use ray
            res_df = ray.get(for_ray.remote(self.forecaster, fh, cv))

        # for no parallelism just forecast
        else:
            if cv is None:
                res_df = self.forecaster.forecast(h=fh)
            else:
                res_df = self.forecaster.cross_validation(h=fh, n_windows=cv)

        return res_df
