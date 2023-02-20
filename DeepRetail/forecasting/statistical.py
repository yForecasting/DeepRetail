import numpy as np
import pandas as pd
from DeepRetail.forecasting.utils import get_numeric_frequency
from DeepRetail.transformations.formats import sktime_forecast_format, transaction_df
from sktime.forecasting.model_selection import (
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import acf


class StatisticalForecaster(object):

    """
    A class for time series forecasting using statistical methods.

    Parameters:
        models : list
            A list of model names to use for forecasting.
            Currently only 'Naive', 'SNaive', 'ARIMA' and 'ETS' are supported.
        freq : str
            The frequency of the time series data.
        n_jobs : int, optional (default=-1)
            The number of parallel jobs to run during model fitting.

    Args:
        freq : str
            The frequency of the data.
        seasonal_length : int
            The length of the seasonal pattern.
        n_jobs : int
            The number of jobs to run in parallel for the fitting process.
        fitted_models : list
            A list of models that have been fitted.
        model_names : list
            A list of the names of the models that have been fitted.
        fc_df : pd.DataFrame
            The formatted forecast dataframe.
        fh : np.ndarray
            The forecast horizon.
        cv : int
            The number of cross-validation folds.
        y_train : pd.DataFrame
            The training data.
        y_test : pd.DataFrame
            The test data.
        cross_validator : SlidingWindowSplitter
            The cross-validation object.
        forecast_df : pd.DataFrame
            The forecast dataframe, including predicted values and any available true values.

    Methods:
        fit(df, format='pivoted')
            Fits the models to the time series data.
        predict(h, cv=1, holdout=True)
            Generates forecasts for a future period.
        get_model_predictions(model, name)
            Generates forecasts for a future period using a specific model.
        add_fh_cv()
            Adds the forecasting horizon and cross-validation fold numbers to the forecast DataFrame.


    Examples:
        # Create the forecaster
        >>> models = ["ETS"]
        >>> freq = "M"
        >>> n_jobs = -1
        >>> forecaster = StatisticalForecaster(models, freq, n_jobs)

        # Fit the forecaster
        >>> df = pd.read_csv("data.csv")
        >>> forecaster.fit(df, format="pivoted")

        # Generate predictions
        >>> h = 12
        >>> cv = 3
        >>> holdout = True
        >>> predictions = forecaster.predict(h, cv, holdout)


    """

    def __init__(self, models, freq, n_jobs=-1, warning=False, seasonal_length=None):
        """
        Initialize the StatisticalForecaster object.

        Args:
            models: list
                A list of models to fit. Currently only ETS is implemented.
            freq: str
                The frequency of the data, e.g. 'D' for daily or 'M' for monthly.
            n_jobs: int, default=-1
                The number of jobs to run in parallel for the fitting process.
            warning: bool, default=False
                Whether to show warnings or not.
            seasonal_length: int, default=None
                The length of the seasonal pattern.
                If None, the seasonal length is inferred from the frequency.
                On frequencies with multiple seasonal patterns, the first seasonal pattern is used.

        """
        self.freq = freq
        if seasonal_length is not None:
            self.seasonal_length = seasonal_length
        else:
            self.seasonal_length = get_numeric_frequency(freq)
        self.n_jobs = n_jobs

        # Set the warnings
        if not warning:
            warnings.filterwarnings("ignore")

        # Add the models and their names
        models_to_fit = []
        model_names = []

        # Append to the list
        if "Naive" in models:
            models_to_fit.append(NaiveForecaster(strategy="last"))
            model_names.append("Naive")
        if "SNaive" in models:
            models_to_fit.append(
                NaiveForecaster(strategy="last", sp=self.seasonal_length)
            )
            model_names.append("Seasonal Naive")
        if "ARIMA" in models:
            models_to_fit.append(
                StatsForecastAutoARIMA(sp=self.seasonal_length, n_jobs=self.n_jobs)
            )
            model_names.append("ARIMA")
        if "ETS" in models:
            models_to_fit.append(
                AutoETS(auto=True, sp=self.seasonal_length, n_jobs=self.n_jobs)
            )
            model_names.append("ETS")

        self.fitted_models = models_to_fit
        self.model_names = model_names

    def fit(self, df, format="pivoted"):
        """
        Fit the model to given the data.

        Args:
            df : pd.DataFrame
                The input data.
            format : str, default='pivoted'
                The format of the input data. Can be 'pivoted' or 'transactional'.

        Raises:
            ValueError : If the format is not 'pivoted' or 'transactional'.

        """

        if format == "pivoted":
            fc_df = transaction_df(df, drop_zeros=False)
        elif format == "transactional":
            fc_df = df.copy()
        else:
            raise ValueError(
                "Provide the dataframe either in pivoted or transactional format."
            )

        # convert to the right format for forecasting
        fc_df = sktime_forecast_format(fc_df)

        # Fix an issue with frequencies
        fc_df = fc_df.asfreq(self.freq)

        # Add to the object
        self.fc_df = fc_df

    def predict(self, h, cv=1, holdout=True):
        """
        Generates predictions using the statistical forecaster.

        Args:
            h : int
                The forecast horizon (i.e., how many time periods to forecast into the future).
            cv : int, optional (default=1)
                The number of cross-validation folds to use. If set to 1, no cross-validation is performed.
            holdout : bool, optional (default=True)
                If True, a holdout set is used for testing the model. If False, the model is fit on the entire data.

        Raises:
            ValueError : If cv > 1 and holdout is False.

        Returns:
            pandas.DataFrame
            The forecasted values, along with the true values (if holdout=True).

        """
        if not holdout and cv > 1:
            raise ValueError("Cannot do cross validation without holdout.")

        self.fh = np.arange(1, h + 1, 1)
        self.cv = cv

        if holdout:
            self.y_train, self.y_test = temporal_train_test_split(
                self.fc_df, test_size=h
            )
            # Convert y_test to the selected format
            self.y_test = pd.melt(
                self.y_test.reset_index(),
                id_vars=["Period"],
                value_vars=self.y_test.columns,
                value_name="True",
                var_name="unique_id",
            ).rename(columns={"Period": "date"})
        else:
            self.y_train = self.fc_df.copy()
            self.y_test = None

        self.cross_validator = SlidingWindowSplitter(
            window_length=len(self.fc_df) - h - (self.cv - 1), fh=self.fh, step_length=1
        )

        # Get the predictions
        y_pred = pd.concat(
            [
                self.get_model_predictions(model, name)
                for model, name in zip(self.fitted_models, self.model_names)
            ]
        )

        # if we have holdout add the true values
        if self.y_test is not None:
            self.forecast_df = pd.merge(y_pred, self.y_test, on=["unique_id", "date"])

        else:
            self.forecast_df = y_pred.copy()

        # add the fh and cv
        self.add_fh_cv()

        # return
        return self.forecast_df

    def get_model_predictions(self, model, name):
        """
        Fits a given skktime model and generates predictions.

        Args:
            model : sktime.BaseForecaster
                A sktime forecaster model to use for generating predictions.
            name : str
                The name of the model to use.

        Returns:
            pandas.DataFrame
                The predictions generated by the given model.
        """
        # fit the model
        model.fit(self.y_train)

        # get the prediction
        y_pred = model.update_predict(self.fc_df, self.cross_validator)

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

        # add the model name
        y_pred["Model"] = name

        # return
        return y_pred

    def add_fh_cv(self):
        """
        Adds the forecasting horizon and cross-validation information to the forecast results.

        Args:
            None

        """

        # add the number of cv and fh
        cv_vals = sorted(self.forecast_df["cutoff"].unique())
        fh_vals = sorted(self.forecast_df["date"].unique())

        cv_dict = dict(zip(cv_vals, np.arange(1, len(cv_vals) + 1)))
        fh_dict = dict(zip(fh_vals, np.arange(1, len(fh_vals) + 1)))

        self.forecast_df["fh"] = [
            fh_dict[date] for date in self.forecast_df["date"].values
        ]
        self.forecast_df["cv"] = [
            cv_dict[date] for date in self.forecast_df["cutoff"].values
        ]

    def get_residuals(self):
        """
        Generates residuals for the fitted models.

        Args:
            None

        Returns:
            pandas.DataFrame
                The residuals generated by the fitted models.

        """

        # For every model
        for model in self.fitted_models:
            # fit it for one step ahead
            model.fit(self.y_train, fh=[1])

        # Get the residuals
        self.residuals = pd.concat(
            [
                self.calculate_residuals(model, name)
                for model, name in zip(self.fitted_models, self.model_names)
            ]
        )
        self.residuals = self.residuals.reset_index()
        # return
        return self.residuals

    def calculate_residuals(self, model, name):
        """
        Calculates the residuals for a given model.

        Args:
            model : sktime.BaseForecaster
                A sktime forecaster model to use for generating predictions.
            name : str
                The name of the model to use.

        Returns:
            pandas.DataFrame
                The residuals generated by the given model.
        """

        # predict the residuals
        residuals = model.predict_residuals(y=self.y_train)

        # Melt
        residuals = residuals.reset_index()
        residuals = pd.melt(
            residuals, id_vars=["Period"], value_name="residual", var_name="unique_id"
        ).set_index(["unique_id", "Period"])

        # Add the model name
        residuals["Model"] = name

        # return
        return residuals

    def residual_diagnosis(self, model, type, agg_func=None, n=None, index_ids=None):
        """
        Plots the residuals for a given model together with the ACF plot and a histogram.

        Args:
            model : str
                The name of the model to use.
            type : str
                The type of residuals to plot. Can be 'aggregate', 'random' or 'individual'.
                - Aggregate aggregates the residuals given the agg_fun
                - Random takes n random unique_ids
                - Individual takes the unique_ids provided in the index_ids list
            agg_func : str
                The function to use for aggregating the residuals. Only used if type is 'aggregate'.
            n : int
                The number of unique_ids to plot. Only used if type is 'random'.
            index_ids : list
                The list of unique_ids to plot. Only used if type is 'individual'.

        """

        # Get residuals if we haven't already
        if self.residuals is None:
            self.get_residuals()

        # filter residuals for the given model
        f_res = self.residuals[self.residuals["Model"] == model]

        # if we have to aggregate
        if type == "aggregate":
            f_res = f_res.groupby(["Model", "Period"]).agg(agg_func).reset_index()
            f_res["unique_id"] = "Aggregate"
            # set n equal to a single output
            n = 1
        elif type == "random":
            # sample n random unique_ids
            ids = np.random.choice(f_res["unique_id"].unique(), n)
            f_res = f_res[f_res["unique_id"].isin(ids)]

        elif type == "individual":
            # take those provided on the index_ids list
            f_res = f_res[f_res["unique_id"].isin(index_ids)]
            n = len(index_ids)

        # Pivot
        f_res = pd.pivot_table(
            f_res,
            index="unique_id",
            columns="Period",
            values="residual",
            aggfunc="first",
        )

        # Plot

        # Extra  values names and periods
        vals = f_res.values
        dates = f_res.columns.values
        # names = f_res.index.values

        # Initialize params
        gray_scale = 0.9

        for idx in range(n):
            fig = plt.figure(figsize=(16, 8), constrained_layout=True)
            gs = GridSpec(2, 2, figure=fig)

            y = vals[idx]
            # name = names[idx]

            # Define axes
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, :-1])
            ax3 = fig.add_subplot(gs[1:, -1])

            # Ax1 has the line plot
            ax1.plot(dates, y, label="y", color="black")
            ax1.set_facecolor((gray_scale, gray_scale, gray_scale))
            ax1.grid()

            # Ax2 is the pacf plot
            acf_ = acf(y, nlags=get_numeric_frequency(self.freq), alpha=0.05)
            # splitting acf and the intervals
            acf_x, confint = acf_[:2]
            acf_x = acf_x[1:]
            confint = confint[1:]

            lags_x = np.arange(0, self.seasonal_length)

            ax2.vlines(lags_x, [0], acf_x)
            ax2.axhline()
            ax2.margins(0.05)
            ax2.plot(
                lags_x,
                acf_x,
                marker="o",
                markersize=5,
                markerfacecolor="red",
                markeredgecolor="red",
            )

            # ax.set_ylim(-1, 1)
            # Setting the limits
            ax2.set_ylim(
                1.25 * np.minimum(min(acf_x), min(confint[:, 0] - acf_x)),
                1.25 * np.maximum(max(acf_x), max(confint[:, 1] - acf_x)),
            )

            lags_x[0] -= 0.5
            lags_x[-1] += 0.5
            ax2.fill_between(
                lags_x, confint[:, 0] - acf_x, confint[:, 1] - acf_x, alpha=0.25
            )

            gray_scale = 0.93
            ax2.set_facecolor((gray_scale, gray_scale, gray_scale))
            ax2.grid()

            # title = "ACF" + str(nam)
            # ax2.set_title(title)

            ax3.hist(y, color="black")
            ax3.grid()
            ax3.set_facecolor((gray_scale, gray_scale, gray_scale))

            plt.show()
