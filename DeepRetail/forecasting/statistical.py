import numpy as np
import pandas as pd
from DeepRetail.forecasting.utils import get_numeric_frequency
from DeepRetail.transformations.formats import (
    transaction_df,
    statsforecast_forecast_format,
)
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA, Naive, SeasonalNaive
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


class StatisticalForecaster(object):

    """
    A class for time series forecasting using statistical methods.

    Methods:
        __init__(models, freq, n_jobs=-1, warning=False, seasonal_length=None)
            Initialize the StatisticalForecaster object.
        fit(df, format="pivoted", fallback=True, verbose=False)
            Fit the model to given the data.
        predict(h, cv=1, step_size=1, refit=True, holdout=True)
            Generates predictions using the statistical forecaster.
        calculate_residuals()
            Calculates the residuals of the model.
        residuals_diagnosis(model = 'ETS', type = 'random', n = 3)
            Plots diagnosis for the residuals
        add_fh_cv()
            Adds the forecast horizon and cross-validation to the predictions.

    Args:
        models: list
            A list of models to fit.
        freq: str
            The frequency of the data, e.g. 'D' for daily or 'M' for monthly.
        n_jobs: int, default=-1
            The number of jobs to run in parallel for the fitting process.
        warning: bool, default=False
            Whether to show warnings or not.
        seasonal_length: int, default=None
            The length of the seasonal pattern.
            If not given, it is inferred from the frequency.
        df: pd.DataFrame
            The input data.
        format: str, default='pivoted'
            The format of the input data.
            Can be 'pivoted' or 'transactional'.
        fallback: bool, default=True
            Whether to fallback to the default model if the model fails to fit.
            Default selection is Naive
        verbose: bool, default=False
            Whether to show the progress of the model fitting.
        h: int
            The forecast horizon.
        cv: int, default=1
            The number of cross-validations to perform.
        step_size: int, default=1
            The step size for the cross-validation.
        refit: bool, default=True
            Whether to refit the model on the entire data after cross-validation.
        holdout: bool, default=True
            Whether to hold out the last observation for cross-validation.
        model: str, default='ETS'
            The model to plot the residuals for.
        type: str, default='random'
            The type of residuals to plot. Can be 'random', aggregate, individual.
        n: int, default=3
            The number of residuals to plot.

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
            # Check if it returns multiple seasonal lengths
            if isinstance(self.seasonal_length, list):
                # take the first
                self.seasonal_length = self.seasonal_length[0]
        self.n_jobs = n_jobs

        # Set the warnings
        if not warning:
            warnings.filterwarnings("ignore")

        # Add the models and their names
        models_to_fit = []
        model_names = []

        # Append to the list
        if "Naive" in models:
            models_to_fit.append(Naive())
            model_names.append("Naive")
        if "SNaive" in models:
            models_to_fit.append(SeasonalNaive(season_length=self.seasonal_length))
            model_names.append("Seasonal Naive")
        if "ARIMA" in models:
            models_to_fit.append(AutoARIMA(season_length=self.seasonal_length))
            model_names.append("ARIMA")
        if "ETS" in models:
            models_to_fit.append(AutoETS(season_length=self.seasonal_length))
            model_names.append("ETS")

        self.fitted_models = models_to_fit
        self.model_names = model_names

    def fit(self, df, format="pivoted", fallback=True, verbose=False):
        """
        Fit the model to given the data.

        Args:
            df : pd.DataFrame
                The input data.
            format : str, default='pivoted'
                The format of the input data. Can be 'pivoted' or 'transactional'.
            fallback : bool, default=True
                Whether to fallback to the default model if the model fails to fit.
                Default selection is Naive
            verbose : bool, default=False
                Whether to show the progress of the model fitting.
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
        fc_df = statsforecast_forecast_format(fc_df)

        # Define the StatsForecaster
        if fallback:
            self.forecaster = StatsForecast(
                df=fc_df,
                models=self.fitted_models,
                freq=self.freq,
                n_jobs=self.n_jobs,
                fallback_model=Naive(),
                verbose=verbose,
            )
        else:
            self.forecaster = StatsForecast(
                df=fc_df,
                models=self.fitted_models,
                freq=self.freq,
                n_jobs=self.n_jobs,
                verbose=verbose,
            )

        # Add to the object
        self.fc_df = fc_df

    def predict(self, h, cv=1, step_size=1, refit=True, holdout=True):
        """
        Generates predictions using the statistical forecaster.

        Args:
            h : int
                The forecast horizon (i.e., how many time periods to forecast into the future).
            cv : int, optional (default=1)
                The number of cross-validation folds to use. If set to 1, no cross-validation is performed.
            step_size : int, optional (default=1)
                The step size to use for cross-validation. If set to 1, the cross-validation folds are non-overlapping
            refit : bool, optional (default=True)
                Weather to refit the model at each cross-validation. Avoid for big datasets.
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

        if holdout and cv is None:
            cv = 1

        # Add to the object
        self.cv = cv
        self.h = h
        self.holdout = holdout
        self.step_size = step_size
        self.refit = refit

        if holdout:
            # Get the cross_validation
            y_pred = self.forecaster.cross_validation(
                df=self.fc_df,
                h=self.h,
                step_size=self.step_size,
                n_windows=self.cv,
                refit=self.refit,
            )

            # edit the format
            # Reset index and rename
            y_pred = y_pred.reset_index().rename(columns={"ds": "date", "y": "True"})
            # Melt
            y_pred = pd.melt(
                y_pred,
                id_vars=["unique_id", "date", "cutoff", "True"],
                var_name="Model",
                value_name="y",
            )

        else:
            # We just forecast
            y_pred = self.forecaster.forecast(df=self.fc_df, h=self.h)

            # edit the format
            # Reset index and rename
            y_pred = y_pred.reset_index().rename(columns={"ds": "date"})
            # Melt
            y_pred = pd.melt(
                y_pred, id_vars=["unique_id", "date"], var_name="Model", value_name="y"
            )

        # Add to the object
        self.forecast_df = y_pred

        # add the fh and cv
        self.add_fh_cv()

        # return
        return self.forecast_df

    def add_fh_cv(self):
        """
        Adds the forecasting horizon and cross-validation information to the forecast results.

        Args:
            None

        """

        # add the number of cv and fh
        if self.holdout:
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
        else:
            # get the forecasted dates
            dates = self.forecast_df["date"].unique()
            # get a dictionary of dates and their corresponding fh
            fh_dict = dict(zip(dates, np.arange(1, len(dates) + 1)))
            # add the fh
            self.forecast_df["fh"] = [
                fh_dict[date] for date in self.forecast_df["date"].values
            ]
            # also add the cv
            self.forecast_df["cv"] = None

    def calculate_residuals(self):
        """
        Calculate residuals for all horizons.

        Args:
            None

        Returns:
            pandas.DataFrame : The residuals for all models and horizons.

        """

        # Uses statsmodels for getting the residuals
        # statsforecast is buggy
        res = self.calculate_residuals_statsmodels()

        # add the number of cv and fh
        cv_vals = sorted(res["cutoff"].unique())
        cv_dict = dict(zip(cv_vals, np.arange(1, len(cv_vals) + 1)))
        res["cv"] = [cv_dict[date] for date in res["cutoff"].values]

        # add the fh
        fh_vals = np.tile(np.arange(1, self.h + 1), int(len(res) / self.h))
        res["fh"] = fh_vals

        # add the residuals
        self.residuals = res

        # return
        return self.residuals

    def calculate_residuals_statsmodels(self):
        """
        Calculates residuals using the statsmodels ETS
        It is used as a fallback when statsforecast fails
        It fails when len(y) < nparams + 4 where nparams the number of ETS parameters

        Args:
            None

        Returns:
            pandas.DataFrame : The residuals for all models and horizons.

        """

        # Initialize simmulation parameters
        end_date = self.h + self.cv - 1
        fitting_periods = sorted(self.fc_df["ds"].unique())[:-end_date]
        total_windows = len(fitting_periods) - self.h + 1

        # Pivot the dataframe
        temp_df = pd.pivot_table(
            self.fc_df, index="unique_id", columns="ds", values="y", aggfunc="first"
        )

        # Initialize a df
        temp_residuals = pd.DataFrame()

        # Itterate over each time series
        for i, row in temp_df.iterrows():
            # Cut row at the end date
            row = row[:-end_date]

            model = ETSModel(row, seasonal_periods=self.seasonal_length)
            fit = model.fit(disp=False)
            # initialie a df
            in_sample_df = pd.DataFrame()
            for i in range(total_windows - 1):
                # Run the simulation
                in_sample_multistep = fit.simulate(
                    nsimulations=self.h, anchor=i, repetitions=1, random_errors=None
                ).to_frame()
                # add the cutoff
                in_sample_multistep["cutoff"] = fitting_periods[i]
                # add to the df
                in_sample_df = pd.concat([in_sample_df, in_sample_multistep], axis=0)

            # Edit the format
            # add the unique_id
            in_sample_df["unique_id"] = row.name
            # Add the true values
            row = row.to_frame()
            row.columns = ["y_true"]
            in_sample_df = in_sample_df.merge(row, left_index=True, right_index=True)
            # rename
            in_sample_df = in_sample_df.rename(columns={"simulation": "y_pred"})
            # reset index
            in_sample_df = in_sample_df.reset_index(names="date")
            # add to the df
            temp_residuals = pd.concat([temp_residuals, in_sample_df], axis=0)

        # add the Model
        temp_residuals["Model"] = "AutoETS"

        return temp_residuals

    def residual_diagnosis(self, model, type, agg_func=None, n=1, index_ids=None):
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
        if hasattr(self, "residuals"):
            res = self.residuals.copy()
        else:
            res = self.calculate_residuals()

        # Add the residual
        res["residual"] = res["y_true"] - res["y_pred"]
        self.temp = res
        # filter residuals for the given model
        f_res = res[res["Model"].str.contains(model)]

        # Convert the df to the right format
        # 1st: Keep only 1-step ahead residuals
        f_res = f_res[f_res["fh"] == 1]
        # 2nd: Drop columns and rename
        to_keep = ["date", "unique_id", "residual", "Model"]
        f_res = f_res[to_keep].rename(columns={"date": "Period"})

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
