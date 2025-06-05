import numpy as np
import pandas as pd
from DeepRetail.forecasting.utils import (
    get_numeric_frequency,
    add_fh_cv,
    model_selection,
)
from statsforecast.models import Naive
from DeepRetail.transformations.formats import (
    transaction_df,
    statsforecast_forecast_format,
)
from statsforecast import StatsForecast

import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import dask.dataframe as dd

# from dask.distributed import Client

# from fugue_dask import DaskExecutionEngine
from statsmodels.tsa.seasonal import seasonal_decompose


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

    def __init__(
        self,
        models,
        freq,
        n_jobs=1,
        warning=False,
        seasonal_length=None,
        distributed=False,
        n_partitions=None,
        window_size=None,
        seasonal_window_size=None,
    ):
        """
        Initialize the StatisticalForecaster object.

        Args:
            models: list
                A list of models to fit. Currently only ETS is implemented.
            freq: str
                The frequency of the data, e.g. 'D' for daily or 'M' for monthly.
            n_jobs: int, default=1
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

        # Converts models to statsforecast objects
        models_to_fit, model_names = model_selection(
            models, self.seasonal_length, window_size, seasonal_window_size
        )

        self.fitted_models = models_to_fit
        self.model_names = model_names

        self.distributed = distributed
        self.n_partitions = n_partitions
        # Initiate FugueBackend with DaskExecutionEngine if distributed is True
        # if self.distributed:
        # dask_client = Client()
        # engine = DaskExecutionEngine(dask_client=dask_client)  # noqaf841

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

        # Check if we have distributed training
        if self.distributed:
            # Convert the df to a dask dataframe
            fc_df = dd.from_pandas(fc_df, npartitions=self.n_partitions)

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
        # self.refit = refit

        if holdout:
            # Get the cross_validation
            # If we have distributed
            if self.distributed:
                y_pred = self.forecaster.cross_validation(
                    df=self.fc_df,
                    h=self.h,
                    step_size=self.step_size,
                    n_windows=self.cv,
                    refit=self.refit,
                ).compute()  # add the compute here
            else:
                y_pred = self.forecaster.cross_validation(
                    df=self.fc_df,
                    h=self.h,
                    step_size=self.step_size,
                    n_windows=self.cv,
                    # refit=self.refit,
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
            # If we have distributed
            if self.distributed:
                y_pred = self.forecaster.forecast(
                    df=self.fc_df, h=self.h
                ).compute()  # add the compute here
            else:
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
        self.forecast_df = add_fh_cv(self.forecast_df, self.holdout)

        # Remove the index from the models if there
        self.forecast_df = self.forecast_df[self.forecast_df["Model"] != "index"]

        # return
        return self.forecast_df

    def calculate_residuals(self, type="default"):
        """
        Calculate residuals for all horizons.

        Args:
            type: str, optional (default='default')
                The type of residuals to calculate. Options are 'default' and 'multistep'.

        Returns:
            pandas.DataFrame : The residuals for all models and horizons.

        """

        # Ensure type is either default or multistep
        if type not in ["default", "multistep"]:
            raise ValueError("Type must be either 'default' or 'multistep'.")

        # Uses statsmodels for getting the residuals
        # statsforecast is buggy
        res = self.calculate_residuals_statsmodels(type="default")

        # add the number of cv and fh
        cv_vals = sorted(res["cutoff"].unique())
        cv_dict = dict(zip(cv_vals, np.arange(1, len(cv_vals) + 1)))
        res["cv"] = [cv_dict[date] for date in res["cutoff"].values]

        # add the fh
        fh_vals = np.tile(np.arange(1, self.h + 1), int(len(res) / self.h))
        res["fh"] = 1 if type == "default" else fh_vals

        # add the residuals
        self.residuals = res

        # return
        return self.residuals

    def calculate_residuals_statsmodels(self, type):
        """
        Calculates residuals using the statsmodels ETS
        It is used as a fallback when statsforecast fails
        It fails when len(y) < nparams + 4 where nparams the number of ETS parameters

        Args:
            type: str, optional (default='default')
                The type of residuals to calculate.
                Options are 'default' and 'multistep'.

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

            if type == "multistep":
                # Get multi-step in-sample predictions
                for i in range(total_windows - 1):
                    # Run the simulation
                    in_sample_multistep = fit.simulate(
                        nsimulations=self.h, anchor=i, repetitions=1, random_errors=None
                    ).to_frame()
                    # add the cutoff
                    in_sample_multistep["cutoff"] = fitting_periods[i]
                    # add to the df
                    in_sample_df = pd.concat(
                        [in_sample_df, in_sample_multistep], axis=0
                    )
            else:
                # get the fitted values
                in_sample_df = fit.fittedvalues.to_frame()

            # Edit the format
            # add the unique_id
            in_sample_df["unique_id"] = row.name
            # Add the true values
            row = row.to_frame()
            row.columns = ["y_true"]
            in_sample_df = in_sample_df.merge(row, left_index=True, right_index=True)
            # rename
            in_sample_df = in_sample_df.rename(
                columns={"simulation": "y_pred", 0: "y_pred"}
            )
            # reset index
            in_sample_df = in_sample_df.reset_index(names="date")
            # add the cutoff for default tyoe
            if type == "default":
                in_sample_df["cutoff"] = in_sample_df["date"].shift(1)
                # drop the first row
                in_sample_df = in_sample_df.dropna()

            # add to the df
            temp_residuals = pd.concat([temp_residuals, in_sample_df], axis=0)

        # add the Model
        temp_residuals["Model"] = "AutoETS"

        # Calculate the residuals
        temp_residuals["residual"] = temp_residuals["y_true"] - temp_residuals["y_pred"]

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


class SeasonalDecomposeForecaster(object):
    """
    A class for time series forecasting by using Seasonal Decomposition and then Statistical forecasting.

    Currently works only when holdout set is given. Can easily be extended for not using holdout sets.

    """

    def __init__(
        self,
        models,
        freq,
        n_jobs=1,
        warning=False,
        seasonal_length=None,
        distributed=False,
        n_partitions=None,
    ):
        """
        Initialize the StatisticalForecaster object.

        Args:
            models: list
                A list of models to fit. Currently only ETS is implemented.
            freq: str
                The frequency of the data, e.g. 'D' for daily or 'M' for monthly.
            n_jobs: int, default=1
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

        self.models = models

        self.distributed = distributed
        self.n_partitions = n_partitions
        # Initiate FugueBackend with DaskExecutionEngine if distributed is True
        # if self.distributed:
        # dask_client = Client()
        # engine = DaskExecutionEngine(dask_client=dask_client)  # noqaf841

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

        # Extract the seasonalities
        self.seasonal_adjusted_df, seasonalities_df = self.extract_seasonalities(
            df, self.seasonal_length
        )

        # Collapse the seasonalities
        # Collapse it
        self.seasonalities_df = transaction_df(seasonalities_df).rename(
            columns={"y": "seasonality"}
        )

        # Define and fit the statistical forecaster
        self.forecaster = StatisticalForecaster(
            models=self.models,
            freq=self.freq,
            n_jobs=self.n_jobs,
            warning=False,
            distributed=self.distributed,
            n_partitions=self.n_partitions,
            seasonal_length=self.seasonal_length,
        )

        # Fit the forecaster
        self.forecaster.fit(
            self.seasonal_adjusted_df, fallback=fallback, verbose=verbose
        )

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

        # Generate predictions
        forecast_df = self.forecaster.predict(
            h, cv=cv, step_size=step_size, refit=refit, holdout=holdout
        )

        if holdout is True:
            # Generate predictions
            # forecast_df = self.forecaster.predict(h, cv=cv, step_size=step_size, refit=refit, holdout=holdout)

            # Merge with the forecast_df
            forecast_df = pd.merge(
                forecast_df, self.seasonalities_df, on=["date", "unique_id"]
            )

        else:
            # First forecast seasonalities
            temp_seas = pd.pivot_table(
                self.seasonalities_df,
                index="unique_id",
                columns="date",
                values="seasonality",
            )

            # define a seasonal forecaster
            seasonal_forecaster = StatisticalForecaster(
                models=["SNaive"], freq=self.freq
            )
            seasonal_forecaster.fit(temp_seas)

            # Predict for h steps ahead
            seasonal_prediction = (
                seasonal_forecaster.predict(h=h, holdout=False)
                .rename(columns={"y": "seasonality"})
                .drop(columns=["Model", "fh", "cv"])
            )

            # Merge with the forecast_df
            forecast_df = pd.merge(
                forecast_df, seasonal_prediction, on=["unique_id", "date"]
            )

        # Make the seasonal correcation
        forecast_df["y"] = forecast_df["y"] + forecast_df["seasonality"]

        # Drop the seasonality
        forecast_df = forecast_df.drop(columns=["seasonality"])

        # Add the "Decomposed_" on the Model
        forecast_df["Model"] = "Decomposed_" + forecast_df["Model"]

        return forecast_df

    def extract_seasonalities(self, df, seasonal_length):
        """
        Extracts the seasonalities from the data.

        Args:
            df : pd.DataFrame
                The input data.
            seasonal_length : int
                The length of the seasonal pattern.

        Returns:
            seasonal_adjusted_df : pd.DataFrame
                The data with the seasonalities removed.
            seasonalities_df : pd.DataFrame
                The extracted seasonalities.

        """

        seasonal_adjusted_list = []
        seasonalities_list = []

        for idx, row in df.iterrows():
            temp_seasonality = seasonal_decompose(
                row, model="additive", period=seasonal_length
            ).seasonal
            temp_seasonal_adjusted = row - temp_seasonality

            seasonal_adjusted_list.append(temp_seasonal_adjusted)
            seasonalities_list.append(temp_seasonality)

        seasonal_adjusted_df = pd.DataFrame(
            seasonal_adjusted_list, index=df.index, columns=df.columns
        )
        seasonalities_df = pd.DataFrame(
            seasonalities_list, index=df.index, columns=df.columns
        )

        return seasonal_adjusted_df, seasonalities_df

    def calculate_residuals(self):
        """
        Calculates the residuals of the model.
        Temporal method. Not stable for now.

        Args:


        Returns:
            residuals_df : pd.DataFrame
                The residuals of the model.

        """

        # Calculate the residuals
        residuals_df = self.forecaster.calculate_residuals()

        return residuals_df
