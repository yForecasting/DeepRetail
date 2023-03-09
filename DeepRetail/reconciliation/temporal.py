from DeepRetail.forecasting.utils import get_numeric_frequency
from DeepRetail.forecasting.statistical import StatisticalForecaster
from DeepRetail.reconciliation.utils import (
    get_factors,
    compute_resampled_frequencies,
    compute_matrix_S_temporal,
    resample_temporal_level,
    reverse_order,
    get_w_matrix_structural,
    compute_y_tilde,
    get_w_matrix_mse,
)
import numpy as np
import pandas as pd


class TemporalReconciler(object):
    """
    A class for temporal reconciliation.
    Reconciles base forecasts of different temporal levels.

    Currently supports only structural and mse reconciliation.
    Can handle simmulnteous reconciliation of multiple time series.
    Is also extended to support cross validation.

    Args:
        bottom_level_freq : str
            The frequency of the bottom level forecasts.
        bottom_level_numeric_freq : int
            The numeric frequency of the bottom level forecasts.
        factors : list
            The factors to use for the reconciliation.
        fhs : list
            The forecast horizons of the forecasts.
        frequencies : list
            The frequencies of the forecasts.
        m : list
            The frequencies of the forecasts.
        max_freq : int
            The maximum frequency of the forecasts.
        total_levels : int
            The total number of levels.
        Smat : numpy.ndarray
            The matrix S.
        base_fc_df : pandas.DataFrame
            The base forecasts.
        reconciliation_ready_df : pandas.DataFrame
            The base forecasts in the format for reconciliation.
        Wmat : numpy.ndarray
            The matrix W with the reconciliation weights
        reconciled_predictions : pandas.DataFrame
            The reconciled predictions.

    Methods:
        get_reconciliation_format()
            Converts the base forecasts to the format for reconciliation.
        compute_matrix_W(reconciliation_method, residual_df=None)
            Computes the matrix W.
        get_reconciled_predictions()
            Gets the reconciled predictions.

    Examples:
        >>> from DeepRetail.reconciliation.temporal import TemporalReconciler
        >>> import pandas as pd

        >>> # Create a reconciler
        >>> temporal_reconciler = TemporalReconciler(bottom_level_freq = 'Q')

        >>> # Create the base forecasts
        >>> base_fc_df = pd.DataFrame({
        ...     'unique_id': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
        ...     'temporal_level': [1, 2, 2, 4, 4, 4, 4],
        ...     'fh': [1, 1, 2, 1, 2, 3, 4],
        ...     'y': [20, 12, 11, 5, 3, 4, 2],
        ...     'Model': ['Model1', 'Model1', 'Model1', 'Model1', 'Model1', 'Model1', 'Model1']
        ... })

        >>> # Fit the reconciler
        >>> temporal_reconciler.fit(base_fc_df)

        >>> # Reconcile
        >>> reconciled = temporal_reconciler.reconcile('struc')
    """

    def __init__(
        self, bottom_level_freq, factors=None, top_fh=1, holdout=False, cv=None
    ):
        """
        Initializes the TemporalReconciler class.

        Args:
            bottom_level_freq (str): The frequency of the bottom level forecasts.
                Should be given if factors are not given.
            factors (list, optional): The factors to use for the reconciliation.
                Should be given if bottom_freq is not given.
            top_fh (int, optional): The forecast horizon of the top level forecasts.
                The default is 1.
            holdout (bool, optional): Whether to use holdout or not.
            cv (int, optional): The number of folds to use for holdout.

        Raises:
            TypeError: If neither factors nor bottom_freq is given.

        Returns:
            None
        """
        # Ensure that either factors or bottom_freq is given
        # Raise an error otherwise
        if factors is None and bottom_level_freq is None:
            raise TypeError("Either factors or bottom_freq should be given")

        # Get the numeric frequency
        self.bottom_level_freq = bottom_level_freq
        self.bottom_level_numeric_freq = get_numeric_frequency(self.bottom_level_freq)

        # Construct all factors if they are not given
        if factors is None:
            factors = get_factors(self.bottom_level_numeric_freq)
            self.factors = factors
        else:
            self.factors = factors

        # Initialize extra parameters
        # Build the forecast horizons
        self.fhs = np.flip(factors) * top_fh
        self.frequencies = self.fhs.copy()
        self.m = np.flip(self.frequencies)
        self.max_freq = max(self.frequencies)
        self.total_levels = len(self.factors)

        # Construct the Smat
        self.Smat = compute_matrix_S_temporal(self.factors)

        self.holdout = holdout
        self.cv = cv

    def get_reconciliation_format(self):
        """
        Converts the base forecasts to the format for reconciliation.

        Args:
            None

        Returns:
            pandas.DataFrame: The base forecasts in the format for reconciliation.

        """
        # Converts to the format for reconciliation

        temp_df = self.base_fc_df.copy()

        # Add the temporal indexer to identify temporal levels
        temp_df["temp_indexer"] = (
            temp_df["temporal_level"].astype(str) + "_" + temp_df["fh"].astype(str)
        )

        # Add the model to the unique_id
        temp_df["unique_id_model"] = temp_df["unique_id"] + "-" + temp_df["Model"]

        # Pivot
        temp_df = pd.pivot_table(
            temp_df,
            index="unique_id_model",
            columns="temp_indexer",
            values="y",
            aggfunc="first",
        )

        # Should have another option for when we have cv too.

        # order the columns
        cols = temp_df.columns.tolist()
        # first should be the column with the highest value before the _ and the lowest after the _
        cols.sort(key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))
        # reverse the order
        cols = reverse_order(cols, self.frequencies)
        # add the new order
        temp_df = temp_df[cols]

        return temp_df

    def compute_matrix_W(self, reconciliation_method, residual_df=None):
        """
        Computes the weight matrix W based on the reconciliation method given

        Args:
            reconciliation_method (str): The reconciliation method to use.
                Should be one of "struc" or "mse".
            residual_df (pandas.DataFrame, optional): The residuals of the base forecasts.
                Should be given if reconciliation_method is "mse".

        Raises:
            ValueError: If reconciliation_method is "mse" and residual_df is not given.

        Returns:
            numpy.ndarray: The weight matrix W.

        """
        # Computes the matrix W

        # For structural scalling
        if reconciliation_method == "struc":
            Wmat = get_w_matrix_structural(
                self.frequencies, len(self.reconciliation_ready_df)
            )

        elif reconciliation_method == "mse":
            if residual_df is None:
                raise ValueError("Residuals should be given for mse reconciliation")

            # Get the squared residuals
            residual_df["residual_squarred"] = residual_df["residual"] ** 2

            # Groupby unique_id, temporal_level and fh and get the Mean Squared Error
            residual_df = (
                residual_df.groupby(["unique_id", "temporal_level"])
                .agg({"residual_squarred": "mean"})
                .reset_index()
            )

            # Get the matrix W
            Wmat = get_w_matrix_mse(residual_df, self.factors)

        elif reconciliation_method == "variance":
            ...

        else:
            raise ValueError("Reconciliation method not supported")

        return Wmat

    def get_reconciled_predictions(self):
        """
        Reconciles base forecasts.

        Args:
            None

        Returns:
            pandas.DataFrame: The reconciled forecasts.
        """
        # function to get the reconciliated predictions for every value on the df

        # First extract values from the dataframe
        values = self.reconciliation_ready_df.values
        ids = self.reconciliation_ready_df.index
        cols = self.reconciliation_ready_df.columns

        # For every set of base forecasts reconcile using the reconciliation function compute_y_tilde
        reconciled_values = [
            compute_y_tilde(y, self.Smat, mat) for y, mat in zip(values, self.Wmat)
        ]

        # Convert to dataframe
        reconciled_df = pd.DataFrame(reconciled_values, index=ids, columns=cols)

        return reconciled_df

    def reverse_reconciliation_format(self, reco_method):
        """
        Reverse the reconciliation format to the forecast format supported in DeepRetail.

        Args:
            reco_method (str): The reconciliation method used to reconcile the forecasts.

        Returns:
            pandas.DataFrame: The reconciled forecasts in the format supported in DeepRetail.
        """
        # Function to reverse the reconciliation format to the original one

        temp_df = self.base_fc_df.copy()
        reco_df = self.reconciled_df

        # Prepare the base forecasts
        # Get the temp indexer
        temp_df["temp_indexer"] = (
            temp_df["temporal_level"].astype(str) + "_" + temp_df["fh"].astype(str)
        )

        # Melt the reconciled dataframe
        reco_df = reco_df.reset_index()
        reco_df = pd.melt(
            reco_df,
            id_vars="unique_id_model",
            value_vars=reco_df.columns[1:],
            var_name="temp_indexer",
            value_name="y",
        )

        # Split the unique id and the model
        reco_df["unique_id"] = reco_df["unique_id_model"].apply(
            lambda x: x.split("-")[0]
        )
        reco_df["Base_Model"] = reco_df["unique_id_model"].apply(
            lambda x: x.split("-")[1]
        )

        # Add the Model
        reco_df["Model"] = "TR" + "-" + reco_method + "-" + reco_df["Base_Model"]

        # Drop the unique_id_model and the base model
        reco_df = reco_df.drop(["unique_id_model", "Base_Model"], axis=1)

        # rename the column for base forecasts
        temp_df = temp_df.rename(columns={"y": "y_base"})

        # merge with the base forecasts
        reco_df = reco_df.merge(
            temp_df[["unique_id", "temporal_level", "fh", "temp_indexer", "y_base"]],
            on=["unique_id", "temp_indexer"],
            how="left",
        ).drop("temp_indexer", axis=1)

        return reco_df

    def fit(self, base_fc_df):
        """
        Fits the reconciler on the given dataframe of base forecasts

        Args:
            base_fc_df (pandas.DataFrame): The dataframe of base forecasts.


        Returns:
            None
        """

        # if we have holdout:
        if self.holdout:
            # Initialize a list for the each fold
            self.reconciled_df_list = []

            # Keep this one to filter later
            self.original_df = base_fc_df.copy()

            # Iterate over the folds
            for k in range(self.cv):
                # Filter on the cv
                self.base_fc_df = self.original_df[self.original_df["cv"] == k + 1]

                # Converts to the right forecast format
                self.reconciliation_ready_df = self.get_reconciliation_format()

                # Append
                self.reconciled_df_list.append(self.reconciliation_ready_df)

        else:
            self.base_fc_df = base_fc_df.copy()
            # Converts to the right forecast format
            self.reconciliation_ready_df = self.get_reconciliation_format()

    def reconcile(self, reconciliation_method, residual_df=None):
        """
        Reconciles the base forecasts.

        Args:
            reconciliation_method (str): The reconciliation method to use.
            residual_df (pandas.DataFrame, optional): The dataframe of residuals.

        Returns:
            pandas.DataFrame: The reconciled forecasts.

        """
        # reconciles

        # If we have a holdout
        if self.holdout:
            # Initialize a list for the each fold
            reconciled_df_list = []

            # Iterate over folds
            for k in range(self.cv):
                # Filter on the fold
                self.base_fc_df = self.original_df[self.original_df["cv"] == k + 1]
                self.reconciliation_ready_df = self.reconciled_df_list[k]
                temp_residual_df = residual_df[residual_df["cv"] == k + 1]

                # Get the Weight matrix
                self.Wmat = self.compute_matrix_W(
                    reconciliation_method, residual_df=temp_residual_df
                )

                # Reconciles
                self.reconciled_df = self.get_reconciled_predictions()

                # reverses the format
                self.reconciled_df = self.reverse_reconciliation_format(
                    reconciliation_method
                )

                # Add the cv
                self.reconciled_df["cv"] = k + 1

                # Append
                reconciled_df_list.append(self.reconciled_df)

            # concat
            self.reconciled_df = pd.concat(reconciled_df_list)

            # Add the true values
            true_vals_df = self.original_df.copy().drop(["y", "Model"], axis=1)

            # merge
            self.reconciled_df = self.reconciled_df.merge(
                true_vals_df, on=["unique_id", "temporal_level", "fh", "cv"], how="left"
            )

            return self.reconciled_df

        else:
            # Get the Weight matrix
            self.Wmat = self.compute_matrix_W(
                reconciliation_method, residual_df=residual_df
            )

            # Reconciles
            self.reconciled_df = self.get_reconciled_predictions()

            # reverses the format
            self.reconciled_df = self.reverse_reconciliation_format(
                reconciliation_method
            )

        return self.reconciled_df


class THieF(object):

    """
    A class for the Temporal Hierarcies Forecasting (THieF) algorithm.

    The flow of the class is the following:
    (1) Construct all temporal levels based on the bottom level frequency or the given factors
    (2) Fits a base forecasting model for each temporal level
    (3) Generates predictions for each level (base forecasts)
    (4) Reconciles base forecasts to get coherent predictions

    References:
        [1] https://www.sciencedirect.com/science/article/pii/S0377221717301911
        [2] https://github.com/cran/thief

    Args:
        bottom_level_freq (str):
            The frequency of the bottom level.
        factors (list, optional):
            The factors to use for the temporal levels.
        top_fh (int, optional):
            The top forecast horizon.
        base_model (str, optional):
            The base model to use.
        base_model_params (dict, optional):
            The parameters for the base model.
        base_model_fit_params (dict, optional):
            The fit parameters for the base model.
        base_model_predict_params (dict, optional):
            The predict parameters for the base model.
        reconciliation_method (str, optional):
            The reconciliation method to use.
        holdout (bool, optional):
            Whether to use holdout or not.
        cv (int, optional):
            The number of folds to use for holdout.


    Methods:
        fit:
            Fits the model on the given dataframe.
        predict:
            Predicts the base forecasts
        reconcile:
            Reconciles the base forecasts.
        get_residuals:
            Gets the residuals for the base forecasts.


    Examples:
        >>> from DeepRetail.reconciliation.temporal import THieF

        >>> # Initialize THieF
        >>> thief = thief = THieF(bottom_level_freq = bottom_level_freq)

        >>> # Fit thief
        >>> thief.fit(df, holdout = False, format = 'pivoted')

        >>> # Predict base forecasts
        >>> base_fc_df = thief.predict('ETS')

        >>> # Get residuals
        >>> residual_df = thief.base_forecast_residuals

        >>> # Reconcile base forecasts
        >>> reconciled_df = thief.reconcile('mse')

    """

    def __init__(
        self, bottom_level_freq, factors=None, top_fh=1, holdout=True, cv=None
    ):
        """
        Initializes the THieF class.
        It constructs the temporal levels and assigns fundamental parameters.
        For example the summation matrix S.

        Args:
            bottom_level_freq (str):
                The frequency of the bottom level.
            factors (list, optional):
                The factors to use for the temporal levels.
            top_fh (int, optional):
                The top forecast horizon.
            holdout (bool, optional):
                Whether to use holdout or not.
            cv (int, optional):
                The number of folds to use for holdout.

        Returns:
            None
        """
        # Ensure that either factors or bottom_freq is given
        # Raise an error otherwise
        if factors is None and bottom_level_freq is None:
            raise TypeError("Either factors or bottom_freq should be given")

        # Get the numeric frequency
        self.bottom_level_freq = bottom_level_freq
        self.bottom_level_numeric_freq = get_numeric_frequency(self.bottom_level_freq)

        # Construct all factors if they are not given
        if factors is None:
            factors = get_factors(self.bottom_level_numeric_freq)
            self.factors = factors
        else:
            self.factors = factors

        # Initialize extra parameters
        # Build the forecast horizons
        self.fhs = np.flip(factors) * top_fh
        self.frequencies = self.fhs.copy()
        self.m = np.flip(self.frequencies)
        self.max_freq = max(self.frequencies)
        self.total_levels = len(self.factors)

        # array indices for the flatten base forecasts
        self.level_indices = np.cumsum(np.insert(self.fhs, 0, 0))
        # self.level_indices = np.cumsum(self.fhs)

        # Get the resampled frequencies for upscaling the data
        self.resampled_factors = compute_resampled_frequencies(
            self.factors, self.bottom_level_freq
        )

        # Construct the Smat
        self.Smat = compute_matrix_S_temporal(self.factors)

        # add the holdout and the cv
        self.holdout = holdout
        self.cv = cv

    def fit(self, original_df, format="pivoted"):
        """
        Fits the model on the given dataframe.
        The functions prepares the input dataframe to the right format.

        Args:
            original_df (pd.DataFrame):
                The original dataframe.
            format (str, optional):
                The format of the input dataframe.
                It can be either 'pivoted' or 'transaction'.

        Returns:
            None

        """

        # In this method I build the hierarchy
        # I need to see how I will use the holdout and the cv paremeter
        self.original_df = original_df

        # If we have holdout:
        if self.holdout:
            # Initialize variables

            end_point = self.bottom_level_numeric_freq + self.cv - 1

            # Initialize lists for train and test
            self.resampled_train_dfs = []
            self.resampled_test_dfs = []

            for z in range(self.cv):
                # Manual cross-validation
                # Define start and end points for each fold
                temp_startpoint = end_point - z
                temp_endpoint = self.cv - z - 1

                # Split
                temp_test_df = (
                    self.original_df.iloc[:, -temp_startpoint:-temp_endpoint]
                    if temp_endpoint != 0
                    else self.original_df.iloc[:, -temp_startpoint:]
                )
                temp_train_df = self.original_df.iloc[:, :-temp_startpoint]

                # Resample train first
                temp_resampled_train_df = [
                    resample_temporal_level(temp_train_df, i, self.bottom_level_freq, j)
                    for i, j in zip(self.factors, self.resampled_factors)
                ]
                # Convert to dictionary with factors as keys
                self.resampled_dfs = {
                    self.factors[i]: temp_resampled_train_df[i]
                    for i in range(len(self.factors))
                }

                # add to list
                self.resampled_train_dfs.append(self.resampled_dfs)

                # Repeat for test set
                # I also melt to merge right away
                temp_resampled_test_df = [
                    (
                        resample_temporal_level(
                            temp_test_df, i, self.bottom_level_freq, j
                        )
                        .reset_index()
                        .melt(id_vars="unique_id", value_name="y_true", var_name="date")
                    )
                    for i, j in zip(self.factors, self.resampled_factors)
                ]
                # Add the factor as temporal level to each test dataframe
                for i in range(len(self.factors)):
                    temp_resampled_test_df[i]["temporal_level"] = self.factors[i]

                # Concat the list
                self.resampled_test_dfs.append(pd.concat(temp_resampled_test_df))

            # Add the cv to each test set
            for i in range(len(self.resampled_test_dfs)):
                self.resampled_test_dfs[i]["cv"] = i + 1

            # Concat
            self.resampled_test_dfs = pd.concat(self.resampled_test_dfs)

        # If not, generate a single split.
        else:
            # Get the list of the resampled dataframes
            resampled_dfs = [
                resample_temporal_level(self.original_df, i, self.bottom_level_freq, j)
                for i, j in zip(self.factors, self.resampled_factors)
            ]

            # convert it to a dictionary with the factors as keys
            self.resampled_dfs = {
                self.factors[i]: resampled_dfs[i] for i in range(len(self.factors))
            }

    def predict(self, models, to_return=True):
        """
        Generates base forecasts for each temporal level

        Args:
            models (str or dict):
                The models to use for each temporal level.
                It can be either a string or a dictionary.
                If it is a string, the same model will be used for all temporal levels.
                If it is a dictionary, the keys should be the temporal levels and the values the models.
            to_return (bool, optional):
                Whether to return the base forecasts or not.
                Default is True

        Returns:
            pd.DataFrame:
                The base forecasts for each temporal level.
        """

        # generates base forecasts
        # models is str or list dictionary for each factor
        # for example model_example = {1: ['ETS', 'Naive'], 3: 'ETS', 4: 'ARIMA', 6: 'ETS', 12: 'ETS'}

        # Check if we have a model for each level
        if isinstance(models, str):
            # If not, use the same model for all levels
            models = {i: models for i in self.factors}
        # Check if we have enough models
        elif len(models) != len(self.factors):
            raise ValueError(
                "The number of models should be equal to the number of factors"
            )

        # If we have holdout
        if self.holdout:
            # Initialize a list for the base forecasts and residuals
            temp_total_base_forecasts = []
            temp_total_residuals = []

            for z in range(self.cv):
                # Take the temporal hierarchy for the fold
                self.resampled_dfs = self.resampled_train_dfs[z]

                # Initialize a StatisticalForecaster for each factor
                # Currently only supporting StatisticalForecaster
                self.base_forecasters = {
                    factor: StatisticalForecaster(
                        models=models[factor], freq=self.resampled_factors[i]
                    )
                    for i, factor in enumerate(self.factors)
                }

                # Fit
                for factor in self.factors:
                    self.base_forecasters[factor].fit(
                        self.resampled_dfs[factor], format="pivoted"
                    )

                # Generate base forecasts
                temp_base_forecasts = {
                    factor: self.base_forecasters[factor].predict(
                        h=self.frequencies[i], holdout=False
                    )
                    for i, factor in enumerate(self.factors)
                }

                # Concat in a single dataframe
                self.base_forecasts = pd.concat(temp_base_forecasts, axis=0)

                # Reset index and drop column from multi-index
                # also rename the remaining index to get the temporal level
                self.base_forecasts = (
                    self.base_forecasts.reset_index()
                    .drop(columns="level_1")
                    .rename(columns={"level_0": "temporal_level"})
                )

                # Get residuals
                # NOTE: change the get_residual to accept as argument the df
                # This way I can have temp dataframes instead of self.resampled_dfs

                temp_residuals = self.get_residuals()

                # Add the cv to the predictions and the cv
                self.base_forecasts["cv"] = z + 1
                temp_residuals["cv"] = z + 1

                # Add to list
                temp_total_base_forecasts.append(self.base_forecasts)
                temp_total_residuals.append(temp_residuals)

            # Concat
            self.base_forecasts = pd.concat(temp_total_base_forecasts)
            self.base_forecast_residuals = pd.concat(temp_total_residuals)

            # Merge with the true
            self.base_forecasts = pd.merge(
                self.base_forecasts,
                self.resampled_test_dfs,
                on=["unique_id", "date", "temporal_level", "cv"],
                how="left",
            )

            # Return the predictions
            if to_return:
                return self.base_forecasts

        # If we don't have holdout
        else:
            # Initialize a StatisticalForecaster for each factor
            # Currently only supporting StatisticalForecaster
            self.base_forecasters = {
                factor: StatisticalForecaster(
                    models=models[factor], freq=self.resampled_factors[i]
                )
                for i, factor in enumerate(self.factors)
            }

            # Fit the forecasters
            for factor in self.factors:
                self.base_forecasters[factor].fit(
                    self.resampled_dfs[factor], format="pivoted"
                )

            # Generate base forecasts
            temp_base_forecasts = {
                factor: self.base_forecasters[factor].predict(
                    h=self.frequencies[i], holdout=False
                )
                for i, factor in enumerate(self.factors)
            }

            # Concat in a single dataframe
            self.base_forecasts = pd.concat(temp_base_forecasts, axis=0)

            # Reset index and drop column from multi-index
            # Also rename the remaining index to get the temporal level
            self.base_forecasts = (
                self.base_forecasts.reset_index()
                .drop("level_1", axis=1)
                .rename(columns={"level_0": "temporal_level"})
            )

            # Get the residuals here
            self.base_forecast_residuals = self.get_residuals()

        if to_return:
            return self.base_forecasts

    def get_residuals(self):
        """
        Estimates the residuals for each base forecaster on every temporal level.

        Args:
            None

        Returns:
            pd.DataFrame:
                The residuals for each base forecaster on every temporal level.
        """
        # Estimate residuals for all base forecasters
        temp_residuals = {
            factor: self.base_forecasters[factor].calculate_residuals()
            for factor in self.factors
        }

        # concat
        temp_residuals = pd.concat(temp_residuals, axis=0)

        # add to the right format
        temp_residuals = (
            temp_residuals.reset_index()
            .drop("level_1", axis=1)
            .rename(columns={"level_0": "temporal_level"})
        )

        # Calculate residuals
        # temp_residuals["residual"] = temp_residuals["y_true"] - temp_residuals["y_pred"]

        # keep only relevant columns
        to_keep = ["temporal_level", "unique_id", "cv", "fh", "Model", "residual"]

        # Add to the object
        # self.base_forecast_residuals = temp_residuals[to_keep]

        return temp_residuals[to_keep]

    def reconcile(self, reconciliation_method):
        """
        Reconciles base forecasts using the TemporalReconciler.

        Args:
            reconciliation_method (str):
                The method to use for reconciliation.
                Currently only supporting "struc".

        Returns:
            pd.DataFrame:
                The reconciled forecasts.

        """

        # Reconciles base predictions
        # Currently only supporting struc

        # Define the TemporalReconciler
        self.temporal_reconciler = TemporalReconciler(
            bottom_level_freq=self.bottom_level_freq,
            factors=self.factors,
            holdout=self.holdout,
            cv=self.cv,
        )

        # Fit the reconciler
        self.temporal_reconciler.fit(self.base_forecasts)

        # Reconcile
        self.reconciled_df = self.temporal_reconciler.reconcile(
            reconciliation_method, residual_df=self.base_forecast_residuals
        )

        # Merge with the base forecasts
        return self.reconciled_df
