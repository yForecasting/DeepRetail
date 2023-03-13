from DeepRetail.transformations.formats import (
    extract_hierarchical_structure,
    build_cross_sectional_df,
)
from DeepRetail.reconciliation.utils import (
    compute_matrix_S_cross_sectional,
    compute_y_tilde,
    shrink_estim,
    cross_product,
)
from DeepRetail.forecasting.statistical import StatisticalForecaster

import pandas as pd
import numpy as np


class CHieF(object):
    """
    A class for Cross-sectional Hierarchical Forecasting (CHieF) algorithm

    The flow of the class is the following:
    (1) Extract hierarchical infromation from a dataframe given the schema
    (2) Compute the summation matrix S
    (3) Extemd the dataframe to include all hierarchical levels.
        It also renames the names of the time series accordingly.
    (4) Fits a forecaster on the extended dataframe
    (5) Generates base forecasts for all levels
    (6) Reconciles the base forecasts to get coherent predictions

    Args:
        bottom_level_freq (str):
            The frequency of the time series
        h (int):
            The number of periods to forecast
        cv (int, optional):
            The number of folds for cross-validation.
            Defaults to None.
        holdout (bool, optional):
            Whether to use a holdout set.
            Defaults to False.
        df (pd.DataFrame):
            Dataframe with the original time series
        current_format (list):
            The current format of the schema on the unique_id
            For example ['top_level', 'middle_level', 'bottom_level']
        corrected_format (list):
            A list with the levels orders in the right order
            Lower levels are first.
            For example ['bottom_level', 'middle_level', 'top_level']
        splitter (str):
            The splitter used to separate the levels on the unique_id
            For example '_'
        add_total (bool, optional):
            Whether to add a total column to the dataframe.
            The higher level that aggregates everything.
            Defaults to True.
        format (str, optional):
            The format of the dataframe.
            Options are "pivoted" or "transaction".
            Defaults to "pivoted".
        models (list):
            A list with the models to use for producing base forecasts
        method (str):
            The method to use for reconciliation

        Methods:
            fit:
                Extracts the hierarchical structure and computes the S matrix.
                It also extends the dataframe to incldue all levels.
            predict:
                Generates base forecasts
            reconcile:
                Reconciles the generated base forecasts



        Examples:
            >>> from DeepRetail.reconciliation.cross_sectional import CHieF

            >>> # Initialize parameters
            >>> freq = 'M'
            >>> h = 12
            >>> models = ['ETS']

            >>> # Current format of the unique_id:
            >>> # cat_catnum_itemnum_state_storenum
            >>> # Example: Hobbies_1_1001_CA_1
            >>> # Define the parameters
            >>> current = ['cat', 'catnum', 'itemnum', 'state', 'storenum']
            >>> correct = ['itemnum', 'catnum', 'cat', 'storenum', 'state']
            >>> splitter = "_"
            >>> total = False

            >>> # Parameters for forecasting
            >>> holdout = True
            >>> cv = 1

            >>> # Define CHieF
            >>> chief = CHieF(bottom_level_freq = freq, h=h, holdout = holdout)

            >>> # Fit CHieF
            >>> chief.fit(
            >>>        df = sample_df, current_format=current,
            >>>        corrected_format=correct, splitter=splitter,
            >>>        add_total = total, format = 'pivoted'
            >>>        )

            >>> # Get base forecasts
            >>> base_forecasts = chief.predict(models = models)

            >>> # Define reconciliation method
            >>> method = 'var'

            >>> # reconcile
            >>> chief.reconcile(method = method)

    """

    def __init__(self, bottom_level_freq, h, cv=None, holdout=None):
        """
        Initializes the CHieF class
        Assigns some attributes to the object

        Args:
            bottom_level_freq (str):
                The frequency of the time series
            h (int):
                The number of periods to forecast
            cv (int, optional):
                The number of folds for cross-validation.
                Defaults to None.
            holdout (bool, optional):
                Whether to use a holdout set.

        Returns:
            None
        """
        self.bottom_level_freq = bottom_level_freq
        self.h = h
        self.cv = cv
        self.holdout = holdout
        # Repeat for temporal too -> move the holdout and the cv to the init.

    def fit(
        self,
        df,
        current_format,
        corrected_format,
        splitter,
        add_total=True,
        format="pivoted",
    ):
        """
        Fits the CHieF algorithm on the given dataframe.
        In particular the method: Extracts the hierarchical structure and computes the S matrix.
        It also extends the dataframe to incldue all levels

        Args:
            df (pd.DataFrame):
                Dataframe with the original time series
            current_format (list):
                The current format of the schema on the unique_id
                For example ['top_level', 'middle_level', 'bottom_level']
            corrected_format (list):
                A list with the levels orders in the right order
                Lower levels are first.
                For example ['bottom_level', 'middle_level', 'top_level']
            splitter (str):
                The splitter used to separate the levels on the unique_id
                For example '_'
            add_total (bool, optional):
                Whether to add a total column to the dataframe.
                The higher level that aggregates everything.
                Defaults to True.
            format (str, optional):
                The format of the dataframe.
                Options are "pivoted" or "transaction".
                Defaults to "pivoted".

        Returns:
            None
        """
        # add to the object
        self.original_df = df
        self.format = format

        # Extract the hierarchical structure
        self.hierarchical_format = extract_hierarchical_structure(
            self.original_df,
            current_format,
            corrected_format,
            splitter,
            add_total,
            self.format,
        )

        # Compute the S matrix
        self.S_mat = compute_matrix_S_cross_sectional(self.hierarchical_format)

        # Build the new dataframe with the cross-sectional format
        self.cross_sectional_df = build_cross_sectional_df(
            self.original_df, self.hierarchical_format, format=self.format
        )

    def predict(self, models):
        """
        Computes base forecasts for all hierarchical levels given the selected model
        In newer versions, user will select which model they want for every level

        Args:
            models (list):
                A list with the models to use for producing base forecasts

        Returns:
            base_forecasts (pd.DataFrame):
                A dataframe with the base forecasts

        """
        self.base_models = models

        # Define the forecaster
        self.base_forecaster = StatisticalForecaster(
            models=models, freq=self.bottom_level_freq
        )

        # Fit the forecaster
        self.base_forecaster.fit(self.cross_sectional_df, format="pivoted")

        # Make base predictions
        self.base_forecasts = self.base_forecaster.predict(
            h=self.h, cv=self.cv, holdout=self.holdout
        )

        return self.base_forecasts

    def reconcile(self, method):
        """
        Reconciles the generated base forecasts using the CrossSectionalReconciler

        Args:
            method (str):
                The method to use for reconciliation

        Returns:
            reconciled_forecasts (pd.DataFrame):
                The reconciled forecasts

        """

        # Define the reconciler
        self.reconciler = CrossSectionalReconciler(
            self.bottom_level_freq, self.h, cv=self.cv, holdout=self.holdout
        )

        # Fit the reconciler
        self.reconciler.fit(self.base_forecasts, self.S_mat)

        # If we have method that needs residuals
        if method in ["var", "sam", "shrink", "mse"]:
            # estimate the residuals
            self.residuals = self.base_forecaster.calculate_residuals()
        else:
            self.residuals = None

        # Reconcile
        self.reconciled_forecasts = self.reconciler.reconcile(
            method, residual_df=self.residuals
        )

        return self.reconciled_forecasts


class CrossSectionalReconciler(object):
    """
    A class for Cross-sectional reconciliation
    Reconciles base forecasts of different hierarchical levels.

    Supports the following methods:

    Diagonal methods:
    - ols: Identity Reconcilation (Hyndman et al, 2011)
    - struc: Structural Scaling  (Athanasopouls et al, 2017)
    - var: Variance Scaling (Athanasopouls et al, 2017)
    - mse: In-sample MSE Scaling (Athanasopouls et al, 2017)

    Full covariance methods:
    - sam: Sample Covariance matrix (Wickramasuriya et al, 2019)
    - shrink: Shurnk Covariance matrix (Wickramasuriya et al, 2019)

    Others:
    custom: Uses a user-given reconciliation matrix G.

    Note: Full covariance methods are unstable. Prefer the others

    Args:
        bottom_level_freq (str):
            The frequency of the time series
        h (int):
            The number of periods to forecast
        cv (int, optional):
            The number of folds for cross-validation.
            Defaults to None.
        holdout (bool, optional):
            Whether to use a holdout set.
            Defaults to False.
        df (pd.DataFrame):
            Dataframe with base forecasts for each hierarchical level
        s_mat (pd.DataFrame):
            The summing matrix S matrix for the reconciliation
        method (str):
            The method to use for reconciliation.
            Supports the methods described above
        residual_df (pd.DataFrame, optional):
            The dataframe with the residuals of the base forecasts.
            Used for methods using the residuals.
            Defaults to None.

        Methods:
            fit: Fits the reconciler on the given hierarchialy structured dataframe
            reconcile: Reconciles the base forecasts

        Examples:
            >>> # Import the reconciler
            >>> from DeepRetail.reconciliation.cross_sectional import CrossSectionalReconciler
            >>> from DeepRetail.forecasting.statistical import StatisticalForecaster

            >>> # Initialize some paramters
            >>> holdout = True
            >>> cv = 2
            >>> reconciliation_method = 'struc'
            >>> freq = 'M'
            >>> h = 12
            >>> models = ['ETS']

            >>> # Ensure we have the s_mat and a dataframe with all hierarchical levels

            >>> # Define a forecaster
            >>> forecaster = StatisticalForecaster(models = models, freq = freq)

            >>> # Fit the forecaster
            >>> forecaster.fit(hierarchical_df, format = 'pivoted')

            >>> # predict
            >>> base_forecasts = forecaster.predict(h = h, cv = cv, holdout = holdout)

            >>> # Get the residuals
            >>> residuals = forecaster.calculate_residuals()

            >>> # Define the reconciler
            >>> reconciler = CrossSectionalReconciler(bottom_level_freq=freq, h = h, holdout = holdout, cv = cv)

            >>> # fit the reconciler
            >>> reconciler.fit(df = base_forecasts, s_mat = s_mat)

            >>> # reconcile
            >>> method = 'var' # variance scalling
            >>> reconciled_forecasts = reconciler.reconcile(method = method, residual_df = residuals)


        References:
        - Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G., Shang, H.L.(2011),
                Optimal combination forecasts for hierarchical time series,
                Computational Statistics & Data Analysis, 55, 9, 2579-2589.
        - Athanasopoulos, G., Hyndman, R. J., Kourentzes, N., & Petropoulos, F. (2017).
                Forecasting with temporal hierarchies.
                European Journal of Operational Research, 262(1), 60–74.
        - Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019).
                Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization.
                Journal of the American Statistical Association, 114(526), 804–819.

    """

    def __init__(self, bottom_level_freq, h, cv=None, holdout=False):
        """
        Initializes the Cross-sectional reconciler

        Args:
            bottom_level_freq (str):
                The frequency of the time series
            h (int):
                The number of periods to forecast
            cv (int, optional):
                The number of folds for cross-validation.
                Defaults to None.
            holdout (bool, optional):
                Whether to use a holdout set.
                Defaults to False.

        Returns:
            None
        """
        self.bottom_level_freq = bottom_level_freq
        self.h = h
        self.cv = cv
        self.holdout = holdout

    def fit(self, df, s_mat):
        """
        Fits the reconciler on the given hierarchialy structured dataframe

        Args:
            df (pd.DataFrame):
                The hierarchical dataframe. Lower hierarchical levels are first.
            s_mat (pd.DataFrame):
                The S matrix for the reconciliation

        Returns:
            None

        """

        # here adjust for holdout!
        self.original_df = df
        self.S_mat = s_mat

        # if we have a holdout
        if self.holdout:
            # Initialize a list to include a df for every fold
            self.reconciliation_ready_cv_dfs = []

            # Itterate over the folds
            for fold in range(self.cv):
                # Filter the df
                temp_df = self.original_df[self.original_df["cv"] == fold + 1]
                # prepare
                temp_df = self.get_reconciliation_format(temp_df)
                # Append
                self.reconciliation_ready_cv_dfs.append(temp_df)
        else:
            # Prepare
            self.reconciliation_ready_df = self.get_reconciliation_format(
                self.original_df
            )

    def reconcile(self, method, residual_df=None, Gmat=None):
        """
        Reconciles base forecasts using the given method

        Args:
            method (str):
                The method to use for reconciliation.
                Supports the methods described in class documentation
            residual_df (pd.DataFrame, optional):
                The dataframe with the residuals of the base forecasts.
                Used for methods using the residuals.
                Defaults to None.
            Gmat (np.array, optional):
                A custom G matrix for reconciliation.
                Used when method = 'custom'.

        Returns:
            pd.DataFrame: The reconciled forecasts

        """

        self.reconciliation_method = method
        self.residual_df = residual_df

        # get the w matrix
        if self.reconciliation_method != "custom":
            self.W_mat = self.compute_w_matrix()
        elif self.reconciliation_method == "custom":
            # ensure Gmat is given
            if Gmat is None:
                raise ValueError(
                    "When using the custom method, you need to provide a custom G matrix"
                )

        # Extract the values from the smat
        S_mat_vals = self.S_mat.values

        # If we have a holdout
        if self.holdout:
            # Initialize a list to include the reconciled dataframes for every fold
            self.reconciled_cv_dfs = []
            self.G_mats = []

            # Itterate over the folds
            for fold in range(self.cv):
                # Extract the values from base forecasts
                y_hat_vals = self.reconciliation_ready_cv_dfs[fold].values

                # Compute the reconciled forecasts
                self.y_tild_vals, G_mat = compute_y_tilde(
                    y_hat_vals, S_mat_vals, self.W_mat, Gmat=Gmat, return_G=True
                )

                # take the fold on the original df
                temp_original_df = self.original_df[self.original_df["cv"] == fold + 1]
                # drop the cv column
                temp_original_df = temp_original_df.drop(columns=["cv"])

                # Give the right format
                self.reconciled_df = self.reverse_reconciliation_format(
                    self.reconciliation_ready_cv_dfs[fold], temp_original_df
                )

                # Include the fold
                self.reconciled_df["cv"] = fold + 1

                # Append
                self.reconciled_cv_dfs.append(self.reconciled_df)
                self.G_mats.append(G_mat)

            # Concatenate
            self.reconciled_df = pd.concat(self.reconciled_cv_dfs, axis=0)

        else:
            # Extract the values from base forecasts
            y_hat_vals = self.reconciliation_ready_df.values

            # Compute the reconciled forecasts
            self.y_tild_vals, self.G_mats = compute_y_tilde(
                y_hat_vals, S_mat_vals, self.W_mat, Gmat=Gmat, return_G=True
            )

            # Give the right format
            self.reconciled_df = self.reverse_reconciliation_format(
                self.reconciliation_ready_df, self.original_df
            )

        # Return
        return self.reconciled_df

    def get_reconciliation_format(self, temp_df):
        """
        Prepares the dataframe for reconciliation

        Args:
            df (pd.DataFrame):
                The dataframe to prepare

        Returns:
            temp_df (pd.DataFrame):
                The dataframe in the format required for reconciliation
        """

        # Pivot
        temp_df = pd.pivot_table(
            temp_df,
            index="unique_id",
            columns="fh",
            values="y",
            aggfunc="first",
        )
        # order the index based on the smat
        temp_df = temp_df.reindex(self.S_mat.index)

        return temp_df

    def prepare_residuals_matrix(self):
        """
        Conversts the dataframe with the residual in the appropriate format for the reconciliation

        Args:
            None

        Returns:
            res_mat (np.array): the matrix with the residuals
        """
        # Get a copy
        residual_df = self.residual_df.copy()

        # Filter
        residual_df = residual_df[residual_df["fh"] == 1]

        # Aggregate -> removed for now
        # residual_df = residual_df.groupby(['unique_id', 'fh']).agg({'residual': 'mean'}).reset_index()

        # pivot
        residual_df = pd.pivot_table(
            residual_df,
            index="unique_id",
            columns="cv",
            values="residual",
            aggfunc="first",
        )

        # Take the values only
        res_vals = residual_df.values

        # transpose
        res_vals = res_vals.T

        # return
        return res_vals

    def compute_w_matrix(self):
        """
        Computes the W matrix with the weights for the reconciliation

        Args:
            None

        Returns:
            pd.DataFrame:
                The W matrix
        """
        # Initiate two lists to cluster methods
        res_methods = ["var", "sam", "shrink", "mse"]

        if self.reconciliation_method in res_methods:
            # If we dont have residuals
            if self.residual_df is None:
                raise ValueError("Residuals are needed for this method")

            # Prepare the residuals
            res_mat = self.prepare_residuals_matrix()

            # Get the covariance matrix
            # taking the ma.cov to ignore the nan values
            cov_mat = np.ma.cov(res_mat, rowvar=False).data

            # Generate the W matrix for every method

            # Variance scalling
            if self.reconciliation_method == "var":
                # Get the diagonal
                W_mat = np.diag(np.diag(cov_mat))
                # Take the reciprocal
                W_mat = 1 / W_mat
                # Replaces nans withs 0s
                W_mat = np.nan_to_num(W_mat, posinf=0.0, neginf=0.0)

            # Sample covariance mat
            elif self.reconciliation_method == "sam":
                # cross product of the residuals
                W_mat = cross_product(res_mat) / res_mat.shape[0]

            # Shrinkage covariance method
            elif self.reconciliation_method == "shrink":
                W_mat, _ = shrink_estim(res_mat, cov_mat)  # _ is for lambda

            # MSE method
            elif self.reconciliation_method == "mse":
                # Mean squared in-sample error
                squared_mse = np.square(res_mat).mean(axis=0)
                # diag 1/w
                W_mat = np.diag(1 / squared_mse)
                # Replace nans with 0 due to zeroed residuals
                W_mat = np.nan_to_num(W_mat, nan=0.0, posinf=0.0, neginf=0.0)

        # for non residual methods
        # ols
        elif self.reconciliation_method == "ols":
            W_mat = np.eye(self.S_mat.shape[0])

        # Structural scaling
        elif self.reconciliation_method == "struc":
            if self.S_mat is None:
                raise ValueError("S matrix is needed for structural scaling")
            # Get the diagonal
            W_mat = np.diag(1 / self.S_mat.sum(axis=1).values)
            # Replace nans with 0 due to potential zero weights
            W_mat = np.nan_to_num(W_mat, nan=0.0, posinf=0.0, neginf=0.0)

        else:
            raise ValueError("Reconciliation method not recognized")

        return W_mat

    def reverse_reconciliation_format(self, y_hat_df, original_df):
        """
        Converts the reconciled matrix to the original format

        Args:
            y_hat_df (pd.DataFrame):
                The matrix with the base forecasts prepared for reconciliation
            original_df (pd.DataFrame):
                The original dataframe before conversion to reconciliation format


        Returns:
            reconciled_df (pd.DataFrame):
                The reconciled dataframe in the original format

        """

        # convert the matrix to a dataframe
        # Using the format of the original dataframe converted to reconciliation format
        reconciled_df = pd.DataFrame(
            self.y_tild_vals, index=y_hat_df.index, columns=y_hat_df.columns
        )

        # melt
        reconciled_df = reconciled_df.reset_index(names="unique_id").melt(
            id_vars=["unique_id"], value_vars=y_hat_df.columns
        )

        # Rename
        reconciled_df = reconciled_df.rename(columns={"value": "y_pred"})

        # Add the true values and the base forecasts
        reconciled_df = reconciled_df.merge(original_df, on=["unique_id", "fh"]).rename(
            columns={"y": "y_base"}
        )

        # Add the name
        reconciled_df["Model"] = (
            "HR-" + self.reconciliation_method + "-" + reconciled_df["Model"]
        )

        return reconciled_df
