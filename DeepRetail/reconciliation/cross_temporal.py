import pandas as pd
import numpy as np
import warnings

from DeepRetail.forecasting.utils import get_numeric_frequency
from DeepRetail.reconciliation.cross_sectional import CHieF, CrossSectionalReconciler
from DeepRetail.reconciliation.temporal import THieF, TemporalReconciler
from DeepRetail.transformations.formats import hierarchical_to_transaction
from DeepRetail.reconciliation.utils import convert_offset_to_lower_freq, get_factors


# surpress warnings
warnings.filterwarnings("ignore")

class CrossTemporalReconciler(object):
    """
    A class for Cross-Temporal Reconciliation.
    Reconciles base forecasts of different temporal and hierarchical levels.
    Uses the methodology followed in Kourentzes & Athanasopoulos (2019).

    First, base forecasts at all cross-sectional levels are reconciled temporaly into y_temp_tild.
    Then for each temporal level i = 1, ..., k we define a CrossSectional Reconciler.
    From each CrossSectionalReconciler we extract the reconciliation matrix G.
    This returns a total of k reconciliation matrices G.
    Then we take the average of all matrices to estimate the reconciliation matrix G_bar = 1/k * sum(G_i).
    We use the reconciliation matrix G_bar to cross-sectionaly reconcile the temporaly reconciled forecasts y_temp_tild.
    Finaly we multiply with the temporal S matrix to get the cross-temporal reconciled forecasts.


    Supports the following reconciliation methods for both cross-sectional and temporal reconciliation:

    Diagonal methods:
    - ols: Identity Reconcilation (Hyndman et al, 2011)
    - struc: Structural Scaling  (Athanasopouls et al, 2017)
    - var: Variance Scaling (Athanasopouls et al, 2017)
    - mse: In-sample MSE Scaling (Athanasopouls et al, 2017)

    Full covariance methods:
    - sam: Sample Covariance matrix (Wickramasuriya et al, 2019)
    - shrink: Shurnk Covariance matrix (Wickramasuriya et al, 2019)

    Note: Full covariance methods and var are unstable. Prefer the others

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
        base_fc (pd.DataFrame):
            The base forecasts for each temporal level.
        cross_sectional_Smat (pd.DataFrame):
            The cross-sectional S matrix
        residual_df (pd.DataFrame, optional):
            The residuals of the base forecasts.
            Defaults to None.
        temporal_method (str):
            The method to use for temporal reconciliation.
            Currently Supports: mse and struc
            For details refer to the TemporalReconciler class
        cross_sectional_method (str):
            The method to use for cross-sectional reconciliation.
            Currently Supports: ols, struc, mse, var, shrink and sam
            For details refer to the CrossSectionalReconciler class
            Note: var, shrink and sam methods are unstable. Prefer the others

    Methods
        fit:
            Fits the Cross-Temporal reconciler on the base forecasts
        reconcile:
            Reconciles the base forecasts
    
    Returns
        reconciled_fc (pd.DataFrame):
            The reconciled forecasts for each temporal and cross-sectional level.


    Examples:
        >>> from DeepRetail.reconciliation.cross_temporal import CrossTemporalReconciler
        >>> from DeepRetail.reconciliation.cross_sectional import CHieF
        >>> from DeepRetail.reconciliation.temporal import THieF
        >>> from DeepRetail.transformations.formats import hierarchical_to_transaction


        >>> # Initialize arguments
        >>> current = ['cat', 'catnum', 'itemnum', 'storenum']
        >>> correct = ['itemnum', 'catnum', 'cat', 'storenum']
        >>> splitter = "_"
        >>> total = False 

        >>> # ARguments for the schema of the temporal hierarchy
        >>> freq = 'M'
        >>> h = 12

        >>> holdout= True
        >>> cv = 2

        >>> # Define CHieF
        >>> chief = CHieF(bottom_level_freq = freq, h=h, holdout = holdout, cv = cv)

        >>> # Fit CHieF -> builds the hierarchical format
        >>> chief.fit(df = sample_df, 
        >>>           current_format=current, 
        >>>           corrected_format=correct, 
        >>>           splitter=splitter, 
        >>>           add_total = total, 
        >>>           format = 'pivoted')

        >>> # Extract values
        >>> h_df, h_format, s_mat = chief.cross_sectional_df, chief.hierarchical_format, chief.S_mat

        >>> # Define THief
        >>> thief = THieF(bottom_level_freq = freq, holdout = holdout, cv = cv)

        >>> # fit thief to the entire hierarchical df
        >>> thief.fit(h_df, format = 'pivoted')

        >>> # predict base forecasts (also get residuals)
        >>> base_fc = thief.predict('ETS')
        >>> res = thief.base_forecast_residuals

        >>> # Extend the base forecasts and residuals to the cross-sectional levels
        >>> base_fc_extended = hierarchical_to_transaction(base_fc, h_format, format = 'transaction')
        >>> res_extended = hierarchical_to_transaction(res, h_format, format = 'transaction')

        >>> # Define the reconciler
        >>> cross_temporal_reconciler = CrossTemporalReconciler(bottom_level_freq = freq, 
        >>>                                                     h = h, 
        >>>                                                     holdout = holdout, 
        >>>                                                     cv = cv)

        >>> # fit
        >>> cross_temporal_reconciler.fit(base_fc_extended, cross_sectional_Smat = s_mat, residual_df = res_extended)

        >>> # Reconcile
        >>> rec = cross_temporal_reconciler.reconcile('struc', 'mse')


    References:
        -Kourentzes, N., & Athanasopoulos, G. (2019). 
        Cross-temporal coherent forecasts for Australian tourism. 
        Annals of Tourism Research, 75, 393–409. [DOI]

        -Di Fonzo T., Girolimetto D. (2021). 
        “Cross-temporal forecast reconciliation: Optimal combination method and heuristic alternatives.
        ” International Journal of Forecasting, (in press). doi: 10.1016/j.ijforecast.2021.08.004

        
    """

    def __init__(self, bottom_level_freq, h, holdout=False, cv=None):
        """
        Initializes the Cross-Temporal reconciler

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
        self.holdout = holdout
        if cv is None:
            self.cv = 1
        self.cv = cv

    def fit(self, base_fc, cross_sectional_Smat, residual_df=None):
        """
        Fits the Cross-Temporal reconciler on the base forecasts

        Args:
            base_fc (pd.DataFrame):
                The base forecasts for each temporal level.
            cross_sectional_Smat (pd.DataFrame):
                A dataframe with the S matrix for cross-sectional reconciliation
            residual_df (pd.DataFrame, optional):
                A dataframe with the residuals of the base forecasts.

        Returns:
            None
        """

        # Ensure we have the "temporal_level" and the cross_sectional_level columns
        if "temporal_level" not in base_fc.columns:
            raise ValueError(
                "The base forecasts should have the correspoding temporal level"
            )
        if "cross_sectional_level" not in base_fc.columns:
            raise ValueError(
                "The base forecasts should have the correspoding cross-sectional level"
            )

        # Add to the object
        self.base_fc = base_fc
        self.residual_df = residual_df
        self.cross_sectional_Smat = cross_sectional_Smat

    def reconcile(self, temporal_method, cross_sectional_method):
        """
        Reconciles base forecasts cross-temporaly using the given methods.

        Args:
            temporal_method (str):
                The method to use for temporal reconciliation.
                Currently Supports: mse and struc
                For details refer to the TemporalReconciler class
            cross_sectional_method (str):
                The method to use for cross-sectional reconciliation.
                Currently Supports: ols, struc, mse, var, shrink and sam
                For details refer to the CrossSectionalReconciler class

        Returns:
            reconciled_fc (pd.DataFrame):
                The reconciled forecasts for each temporal and cross-sectional level.

        """

        # First we reconcile temporally

        if self.holdout:
            # Initialize a list for every fold
            temporaly_reconciled_dfs = []

            # Loop through the folds
            for cv in range(1, self.cv + 1):
                # Filter base fcs and residuals on the fold
                base_fc_fold = self.base_fc[self.base_fc["cv"] == cv]
                # A hacky way to evade an error
                # replace the values at cv column with 1
                base_fc_fold = base_fc_fold.drop(columns=["cv"])
                base_fc_fold["cv"] = 1

                # repeat for residuals
                residual_df_fold = self.residual_df.copy()
                # if self.residual_df is not None:
                # residual_df_fold = self.residual_df[self.residual_df["cv"] == cv]
                # Repeat the hack
                # residual_df_fold = residual_df_fold.drop(columns=["cv"])
                # residual_df_fold["cv"] = 1
                # residual_df_fold = residual_df_fold.drop(columns=["cv"])

                # Reconcile temporally
                temporaly_reconciled = self.temporal_reconciliation(
                    base_fc_fold=base_fc_fold,
                    temporal_method=temporal_method,
                    residual_fold=residual_df_fold,
                )

                # Then we get the list of cross-sectional matrices G for each temporal level
                G_list = self.get_cross_sectional_G_matrices(
                    temporaly_reconciled,
                    cross_sectional_method,
                    residual_df=self.residual_df,
                )

                # We estimate G_bar, the average of the G matrices
                self.G_bar = np.mean(G_list, axis=0)[0]

                # We get the bottom level of the temporally reconciled predictions
                self.temporaly_reconciled_bottom = temporaly_reconciled[
                    temporaly_reconciled["temporal_level"]
                    == temporaly_reconciled["temporal_level"].min()
                ]

                # We reconcile cross-sectionally using G_bar
                # Define a cross-sectional reconciler
                self.cross_temporal_reconciler = CrossSectionalReconciler(
                    bottom_level_freq=self.bottom_level_freq,
                    h=self.h,
                    holdout=self.holdout,
                    cv=1,  # passing jsut one here
                )
                # fit the reconciler
                self.cross_temporal_reconciler.fit(
                    self.temporaly_reconciled_bottom, s_mat=self.cross_sectional_Smat
                )
                # Reconcile
                cross_sectionaly_reconciled = self.cross_temporal_reconciler.reconcile(
                    method="custom", Gmat=self.G_bar
                )

                # Finally we S-up to get the cross-temporally reconciled predictions.
                cross_temporaly_reconciled = self.S_up(
                    cross_sectionaly_reconciled, temporaly_reconciled
                )

                # We add the cv
                cross_temporaly_reconciled["cv"] = cv

                # We append to the list
                temporaly_reconciled_dfs.append(cross_temporaly_reconciled)

            # We concatenate the dfs
            self.cross_temporaly_reconciled = pd.concat(temporaly_reconciled_dfs)

        else:
            # Reconcile temporally
            temporaly_reconciled = self.temporal_reconciliation(
                base_fc_fold=self.base_fc,
                temporal_method=temporal_method,
                residual_fold=self.residual_df,
            )

            # Then we get the list of cross-sectional matrices G for each temporal level
            G_list = self.get_cross_sectional_G_matrices(
                temporaly_reconciled,
                cross_sectional_method,
                residual_df=self.residual_df,
            )

            # We estimate G_bar, the average of the G matrices
            G_bar = np.mean(G_list, axis=0)

            # We get the bottom level of the temporally reconciled predictions
            self.temporaly_reconciled_bottom = temporaly_reconciled[
                temporaly_reconciled["temporal_level"]
                == temporaly_reconciled["temporal_level"].min()
            ]

            # We reconcile cross-sectionally using G_bar
            # Define a cross-sectional reconciler
            self.cross_temporal_reconciler = CrossSectionalReconciler(
                bottom_level_freq=self.bottom_level_freq,
                h=self.h,
                holdout=self.holdout,
                cv=self.cv,
            )
            # fit the reconciler
            self.cross_temporal_reconciler.fit(
                self.temporaly_reconciled_bottom, s_mat=self.cross_sectional_Smat
            )
            # Reconcile
            self.cross_sectionaly_reconciled = self.cross_temporal_reconciler.reconcile(
                method="custom", Gmat=G_bar
            )

            # Finally we S-up to get the cross-temporally reconciled predictions.
            self.cross_temporaly_reconciled = self.S_up(
                self.cross_sectionaly_reconciled, temporaly_reconciled
            )

        return self.cross_temporaly_reconciled

    def temporal_reconciliation(
        self, base_fc_fold, temporal_method, residual_fold=None
    ):
        """
        Reconciles temporally for each cross-sectional level.

        Args:
            base_fc_fold (pd.DataFrame):
                The base forecasts for each temporal and cross-sectional level.
            temporal_method (str):
                The method to use for temporal reconciliation.
            residual_fold (pd.DataFrame):
                The residuals for each temporal and cross-sectional level.

        Returns:
            pd.DataFrame:
                The temporally reconciled forecasts.

        """

        # Extract the cross-sectional levels
        cross_sectional_levels = base_fc_fold["cross_sectional_level"].unique()

        # Initialize a dataframe
        temporaly_reconciled = pd.DataFrame()

        # Iterate through the cross-sectional levels
        for level in cross_sectional_levels:
            # filter on the level
            base_fc_cross_section_level = base_fc_fold[
                base_fc_fold["cross_sectional_level"] == level
            ]

            # I should also filter the residuals here if given
            if residual_fold is not None:
                residual_cross_section_level = residual_fold[
                    residual_fold["cross_sectional_level"] == level
                ]

            # Sort based on the temporal level and the fh
            base_fc_cross_section_level = base_fc_cross_section_level.sort_values(
                by=["temporal_level", "fh"], ascending=[False, True]
            )

            # Define a reconciler
            temp_cv = (
                1 if self.holdout else None
            )  # We dont want multiple cvs here we iterate outside the function
            self.dummy_temporal_reconciler = TemporalReconciler(
                bottom_level_freq=self.bottom_level_freq,
                holdout=self.holdout,
                cv=temp_cv,
            )

            # Fit
            self.dummy_temporal_reconciler.fit(base_fc_cross_section_level)

            # Reconcile
            temp_temporaly_reconciled = self.dummy_temporal_reconciler.reconcile(
                temporal_method, residual_df=residual_cross_section_level
            )

            # add the level
            temp_temporaly_reconciled["cross_sectional_level"] = level

            # Concat
            temporaly_reconciled = pd.concat(
                [temporaly_reconciled, temp_temporaly_reconciled]
            )

            # Assigns the smat to the object for later use
            self.temporal_Smat = self.dummy_temporal_reconciler.Smat

        # return
        return temporaly_reconciled

    def get_cross_sectional_G_matrices(
        self, temporaly_reconciled, cross_sectional_method, residual_df=None
    ):
        """
        Estimates the G matrix used for cross-sectional reconciliation on each temporal level.

        Args:
            temporaly_reconciled (pd.DataFrame):
                The temporally reconciled forecasts.
            cross_sectional_method (str):
                The method to use for cross-sectional reconciliation.
            residual_df (pd.DataFrame):
                The residuals for each temporal and cross-sectional level.

        Returns:
            G_list (list):
                A list of G matrices for each temporal level.

        """

        # Extract the temporal levels
        temporal_levels = temporaly_reconciled["temporal_level"].unique()

        # Initialize a list to keep the G matrices
        G_list = []

        # Iterate over the temporal levels
        for level in temporal_levels:
            # filter
            base_fc_temporal_level = temporaly_reconciled[
                temporaly_reconciled["temporal_level"] == level
            ]

            # I should also filter the residuals here if given
            if residual_df is not None:
                residual_df_temporal_level = residual_df[
                    residual_df["temporal_level"] == level
                ]

            # Take the bottom_level and the forecast horizon h
            temp_bottom_level = convert_offset_to_lower_freq(
                str(level) + self.bottom_level_freq
            )
            temp_h = len(base_fc_temporal_level["fh"].unique())
            temp_cv = (
                1 if self.holdout else None
            )  # We dont want multiple cvs here we iterate outside the function
            # Define a reconciler
            dummy_cross_sectional_reconciler = CrossSectionalReconciler(
                bottom_level_freq=temp_bottom_level,
                h=temp_h,
                holdout=self.holdout,
                cv=temp_cv,
            )

            # Fit the reconciler.
            # We use the same s mat for all levels
            dummy_cross_sectional_reconciler.fit(
                base_fc_temporal_level, s_mat=self.cross_sectional_Smat
            )
            # Reconcile
            _ = dummy_cross_sectional_reconciler.reconcile(
                method=cross_sectional_method,
                residual_df=residual_df_temporal_level,
            )

            # Extract the G matrix
            temp_G = dummy_cross_sectional_reconciler.G_mats

            # Add the matrix to the list
            G_list.append(temp_G)

        # Return
        return G_list

    def S_up(self, bottom_level_cross_sectional, total_reconciled_temporal):
        """
        Multiples with the temporal Smatrix to get the cross-temporally reconciled predictions

        Args:
            bottom_level_cross_sectional (pd.DataFrame):
                The cross-sectional forecasts for the bottom level.
            total_reconciled_temporal (pd.DataFrame):
                The temporally reconciled forecasts.

        Returns:
            temporaly_reconciled_df (pd.DataFrame):
                The cross-temporally reconciled forecasts for the entire hierarchy


        """

        # Prepare base forecasts for merging
        # Prepare the base forecasts
        # Get the temp indexer
        total_reconciled_temporal["temp_indexer"] = (
            total_reconciled_temporal["temporal_level"].astype(str)
            + "_"
            + total_reconciled_temporal["fh"].astype(str)
        )
        # Rename the column from y to indicate temporal reconciliation
        total_reconciled_temporal = total_reconciled_temporal.rename(
            columns={"y": "y_rec_temporal"}
        )

        # Prepare for S-UP
        # Pivot
        temp_pivot = pd.pivot_table(
            bottom_level_cross_sectional,
            values="y_pred",
            index="unique_id",
            columns="fh",
            aggfunc="first",
        )

        # Order columns
        temp_pivot = temp_pivot.reindex(sorted(temp_pivot.columns), axis=1)

        # S_up with the S matrix from temporal reconciliation
        temporaly_reconciled_values = np.array(
            [self.temporal_Smat @ ts for ts in temp_pivot.values]
        )

        # Convert to dataframe
        temporaly_reconciled_df = pd.DataFrame(
            temporaly_reconciled_values,
            index=temp_pivot.index,
            columns=total_reconciled_temporal["temp_indexer"].unique(),
        )

        # Melt
        temporaly_reconciled_df = temporaly_reconciled_df.reset_index().melt(
            id_vars="unique_id",
            var_name="temp_indexer",
            value_name="y_rec_cross_temporal",
        )

        # Merge with temp_df
        temporaly_reconciled_df = pd.merge(
            total_reconciled_temporal,
            temporaly_reconciled_df,
            on=["unique_id", "temp_indexer"],
        )

        # Return
        return temporaly_reconciled_df
