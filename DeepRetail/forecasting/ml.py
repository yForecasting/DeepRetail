import pandas as pd
import numpy as np
from DeepRetail.forecasting.utils import (
    add_missing_values,
    create_lags,
    construct_single_rolling_feature,
    split_lag_targets,
    standard_scaler_custom,
)
from DeepRetail.transformations.formats import pivoted_df


class GlobalForecaster(object):
    def __init__(self, model, features, model_name, transformations=None):
        # Fills the features and transformations dictionaries with default values.
        features, transformations = add_missing_values(features, transformations)

        self.model = model
        self.features = features
        self.transformations = transformations
        self.model_name = model_name

        # ensure lags are not None
        if self.features["lags"] is None:
            raise ValueError("Lags cannot be None")

        # Include the target horizon on the lags
        self.fixed_lags = self.features["lags"] + 1

        # dates = input_df.columns.values
        # total_windows = len(dates) - features['lags']

    def fit(self, df, test_size=None, covariates=None, format="pivoted"):
        if format == "transaction":
            df = pivoted_df(df)
        elif format == "pivoted":
            pass
        else:
            raise ValueError('format must be either "transaction" or "pivoted"')

        # Fits the and trains the model
        # Keep the original df for later to build the predictions.
        self.input_df = df

        # Keep the test set asside
        if test_size is not None:
            self.fit_df = df.iloc[:, :-test_size]
        else:
            self.fit_df = df

        # Create the lagged dataframe
        self.lag_df = create_lags(self.fit_df, self.fixed_lags)

        # Split x_train and y_train
        self.lag_df = split_lag_targets(self.lag_df)

        # drop the lag_windows
        self.lag_df = self.lag_df.drop("lag_windows", axis=1)

        # Drop rows that have empty lists as lagged windows
        self.lag_df[self.lag_df["lagged_values"].apply(lambda x: len(x) > 0)]

        # Create the rolling features
        if self.features["rolling_features"] is not None:
            rolling_df = self.build_rolling_features(self.fit_df)

            # Merge the two dfs
            self.lag_df = pd.concat([self.lag_df, rolling_df], axis=1)

        # Perform transformations if given
        if self.transformations["normalize"] is not None:
            if self.transformations["normalize"] == "StandardScaler":
                # currently we only support standrad scaler
                self.lag_df = standard_scaler_custom(self.lag_df)

        # Next we concat all features into the x_train and y_train arrays
        if (
            "normalized_lagged_values" in self.lag_df.columns
        ):  # I might change the name here to "transformed_lagged_values"
            # If we have rolling features
            rolling_columns = [col for col in self.lag_df.columns if "normalized_rolling" in col]
            if len(rolling_columns) > 0:
                # add them to feature list
                for rolling_col in rolling_columns:
                    # add the elements on the lists of the rolling feature column
                    # to the list of the normalized lagged values elementwise
                    self.lag_df["normalized_lagged_values"] = self.lag_df.apply(
                        lambda x: [a + b for a, b in zip(x["normalized_lagged_values"], x[rolling_col])],
                        axis=1,
                    )
                # Drop the rolling columns if memory gets large
                # lag_df.drop(rolling_columns, axis = 1, inplace = True)
            x_train = np.concatenate(self.lag_df["normalized_lagged_values"].values)
            y_train = np.concatenate(self.lag_df["normalized_targets"].values)
        else:
            # Repeat for non-normalized elements
            # If we have rolling features
            rolling_columns = [col for col in self.lag_df.columns if "rolling" in col]
            if len(rolling_columns) > 0:
                # add them to the feature list
                for rolling_col in rolling_columns:
                    # add the elements on the lists of the rolling feature column
                    # to the list of the normalized lagged values elementwise
                    self.lag_df["lagged_values"] = self.lag_df.apply(
                        lambda x: [a + b for a, b in zip(x["lagged_values"], x[rolling_col])],
                        axis=1,
                    )
                # Drop the rolling columns
                # lag_df.drop(rolling_columns, axis = 1, inplace = True)
            # if not take tha lagged values
            x_train = np.concatenate(self.lag_df["lagged_values"].values)
            y_train = np.concatenate(self.lag_df["targets"].values)

        # Fit the model.
        self.model.fit(x_train, y_train.ravel())

    def predict(self, h, cv=None):
        if cv is None:
            cv = 1

        self.h = h
        self.cv = cv

        # Create the dataframe for predictions
        self.pred_df = self.create_cv_df(self.input_df, self.h, self.cv)

        # Forecasts
        self.pred_df["forecasts"] = self.recurent_forecast(self.model, self.pred_df, self.h)

        # Unstack the dataframe and give the right format
        self.pred_df = self.unstack_dataframe(self.pred_df)

        # return
        return self.pred_df

    def build_rolling_features(self, df):
        # Initialize a df to include all rolling features
        rolling_df = pd.DataFrame()
        # take the number of lags
        lags = self.features["lags"]

        # Itterate over the rolling features
        for rolling_aggregation, rolling_windows in self.features["rolling_features"].items():
            # Take the rolling lags
            rolling_lags = self.features["rolling_lags"][rolling_aggregation]
            # Construct the rolling features
            temp_df = construct_single_rolling_feature(df, rolling_aggregation, lags, rolling_windows, rolling_lags)
            # Append to the main df
            rolling_df = pd.concat([rolling_df, temp_df], axis=1)

        # return
        # round the whole df
        return rolling_df

    def create_cv_df(self, df, h, cv):
        """
        Prepares the dataframe for predictions
        Expands the current dataframe to account for every fold on cv by adding a new row.


        Args:
            df (pd.DataFrame):
                The input dataframe
            h (int):
                The number of steps to forecast
            cv (int):
                The number of folds for cross validation

        Returns:
            out_df (pd.DataFrame):
                The output dataframe with the expanded rows for every fold on cv.

        """

        # Initialize a dataframe
        out_df = pd.DataFrame()
        # Define the total out-of-sample length
        total_test_length = h + cv - 1
        # take the dates
        dates = df.columns

        for i in range(cv):
            # Initialize an empty df
            temp_df = pd.DataFrame(index=df.index)

            # take the cutoff date
            temp_df["cutoff"] = dates[-(total_test_length + 1) + i]

            # add the cv
            temp_df["cv"] = i + 1

            # Cut on the last_date and add the in-sample values for the train set in a list
            temp_df["in_sample"] = df.iloc[:, i : -total_test_length + i].values.tolist()  # noqa: E203

            # Repeat for the test set
            # if it is not the last cv
            if i != cv - 1:
                temp_df["out_of_sample"] = df.iloc[
                    :,
                    -total_test_length + i : -(total_test_length - h) + i,  # noqa: E203
                ].values.tolist()
                # add the forecast dates
                temp_df["forecast_dates"] = (
                    np.tile(
                        dates[
                            -total_test_length + i : -(total_test_length - h) + i  # noqa: E203  # noqa: E203
                        ].tolist(),
                        df.shape[0],
                    )
                    .reshape(df.shape[0], -1)
                    .tolist()
                )
            else:
                # just take the last h values here
                temp_df["out_of_sample"] = df.iloc[:, -h:].values.tolist()
                # add the forecast dates
                temp_df["forecast_dates"] = np.tile(dates[-h:].tolist(), df.shape[0]).reshape(df.shape[0], -1).tolist()

            # Append the temp_df to the new_df
            out_df = pd.concat([out_df, temp_df])

        # Add the forecast horizons
        out_df["horizon"] = np.tile(np.arange(1, h + 1), df.shape[0] * cv).reshape(df.shape[0] * cv, -1).tolist()

        return out_df

    def prepare_row_for_forecast(self, pred_values):
        """
        Prepares the input for a single 1-step ahead prediction.

        Args:
            pred_values (pd.DataFrame):
                A dataframe containing the in-sample values and the out-of-sample values.
                It is constructed with the create_cv_df function

        Returns:
            pred_values (pd.DataFrame):
                The dataframe with the lagged values and the rolling features.

        """
        # Lag values
        pred_values["lagged_values"] = pred_values["in_sample"].apply(
            lambda x: np.array(x[-self.features["lags"] :]).reshape(1, -1)  # noqa: E203
        )

        # Rolling features
        if self.features["rolling_features"] is not None:
            # take the in-sample values
            in_sample_df = pd.DataFrame(pred_values["in_sample"].values.tolist(), index=pred_values.index)
            # construct features
            rolling_df = self.build_rolling_features(in_sample_df)
            # for every rolling feature we need to take the last item of the list on each row
            for col in rolling_df.columns:
                rolling_df[col] = rolling_df[col].apply(lambda x: [x[-1]])
            pred_values = pd.concat([pred_values, rolling_df], axis=1)

        # 4. Normalization transformation if passed
        if self.transformations["normalize"] is not None:
            # currently I only have standard scaler
            if self.transformations["normalize"] == "StandardScaler":
                pred_values = standard_scaler_custom(pred_values, mode="test")

        return pred_values

    def recurent_forecast(self, model, df, h):
        """
        Does autoregressive forecasting for h steps ahead.

        Args:
            model (sklearn-type-model):
                The model to be used for forecasting. It should be fitted
            df (pd.DataFrame):
                The dataframe containing the in-sample values and the out-of-sample values.
                It is constructed with the create_cv_df function
            h (int):
                The number of steps to forecast
            features (dict):
                A dictionary containing the features to be used for the prediction.
            transformations (dict):
                A dictionary containing the transformations to be applied to the data.

        Returns:
            forecasts (np.array):
                The array containing the forecasts for h steps ahead.
        """

        # features and transformations will be dictionaries from the forecaster

        # here expands transformations and featuures from dictionaries
        # booleans and features

        # Initialize matrix with forecasts
        forecasts = np.zeros((df.shape[0], h))

        # Initialize the prediction df
        prediction_df = df.copy()

        for i in range(h):
            # function to prepare test row
            # Prepare the df for forecasting
            prediction_df = self.prepare_row_for_forecast(prediction_df)
            # take the values for forecasting
            if "normalized_lagged_values" in prediction_df.columns:
                # If we have rolling features
                rolling_columns = [col for col in prediction_df.columns if "normalized_rolling" in col]
                if len(rolling_columns) > 0:
                    for rolling_col in rolling_columns:
                        # add the elements on the lists of the rolling feature column
                        # to the list of the normalized lagged values elementwise
                        prediction_df["normalized_lagged_values"] = prediction_df.apply(
                            lambda x: [a + b for a, b in zip(x["normalized_lagged_values"], x[rolling_col])],
                            axis=1,
                        )
                    # Drop the rolling columns if memory gets large
                    prediction_df = prediction_df.drop(rolling_columns, axis=1)
                x_test = np.concatenate(
                    prediction_df["normalized_lagged_values"].values
                )  # will change that to transformed_lagged_values
            else:
                # if we have rolling features
                rolling_columns = [col for col in prediction_df.columns if "rolling" in col]
                if len(rolling_columns) > 0:
                    for rolling_col in rolling_columns:
                        # add the elements on the lists of the rolling feature column
                        # to the list of the normalized lagged values elementwise
                        prediction_df["lagged_values"] = prediction_df.apply(
                            lambda x: [a + b for a, b in zip(x["lagged_values"], x[rolling_col])],
                            axis=1,
                        )
                    # Drop the rolling columns
                    prediction_df = prediction_df.drop(rolling_columns, axis=1, inplace=True)
                x_test = np.concatenate(prediction_df["lagged_values"].values)

            # Forecast
            y_pred = model.predict(x_test)

            # Reverse transformations ex.stationarity
            # ...

            # Add forecasts to df
            # if we have normalized reverse the transformation
            if "normalized_lagged_values" in prediction_df.columns:
                # if we have standard scaler
                if self.transformations["normalize"] == "StandardScaler":
                    # reverse normalization
                    y_pred = y_pred * prediction_df["stds"].values + prediction_df["mus"].values

            # append to the forecasts matrix
            forecasts[:, i] = y_pred

            # Append to the in_sample values as the last value
            prediction_df["in_sample"] = [x + [y] for x, y in zip(prediction_df["in_sample"].values, y_pred)]

        # return the forecasts
        return forecasts.tolist()

    def unstack_dataframe(self, df):
        """
        Returns the dataframe with predictions to the output format

        Args:
            df (pd.DataFrame):
                The dataframe containing the predictions
            model_name (str):
                The name of the model

        Returns:
            out_df (pd.DataFrame):
                The dataframe with the predictions in the right format

        """
        # returns the right format

        # set the right index
        ids = ["unique_id", "cutoff", "cv"]
        df = df.reset_index().set_index(ids)

        # take the col names and drop the in-sample
        cols = df.columns.drop("in_sample")

        # Initialize a df
        out_df = pd.DataFrame()

        # Itterate over columns and explode
        for col in cols:
            # Take the column and explode
            temp = df[[col]].explode(col)
            # concat with the main_df
            out_df = pd.concat([out_df, temp], axis=1)

        # reset index and rename
        out_df = out_df.reset_index().rename(
            columns={
                "out_of_sample": "True",
                "forecast_dates": "date",
                "horizon": "fh",
                "forecasts": "y",
            }
        )
        # add the model name
        out_df["Model"] = self.model_name
        return out_df
