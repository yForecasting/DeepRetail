from DeepRetail.transformations.formats import transaction_df, pivoted_df
from DeepRetail.evaluation.metrics import mse, mae, rmsse, scaled_error
from DeepRetail.visuals.evaluation import (
    visualize_forecasts,
    plot_single_hist_boxplot,
    plot_box,
    plot_line,
)
import pandas as pd


def calculate_group_scores(group, metrics):
    """
    Calculates the scores for a group of predictions

    Args:
        group (pd.DataFrame): A group of predictions
        metrics (list): A list of metrics to be calculated

    Returns:
        pd.Series: A series with the scores

    """
    # Extract the true and predicted values
    forecasted = group["y"].values
    true = group["True"].values

    # Getting in-sample values only once
    insample_mse = group["in_sample_Naive_mse"].unique()[0]
    in_sample_mae = group["in_sample_Naive_mae"].unique()[0]

    # Get the names of the metrics
    metric_names = [metric.__name__ for metric in metrics]

    # Sequentially append to dictionary
    scores_dict = dict()
    for name, metric in zip(metric_names, metrics):
        scores_dict[name] = metric(
            true, forecasted, naive_mse=insample_mse, naive_mae=in_sample_mae
        )

    return pd.Series(scores_dict)


class Evaluator(object):
    """
    Class for evaluating the performance of a forecasting model.

    Args:
        original_df (pd.DataFrame): The original input DataFrame containing the time series data.
        result_df (pd.DataFrame): The DataFrame containing the predicted values.
        freq (str): The frequency of the time series data, e.g. 'D' for daily, 'M' for monthly, etc.
        format (str, optional): The format of the input DataFrame. Can be either 'pivoted' or 'transaction'.
            Defaults to 'pivoted'.

    Methods:
        calculate_in_sample_metrics(self):
            Calculates the in-sample metrics for the input DataFrame.

        evaluate(self, metrics, group_scores_by=["unique_id", "Model", "fh", "cv"]):
            Evaluates the predictions and calculates the specified metrics for each group.

        plot_error_distribution(self, metric, group_scores_by=["unique_id", "Model", "fh", "cv"]):
            Plots the distribution of errors for the specified metric.

        plot_model_summary(self, metrics, type, group_scores_by=["unique_id", "Model", "fh", "cv"], fliers=True):
            Plots summary error plots per model. Supports two types of options, line plot and box plot.
        plot_forecasts(self, n, models='all', show_in_sample=True):
            Plots a number of forecasts for the defined models


    Examples:
        >>> from DeepRetail.evaluation import Evaluator

        # Load original and result dataframes
        >>> original_df = pd.read_csv('original_data.csv')
        >>> result_df = pd.read_csv('result_data.csv')

        # Create an Evaluator object
        >>> evaluator = Evaluator(original_df, result_df, freq='D', format='pivoted')

        # Define the metrics to be evaluated
        >>> metrics = [mse, mae]

        # Group scores by unique_id, Model, fh, and cv
        >>> group_scores_by = ['unique_id', 'Model', 'fh', 'cv']

        # Calculate the evaluation metrics
        >>> evaluation_df = evaluator.evaluate(metrics, group_scores_by)

        # Some visualization:
        >>> evaluator.plot_error_distribution()
        >>> evaluator.plot_model_summary(metrics = metrics, type = 'line')
        >>> evaluator.plot_model_summary(metrics = metrics, type = 'boxplot')
        >>> evaluator.plot_forecasts(n=10, models='all', show_in_sample=True)

    """

    def __init__(self, original_df, result_df, freq, format="pivoted"):
        """
        Initializes the Evaluator object.

        Args:
            original_df (pd.DataFrame): The original transactional dataframe
            result_df (pd.DataFrame): The resulting predictions dataframe
            freq (str): The frequency of the time series data
            format (str): The format of the input data. Default is 'pivoted'.

        """

        self.result_df = result_df
        self.freq = freq

        if format == "pivoted":
            original_df = transaction_df(original_df)

        # Keep only ids that exist on the result df
        # This is in case I took a smaller sample to forecast
        self.original_df = original_df[
            original_df["unique_id"].isin(result_df["unique_id"].unique())
        ]

        # Get the fh and the cv
        self.total_cv = len(self.result_df["cv"].unique())
        self.total_fh = len(self.result_df["fh"].unique())

    def calculate_in_sample_metrics(self):
        """
        Calculates the in-sample metrics(mae, mse) for the time series data.
        Its used for estimating RMSSE and MASE.

        Returns:
            pd.DataFrame: A dataframe containing the in-sample metrics.
        """

        # Creates a copy
        in_sample_metrics = pivoted_df(self.original_df, self.freq)

        # Estimates mae and mse
        in_sample_metrics["in_sample_Naive_mse"] = in_sample_metrics.apply(
            lambda row: mse(row.iloc[1:], row.iloc[:-1]), axis=1
        )
        in_sample_metrics["in_sample_Naive_mae"] = in_sample_metrics.apply(
            lambda row: mae(row.iloc[1:], row.iloc[:-1]), axis=1
        )

        # Keeps only relevant columns
        in_sample_metrics = in_sample_metrics[
            ["in_sample_Naive_mse", "in_sample_Naive_mae"]
        ]
        return in_sample_metrics.reset_index()

    def evaluate(self, metrics, group_scores_by=["unique_id", "Model", "fh", "cv"]):
        """
        Evaluates the predictions and calculates the specified metrics for each group.

        Args:
            metrics (list): A list of metrics to be calculated for each group. Each metric should be a function that
                            takes two arguments: a NumPy array of true values and a NumPy array of predicted values. The
                            list of metrics can also include optional arguments to be passed to the metric function.
            group_scores_by (list): A list of columns to group the predictions by.
                The default value groups by unique_id, Model, fh, and cv.

        Returns:
            pd.DataFrame: A DataFrame containing the scores for each group
        """
        # Estimate the insample naive mse and mae
        # They are used for scalling
        in_sample_metrics = self.calculate_in_sample_metrics()

        # Merge with the predictions dataframe
        merged_result_df = pd.merge(
            self.result_df, in_sample_metrics, on="unique_id", how="left"
        )

        # estimate the scores
        evaluation_df = pd.DataFrame(
            merged_result_df.groupby(group_scores_by).apply(
                calculate_group_scores, metrics=metrics
            )
        ).reset_index()

        # add the metrics to the object
        # self.evaluated_metrics = metrics
        # self.evaluate_df = evaluation_df
        # return self.evaluation_df

        return evaluation_df

    def plot_error_distribution(
        self,
        metrics=[rmsse, scaled_error],
        group_scores_by=["unique_id", "Model", "fh", "cv"],
    ):
        """
        Plots the error distribution for the given metrics.
            Default and recommended metrics are rmsse and scaled_error.

        Args:
            metrics (list): A list of metrics to be calculated for each group.
                Metrics should be included in DeepRetail.evaluation.metrics.
                The default value is [rmsse, scaled_error].
            group_scores_by (list): A list of columns to group the predictions by.
                The default value groups by unique_id, Model, fh, and cv.

        """

        # Evaluate on the given metrics
        evaluation_df = self.evaluate(metrics, group_scores_by)

        # Take the name of the metrics
        metric_names = [metric.__name__ for metric in metrics]

        # take the models
        models = evaluation_df["Model"].unique()

        # Plots for every model
        for model in models:
            # Filter by the model
            temp_df = evaluation_df[evaluation_df["Model"] == model]
            # Plot
            plot_single_hist_boxplot(temp_df, metric_names, model)

    def plot_model_summary(
        self,
        metrics,
        type,
        group_scores_by=["unique_id", "Model", "fh", "cv"],
        fliers=True,
    ):
        """
        Plots summary error plots per model. Supports two types of options, line plot and box plot.

        Args:
            metrics: A list of metrics to be calculated for each group.
                Metrics should be included in DeepRetail.evaluation.metrics.
            type: The type of plot create.
                Supports "line" and "boxplot"
            group_scores_by: The grouping for the evaluation.
                Used only on the boxplot. Line plot uses specific grouping.
                Default and recommended is the total evaluation.
            fliers: Whether to include fliers or not on the boxplot.
        """

        if type == "boxplot":
            # Evaluate on the given metrics
            evaluation_df = self.evaluate(metrics, group_scores_by)
            # plot
            plot_box(evaluation_df, metrics, fliers=True)

        elif type == "line":
            # Not allowing for grouping selection.
            grouping = ["Model", "unique_id", "fh"]
            evaluation_df = self.evaluate(metrics=metrics, group_scores_by=grouping)
            # plot
            plot_line(evaluation_df, metrics=metrics)

        else:
            raise ValueError("Currently supporting only boxplot and line plots")

    def plot_forecasts(self, n, models="all", show_in_sample=True):
        """
        Plots the forecasts for the given models.

        Args:
            n (int): The number of samples to plot.
            models (list): A list of models to plot.
                        Default is 'all' which plots all models.
            show_in_sample (bool): Whether to show the in-sample data or not.
                        Default is True.
            group_scores_by (list): A list of columns to group the predictions by.
        """

        if models == "all":
            models = self.result_df["Model"].unique()

        # convert the original df to pivote format
        temp_df = pivoted_df(self.original_df, self.freq)

        visualize_forecasts(
            n,
            self.total_fh,
            self.total_cv,
            self.freq,
            temp_df,
            self.result_df,
            models=models,
            show_in_sample=show_in_sample,
        )
