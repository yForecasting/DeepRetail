from DeepRetail.transformations.formats import transaction_df, pivoted_df
from DeepRetail.evaluation.metrics import mse, mae
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

    # Sequantily append to dictionary
    scores_dict = dict()
    for name, metric in zip(metric_names, metrics):
        scores_dict[name] = metric(
            true, forecasted, naive_mse=insample_mse, naive_mae=in_sample_mae
        )

    return pd.Series(scores_dict)


class Evaluator(object):
    def __init__(self, original_df, result_df, freq, format="pivoted"):
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
        # Estimate the insample naive mse and mae
        # They are used for scalling
        in_sample_metrics = self.calculate_in_sample_metrics()

        # Merge with the predictions dataframe
        merged_result_df = pd.merge(
            self.result_df, in_sample_metrics, on="unique_id", how="left"
        )

        # estimate the scores
        self.evaluation_df = pd.DataFrame(
            merged_result_df.groupby(group_scores_by).apply(
                calculate_group_scores, metrics=metrics
            )
        ).reset_index()

        return self.evaluation_df
