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