from DeepRetail.exploratory.metrics import mse, mae
from DeepRetail.preprocessing.converters import pivoted_df
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def in_sample_metrics(df):
    """Estimate the in-sample Naive mse and mae

    Args:
        df (pd.Dataframe): A pivoted df

    """
    # Creates a copy
    in_sample_metrics = df.copy()

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


def calculate_scores(group, metrics):

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


def plot_box(test, fliers=True):

    # Initialize
    models = test["Model"].unique()
    metrics_list = test.columns[2:]
    total_mets = len(metrics_list)
    gray_scale = 0.9

    # define columns
    cols = total_mets if total_mets < 3 else 3
    rows = total_mets // cols

    # ad-hoc correction for specific cases
    if total_mets % cols != 0:
        rows += 1

    plt.figure(figsize=(14, 8))
    for i, met in enumerate(metrics_list, start=1):

        # Define subplot
        plt.subplot(rows, cols, i)

        # build the graph

        # Keep relevant columns
        temp_df = test[["unique_id", "Model", met]]

        # Pivot to get the right format for the box plot
        temp_df = pd.pivot_table(
            temp_df, index="unique_id", columns="Model", values=met, aggfunc="first"
        )

        ax = temp_df.boxplot(
            color="black", meanline=True, showbox=True, showfliers=fliers
        )
        ax.set_facecolor((gray_scale, gray_scale, gray_scale))
        ax.set_title(met)
        ax.grid()

    # plt.tight_layout()
    plt.show()


def plot_line(test):

    # Initialize
    models = test["Model"].unique()
    metrics_list = test.columns[3:]
    total_mets = len(metrics_list)
    gray_scale = 0.9

    # define columns
    cols = total_mets if total_mets < 3 else 3
    rows = total_mets // cols

    # ad-hoc correction for specific cases
    if total_mets % cols != 0:
        rows += 1

    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    flat_axes = axes.flatten()[:total_mets]

    for i, ax in enumerate(flat_axes):

        # Define the current metric
        met = metrics_list[i]

        # build the graph
        # Keep relevant columns
        temp_df = test[["unique_id", "Model", "fh", met]]

        # Groupby per fh
        temp_df = temp_df.groupby(["Model", "fh"]).agg({met: np.mean}).reset_index()

        # Pivot
        temp_df = pd.pivot_table(
            temp_df, index="Model", columns="fh", values=met, aggfunc="first"
        )

        # Plot
        temp_df.T.plot(marker="o", title=met, ax=ax)

        ax.set_xlabel(None)
        ax.set_facecolor((gray_scale, gray_scale, gray_scale))
        ax.set_title(met)
        ax.set_xticks(np.arange(1, temp_df.shape[1] + 1, 1))
        ax.grid()

    to_drop = np.arange(total_mets, len(axes.flatten()))
    for drop in to_drop:
        drop = drop % 3
        fig.delaxes(axes[rows - 1][drop])

    # plt.tight_layout()
    plt.show()


class Evaluator(object):
    def __init__(self, df, res_df, freq):

        self.res_df = res_df
        self.freq = freq

        # Get the total number of fh and cv
        self.total_cv = len(self.res_df["cv"].unique())
        self.total_fh = len(self.res_df["fh"].unique())

        # if we have a sample and not the whole df
        self.ids = res_df["unique_id"].unique()

        # Filter the df
        self.df = df[df.index.isin(self.ids)]

    def evaluate(self, metrics, per_fh=False):

        """Returns a df with the evaluation scores

        Args:
            metrics (metric-type): Imports from the metrics file
        """

        # Estimate in-sample naive mse, mae and add it to the res_df
        in_sample_naive = in_sample_metrics(self.df)

        # Merge with the main df
        new_res_df = pd.merge(self.res_df, in_sample_naive, on="unique_id", how="left")

        # Estimate the scores
        if per_fh:
            self.eval_df = pd.DataFrame(
                new_res_df.groupby(["unique_id", "Model", "fh"]).apply(
                    calculate_scores, metrics=metrics
                )
            ).reset_index()
        else:
            self.eval_df = pd.DataFrame(
                new_res_df.groupby(["unique_id", "Model"]).apply(
                    calculate_scores, metrics=metrics
                )
            ).reset_index()

        return self.eval_df

    def evaluate_boxplot(self, metrics=None, fliers=True):

        # With evaluate boxplot the per_fh argument should be set to False
        # If it is not, then we re-evaluate

        if not hasattr(self, "eval_df"):
            if metrics is None:
                raise ValueError(
                    "Either provide a list of metrics or call .evaluate() first"
                )
            else:
                # Generate the eval df
                self.eval_df = self.evaluate(metrics, per_fh=False)
        else:

            if "fh" in self.eval_df.columns:
                raise ValueError(
                    "per_fh should be set to False in .evaluate(). Please re-evaluate"
                )

        plot_box(self.eval_df, fliers=fliers)

    def evaluate_lineplot(self, metrics=None):
        # With evaluate lineplots the per_fh argument should be set to True
        # If it is not, then we re-evaluate

        if not hasattr(self, "eval_df"):
            if metrics is None:
                raise ValueError(
                    "Either provide a list of metrics or call .evaluate() first"
                )
            else:
                # Generate the eval df
                self.eval_df = self.evaluate(metrics, per_fh=False)
        else:

            if "fh" not in self.eval_df.columns:
                raise ValueError(
                    "per_fh should be set to True in .evaluate(). Please re-evaluate"
                )

        plot_line(self.eval_df)

    def add_forecast(self, new_fc):
        # adds a new forecast object!
        ...

    def plot_forecasts(self, models, n):

        # define visual specific
        row_to_vis = n if n < 3 else 3
        total_to_vis = n - n % row_to_vis

        # Extract values
        true_vals = self.df.values
        dates = pd.to_datetime(self.df.columns)
        titles = self.df.index.astype(str)
        to_mask = len(dates) - self.total_fh
        gray_scale = 0.9
        end_date = dates[-self.total_fh]
        gray_scale_2 = 0.95
        v_span_color = (gray_scale_2, gray_scale_2, gray_scale_2)

        # Initialize
        for idx in range(0, total_to_vis, row_to_vis):
            plt.figure(figsize=(20, 15))
            for i in range(0, row_to_vis):
                ax = plt.subplot(row_to_vis, row_to_vis, i + 1)

                # the series index to plot
                series_index = idx + i
                series_name = titles[series_index]
                y = true_vals[series_index]

                # Add the title
                plt.title(series_name)

                # Plot the true
                plt.plot(dates, y, label="True")

                # Add a horizonta line
                plt.axvline(x=end_date, color="black", linestyle="--")
                plt.axvspan(xmin=end_date, xmax=dates[-1], color="darkgray")

                # For every model
                for model in models:

                    # filter and get the values
                    temp_df = self.res_df[
                        (self.res_df["Model"] == model)
                        & (self.res_df["unique_id"] == series_name)
                    ]
                    temp_vals = pivoted_df(temp_df, self.freq).values[0]

                    # Mask values & get the mask
                    temp_vals = np.pad(
                        temp_vals, pad_width=(to_mask, 0), constant_values=None
                    )
                    s1mask = np.isfinite(temp_vals)

                    # Plot
                    plt.plot(dates[s1mask], temp_vals[s1mask], label=model)

                # Fix the plot
                plt.xticks(rotation=45)
                plt.grid()
                plt.legend()
                ax.set_facecolor((gray_scale, gray_scale, gray_scale))

            plt.gcf().tight_layout()
            plt.show()
