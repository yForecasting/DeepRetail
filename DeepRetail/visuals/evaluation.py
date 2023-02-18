import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_single_hist_boxplot(eval_df_temp, metrics, model):
    """
    Creates a subplot for the distribution of errors for the given metrics.
    The subplot contains a boxplot on the bottom and a histogram on top.

    Args:
        eval_df_temp (pd.DataFrame): An evaluation dataframe from the Evaluate object.
        metrics (list): The metrics to consider.
        model (str): The names of the models

    """

    # Initialize
    color = (0.2, 0.4, 0.6)
    gray_scale = 0.93

    # pick the number of bins for the histogram based on the sample size
    # sqrt of the number of samples
    bins = int(np.sqrt(len(eval_df_temp)))

    # Take the metrics
    metric_a, metric_b = metrics
    metric_a_vals, metric_b_vals = eval_df_temp[metric_a].values, eval_df_temp[metric_b].values

    # Define axes
    fig, axs = plt.subplots(2, 2, figsize=(12, 3), gridspec_kw={"height_ratios": (.25, .75)})

    # Plot boxplot and histogram for metric 1
    axs[1, 0].hist(metric_a_vals, bins=bins, color=color)
    axs[0, 0].boxplot(
                    metric_a_vals, vert=False, widths=0.4, notch=True, patch_artist=True,
                    boxprops=dict(facecolor=color, color=color))
    axs[0, 0].set_title(metric_a)

    # Edit subplot
    axs[0, 0].set_xticklabels([])  # remove x axis on the boxplot
    plt.subplots_adjust(hspace=0)  # Remove the gap between the subplots
    axs[0, 0].spines['bottom'].set_visible(False)  # remove lines
    axs[1, 0].spines['top'].set_visible(False)

    # Plot boxplot and histogram for metric 2
    axs[1, 1].hist(metric_b_vals, bins=bins, color=color)
    axs[0, 1].boxplot(metric_b_vals, vert=False, widths=0.4, notch=True,
                      patch_artist=True, boxprops=dict(facecolor=color, color=color))
    axs[0, 1].set_title(metric_b)

    # Edit subplot
    axs[0, 1].set_xticklabels([])  # remove x axis on the boxplot
    axs[0, 1].set_yticklabels([])  # remove y axis on the boxplot
    plt.subplots_adjust(hspace=0)  # Remove the gap between the subplots
    axs[0, 1].spines['bottom'].set_visible(False)  # remove lines
    axs[1, 1].spines['top'].set_visible(False)

    # A single title for the whole figure
    fig.suptitle(f'{model}: Distribution of {metric_a} and {metric_b}')

    # Add the grid for y axis only
    axs[1, 0].yaxis.grid(True)
    axs[1, 1].yaxis.grid(True)

    # Change the background color on all axes
    for ax in axs.flat:
        ax.set_facecolor((gray_scale, gray_scale, gray_scale))

    # Show plot
    plt.show()


def plot_box(evaluation_df, metrics, fliers=True):
    """
    Plots boxplots for every model for the provide metrics.

    Args:
        evaluation_df (pd.DataFrame): An evaluation dataframe from the Evaluate object.
        metrics (list): The metrics to consider.
        fliers (bool): Whether to show the outliers or not.
    """

    # Initialize
    total_mets = len(metrics)
    # get the metrics names
    metric_names = [metric.__name__ for metric in metrics]
    gray_scale = 0.9

    # define columns
    cols = total_mets if total_mets < 3 else 3
    rows = total_mets // cols

    # ad-hoc correction for specific cases
    if total_mets % cols != 0:
        rows += 1

    plt.figure(figsize=(14, 8))
    for i, met in enumerate(metric_names, start=1):

        # Define subplot
        plt.subplot(rows, cols, i)

        # build the graph

        # Keep relevant columns
        temp_df = evaluation_df[["unique_id", "Model", met]]

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


def plot_line(evaluation_df, metrics):
    """
    Plots lineplots for every metric. Axis x containts the forecast horizon and y the given metric.
    A line is drawn for every model.
    The dataframe should be grouped as: ['unique_id', 'Model', 'fh']

    Args:
        evaluation_df (pd.DataFrame): An evaluation dataframe from the Evaluate object.
        metrics (list): The metrics to consider.

    """

    # Initialize
    # Initialize
    total_mets = len(metrics)
    # get the metrics names
    metric_names = [metric.__name__ for metric in metrics]
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
        met = metric_names[i]

        # build the graph
        # Keep relevant columns
        temp_df = evaluation_df[["unique_id", "Model", "fh", met]]

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
