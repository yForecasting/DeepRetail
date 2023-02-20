import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DeepRetail.transformations.formats import pivoted_df


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


def visualize_forecasts(n, h, cv, freq, original_df, forecast_df, models, show_in_sample):
    """
    Visualize the forecasts for the given models.
    If defined, plots the in-samples values too.

    Args:
        n (int): The number of series to plot.
        h (int): The forecast horizon.
        freq (str): The frequency of the data.
        original_df (pd.DataFrame): The original dataframe.
        forecast_df (pd.DataFrame): The forecast dataframe.
        models (list): The models to consider.
        show_in_sample (bool): Whether to show the in-sample values or not.
    """

    # Initialize parameters
    dates = pd.to_datetime(original_df.columns)
    titles = original_df.index.astype(str)
    to_mask = len(dates) - h
    end_date = dates[-h]

    # Initialize figure specific parameters
    row_to_vis = n if n < 3 else 3
    total_to_vis = n - n % row_to_vis
    gray_scale = 0.9

    # Run loop
    for idx in range(0, total_to_vis, row_to_vis):
        plt.figure(figsize=(20, 15))
        for i in range(0, row_to_vis):
            ax = plt.subplot(row_to_vis, row_to_vis, i + 1)

            # the series index to plot
            series_index = idx + i
            series_name = titles[series_index]
            y = original_df.loc[series_name]

            if show_in_sample:
                # plot the in_sample values
                y.plot(ax=ax, label='y')
                # Line to seperate train/test
                plt.axvline(x=end_date, color="black", linestyle="--")
                plt.axvspan(xmin=end_date, xmax=dates[-1], color="darkgray")

            # Plot predictions
            for model in models:
                # Filter values
                temp_fc_df = forecast_df[forecast_df['Model'] == model]

                # Currently only plots for the last cv!
                temp_fc_df_fold = temp_fc_df[temp_fc_df['cv'] == cv]
                # pivot the values and take the series to show
                temp_vals = pivoted_df(temp_fc_df_fold, freq).loc[series_name].values

                # if to show in-sample
                if show_in_sample:
                    # pad & convert to series for plotting
                    temp_vals = np.pad(
                        temp_vals, pad_width=(to_mask, 0), constant_values=None
                    )  # we use rolling cv so we have to increase the to pad every time

                    temp_vals = pd.Series(temp_vals, index=dates)

                    # Plot
                    temp_vals.plot(ax=ax, label=model)

                else:
                    # plot without the mask
                    temp_vals.plot(ax=ax, label=model)

            # Title:
            # currently not supporting the metric on title
            title = series_name
            plt.title(title)

            # Edit the format of the plot
            plt.xticks(rotation=45)
            plt.grid()
            plt.legend()

            ax.set_facecolor((gray_scale, gray_scale, gray_scale))
