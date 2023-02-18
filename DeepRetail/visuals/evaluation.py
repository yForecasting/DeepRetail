import matplotlib.pyplot as plt
import numpy as np


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
