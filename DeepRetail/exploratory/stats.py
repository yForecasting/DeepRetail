import pandas as pd
import numpy as np
from tsfeatures import tsfeatures, stl_features, entropy
import matplotlib.pyplot as plt
from DeepRetail.preprocessing.converters import transaction_df
from collections import Counter


def moving_average(a, n=3):
    """Estimate the moving average of a time series given the size of the window

    Args:
        a (array-type): The time series
        n (int, optional): The size of the moving average window.
                         Defaults to 3.

    Returns:
        array-type:  A new time series with moving averages
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def visualize_series(
    pivoted_df,
    n,
    add_moving_average=False,
    month=True,
    quarter=True,
    annual=True,
):
    """Plots the given amount of time series.

    Args:
        pivoted_df(pd.DataFrame): The dataframe with the time series.
                                  Should be in a pivoted format.
        n (int): The total number of time series to show.
                Note: The function shows 3 time series per row.
        moving_average (bool, optional): If to include moving average time series.
                                    Supports monthly, quarterly and annualy.
                                    If set to true, define the periods of interest
                                    Defaults to False.
        month (bool, optional): If to include monthly moving average time series.
                                Defaults to True.
        quarter (bool, optional): If to include quarterly moving average time series.
                                Defaults to True
        annual (bool, optional): If to include annualy moving average time series.
                                Defaults to True
    """

    # Extract variables
    vals = pivoted_df.values
    dates = pd.to_datetime(pivoted_df.columns)
    titles = pivoted_df.index.astype(str)

    # Initialize
    gray_scale = 0.9
    row_to_vis = 3
    total_to_vis = n - n % row_to_vis  # total number should be divided by 3

    for idx in range(0, total_to_vis, row_to_vis):
        plt.figure(figsize=(20, 15))
        for i in range(0, row_to_vis):
            ax = plt.subplot(row_to_vis, row_to_vis, i + 1)
            # the series index to plot
            series_index = idx + i
            y = vals[series_index]

            if add_moving_average:
                # Getting the mas
                ma_monthly = moving_average(y, 4)
                ma_monthly = np.pad(ma_monthly, pad_width=(3, 0), constant_values=None)
                ma_quarterly = moving_average(y, 13)
                ma_quarterly = np.pad(
                    ma_quarterly, pad_width=(12, 0), constant_values=None
                )
                ma_annualy = moving_average(y, 52)
                ma_annualy = np.pad(ma_annualy, pad_width=(51, 0), constant_values=None)

                s1mask = np.isfinite(ma_monthly)
                s2mask = np.isfinite(ma_quarterly)
                s3mask = np.isfinite(ma_annualy)
            else:
                month, quarter, annual = False, False, False

            plt.title("Index:" + titles[series_index])
            plt.plot(dates, y, label="y")

            if month:
                plt.plot(dates[s1mask], ma_monthly[s1mask], label="Monthly Average")
            if quarter:
                plt.plot(dates[s2mask], ma_quarterly[s2mask], label="Quarterly Average")
            if annual:
                plt.plot(dates[s3mask], ma_annualy[s3mask], label="Annualy Average")
            plt.xticks(rotation=45)
            plt.grid()
            plt.legend()

            ax.set_facecolor((gray_scale, gray_scale, gray_scale))
        plt.gcf().tight_layout()
        plt.show()


def print_summary(df):
    """Prints some summary statistics

    Args:
        df (pd.DataFrame): the DataFrame in pivoted format
    """

    total_ts = df.shape[0]
    total_periods = df.shape[1]

    print(f"A total of {total_ts} items for a total of {total_periods} periods")


def visualize_features(features):
    """Plots boxplots for entropy, seasonality and trend

    Args:
        features (pd.DataFrame): A features dataframe created with tsfeatures
    """

    # Drop some nans
    features_interest = features[["unique_id", "entropy", "trend", "seasonal_strength"]]

    features_interest = features_interest.dropna(how="any", axis=0)

    # Take the values
    f_entropy = features_interest["entropy"].values
    f_trend = features_interest["trend"].values
    f_seasonality = features_interest["seasonal_strength"].values

    # plt.figure()
    fig, ax = plt.subplots(figsize=(14, 7))

    labels = ["Entropy", "Trend", "Seasonality"]
    values = [f_entropy, f_trend, f_seasonality]
    c = "#1f77b4"

    ax.boxplot(
        values,
        meanline=True,
        showbox=True,
        patch_artist=True,
        boxprops=dict(facecolor=c, color="black", linewidth=2),
        capprops=dict(color=c, linewidth=2),
        whiskerprops=dict(color=c, linewidth=2),
        flierprops=dict(color=c, markeredgecolor="black"),
        medianprops=dict(color="black", linewidth=1.7),
    )

    ax.set_title("Feature Analysis -Jules-")
    gray_scale = 0.93
    ax.set_xticklabels(labels)
    ax.set_facecolor((gray_scale, gray_scale, gray_scale))
    plt.grid()

    plt.show()


def get_features(df, seasonal_period, plot=True):
    """Estimates entropy, seasonality and trend using tsfeatures

    Args:
        df (pd.DataFrame): The pivoted dataframe
        seasonal_period (int): The seasonal period (frequency)
        plot (bool, optional): If to plot a histogram with the results.
                            Defaults to True.

    Returns:
        _type_: _description_
    """

    # convert to transaction format
    df = transaction_df(df)

    # Estimate the features
    features = tsfeatures(df, freq=seasonal_period, features=[stl_features, entropy])

    if plot:
        visualize_features(features)

    return features


def scatter_hist(df, ax, ax_histx, ax_histy, fig, ax_cor):

    x = df["Level"].values
    y = df["Variance"].values
    c = df["Volume"].values

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    im = ax.scatter(x, y, c=c)

    # Picks the number of bins automaticaly!
    ax_histx.hist(x, bins=np.histogram_bin_edges(x, bins="auto"))
    ax_histy.hist(
        y, bins=np.histogram_bin_edges(y, bins="auto"), orientation="horizontal"
    )

    clb = fig.colorbar(im, cax=ax_cor)
    clb.ax.set_title("Volume")
    # ax_cor.set_xlabel('Volume')


def plot_scatter_hist(df):

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    rect_colorbar = [left + width + spacing, bottom + height + spacing, 0.05, 0.18]

    # start with a square Figure
    fig = plt.figure(figsize=(12, 12))
    gray_scale = 0.93

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_cor = fig.add_axes(rect_colorbar)

    # use the previously defined function
    scatter_hist(df, ax, ax_histx, ax_histy, fig, ax_cor)

    # Edit the plot
    ax.set_xlabel("Level")
    ax.set_ylabel("Variance")
    ax.grid(True)
    ax.set_facecolor((gray_scale, gray_scale, gray_scale))

    ax_histx.grid(True)
    ax_histx.set_facecolor((gray_scale, gray_scale, gray_scale))

    ax_histy.grid(True)
    ax_histy.set_facecolor((gray_scale, gray_scale, gray_scale))

    plt.show()


def plot_level_volume_variance(df, level_threshold=None):
    """Plots a scatter-histogram for the given dataframe

    Args:
        df (pd.DataFrame): The DataFrame to plot
        level_threshold (int): An upper bound for the level to filter out outliers
                                Assists in making the plot more readable
    """

    # Convert to transaction format
    df = transaction_df(df)

    # Estimate volume and variance
    df_group = df.groupby("unique_id").agg(Volume=("y", np.sum), Variance=("y", np.std))

    # Estimate non-zero level
    df_non_zero = df[df["y"] != 0]
    level_df = (
        df_non_zero.groupby("unique_id")
        .agg({"y": np.mean})
        .rename(columns={"y": "Level"})
    )

    # Merge
    df_group = pd.merge(
        level_df, df_group, left_index=True, right_index=True, how="inner"
    )

    # Filter on the threshold
    if level_threshold is not None:
        to_plot = df_group[(df_group["Level"] < level_threshold)]
    else:
        to_plot = df_group.copy()

    plot_scatter_hist(to_plot)


def zeros_hist(df):
    """Plots a histogram with the % of zeros per time series

    Args:
        df (pd.DataFrame): The dataframe with the time series
    """

    zero_per_row = df.isin([0]).sum(axis=1) / len(df.columns)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.hist(zero_per_row, bins=15)

    ticks = plt.xticks()[0][1:-1]
    new_labels = [str(round(tick * 100)) + "%" for tick in ticks]
    gray_scale = 0.93

    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels=new_labels)
    ax.set_title("Histogram for the Perchentage of zeros on individual time series")
    ax.set_facecolor((gray_scale, gray_scale, gray_scale))
    ax.grid(linestyle="-", color="w", linewidth=2)

    plt.show()


def COV(ts):
    """Estimates the coefficient of variation for the given time series

    Args:
        ts (array-type): A single time series

    Returns:
        float: The coefficient of variation
    """
    # filter out zero demand
    ts = ts[ts > 0]
    # Get the std and mean
    ts_std = np.std(ts)
    ts_mean = np.mean(ts)
    cov = ts_std / ts_mean
    return cov


def CV_zeros_plot(df):
    """Plots the perchentage of zeros and the Coefficient of Variation(CV)
    (on the non-zero demand)

    Args:
        df (pd.DataFrame): The dataframe with the time series

    """
    # Checking the intermitency
    # Calculating the coef of variations for the non-zero demand for every ts
    covs = df.apply(lambda row: COV(row), axis=1)

    # Gets the % of zeros per row
    zero_per_row = df.isin([0]).sum(axis=1) / len(df.columns)

    # Plotting vs the % of zeros
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(zero_per_row, covs, alpha=0.5)

    ticks = plt.xticks()[0][1:-1]
    new_labels = [str(round(tick * 100)) + "%" for tick in ticks]
    gray_scale = 0.93

    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels=new_labels)
    ax.set_ylabel("Coefficient of variation of non−zero demand")
    ax.set_xlabel("Percentage of zeros")

    ax.set_facecolor((gray_scale, gray_scale, gray_scale))
    ax.grid(linestyle="-", color="w", linewidth=2)

    plt.show()


def get_class(p, v):
    """
    Estimates the intermittency

    """
    # returns the class
    if p < 1.32:
        if v < 0.49:
            out = "Smooth"
        else:
            out = "Erratic"
    else:
        if v < 0.49:
            out = "Intermittent"
        else:
            out = "Lumpy"

    return out


def get_intermittent_class(ts):
    """Returns the intermittent class of a time series

    Args:
        ts (array-type): The time series to check its intermittency

    Returns:
        str: A description of the intermittency
    """

    # Get the indices of the true demand
    nzd = np.where(ts != 0)[0]

    # If we have at least non-zero demand value
    if len(nzd) > 0:

        # Get the total non-zero observations
        k = len(nzd)

        # Get the actual demand
        z = ts[nzd]  # the demand

        # Get the intervals between non-zero observations
        x = np.diff(nzd)
        x = np.insert(x, 0, nzd[0])

        # Get the average interval -> will be used for classification later
        p = np.mean(x)

        # Get the squared cv
        v = (np.std(z) / np.mean(z)) ** 2

        # classify
        in_class = get_class(p, v)

    # If we have no demand!
    else:
        in_class = "No Demand"

    return in_class


def intermittency_classification(df, plot=True):
    """Classifies the time series of the dataframe based on their intermittency

    Args:
        df (pd.DataFrame): The dataframe with the time series
        plot (bool, optional): If the function returns a barplot with the results.
                                Defaults to True.
    """

    data = df.values

    # Get the classes
    classes = [get_intermittent_class(ts) for ts in data]

    # Counts how many we have on each class
    class_to_dict = dict(Counter(classes))

    print("\n".join("{}: {}".format(k, v) for k, v in class_to_dict.items()))

    if plot:
        # Plotting vs the % of zeros
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)

        ax.bar(range(len(class_to_dict)), class_to_dict.values(), align="center")

        # ticks = range(len(class_to_dict)), list(class_to_dict.keys())
        gray_scale = 0.93

        ticks = np.arange(0, len(class_to_dict), 1)

        ax.set_xticks(ticks=ticks)
        ax.set_xticklabels(labels=class_to_dict.keys())

        ax.set_facecolor((gray_scale, gray_scale, gray_scale))
        ax.grid(linestyle="-", axis="y")
        ax.set_title("Classification of Demand Time Series")

        plt.show()
