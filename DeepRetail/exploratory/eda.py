import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from DeepRetail.transformations.formats import (
    get_reminder,
    StandardScaler_custom,
    transaction_df,
    MinMaxScaler_custom,
)
from tsfeatures import tsfeatures, stl_features, entropy, lumpiness
from statsmodels.tsa.stattools import pacf, acf


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
    return ret[n - 1 :] / n  # noqa: E203


def visualize_series(
    pivoted_df,
    n,
    add_moving_average=False,
    month=True,
    quarter=True,
    annual=True,
    holiday_df=None,  # New parameter
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
        holiday_df (pd.DataFrame, optional): The dataframe with the holidays. Columns: Date, Holiday.

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

            if holiday_df is not None:
                # Convert the 'Date' column to datetime if it's not already
                holiday_df["date"] = pd.to_datetime(holiday_df["date"])

                # Extract the dates that are marked as holidays/special days
                special_days = holiday_df[holiday_df["holiday"] == "Holiday"]["date"]

                # Find indices of the special days in the 'dates' array
                special_indices = [
                    i for i, date in enumerate(dates) if date in special_days.values
                ]

                # Plot markers at the bottom of the chart for special days
                for special_idx in special_indices:
                    plt.plot(
                        dates[special_idx],
                        min(y),
                        "r^",
                        markersize=2,
                        label="Special Day" if idx == 0 and i == 0 else "",
                    )

            plt.xticks(rotation=45)
            plt.grid()
            plt.legend()

            ax.set_facecolor((gray_scale, gray_scale, gray_scale))

            # Make sure the legend only shows one instance of each label
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

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


def Residual_CoV(df, periods):
    """Estimates the coefficient of variation of the residuals of a seasonal decomposition
    Currently not supported due to statsmodels dependency.



    Args:
        df (pd.DataFrame): The pivoted dataframe
        periods (list): A list of seasonal periods

    Returns:
        pd.DataFrame: A dataframe with the CoV of the residuals
    """

    # Get the residuals
    residuals = get_reminder(df, periods=periods)

    # Get the std of the residuals
    std_residuals = np.array([np.std(res) for res in residuals])

    # Get the mean of the original series
    mean_original = np.array([df.mean(axis=1).values])

    # Get the CoV values
    CoV = std_residuals / mean_original

    # Map to [0,1] range using min max scaler
    CoV_scaled = MinMaxScaler_custom(CoV)

    # Reverse the scale so bigger => more forecastable
    CoV_scaled = 1 - CoV_scaled

    return CoV_scaled


def get_features(df, seasonal_period, periods):
    """Estimates coefficient of variation, entropy, seasonality and trend using tsfeatures

    Args:
        df (pd.DataFrame): The pivoted dataframe
        seasonal_period (int): The seasonal period (frequency)
        forecastability (str, optional): The forecastability metric to use.
                                        Accepts either 'entropy' or 'CoV'.
                                        CoV is a bit slower. Defaults to 'entropy'.
        periods (list, optional): The periods to use for the decomposition.

    Returns:
        features (pd.DataFrame):
            A dataframe with the estimate features
    """

    # convert to transaction format
    t_df = transaction_df(df)

    # Estimate the features

    features = tsfeatures(
        t_df, freq=seasonal_period, features=[stl_features, entropy, lumpiness]
    )
    # Estimate CV here
    features["Residual_CoV"] = Residual_CoV(df, periods).squeeze()

    # if plot:
    #    visualize_features(features)

    return features


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
        # k = len(nzd)

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


def scatter_hist(df, ax, ax_histx, ax_histy, fig, ax_cor):
    """
    Desing of the scatter plot with the histograms

    Args:
        df (pd.DataFrame): The dataframe with the time series
        ax (matplotlib.axes): The main axes
        ax_histx (matplotlib.axes): The x-axis axes
        ax_histy (matplotlib.axes): The y-axis axes
        fig (matplotlib.figure): The figure
        ax_cor (matplotlib.axes): The colorbar axes

    Returns:
        None
    """

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
    """
    Plots a scatter plot with histograms on the axes margins.

    Args:
        df (pd.DataFrame): The dataframe with the time series

    Returns:
        None
    """

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


def visualize_pacf(pivoted_df, n, lags, alpha, method="ywm"):
    """
    Visualizes the pacf for the given dataframe

    Args:
        pivoted_df (pd.DataFrame): The dataframe with the time series
        n (int): The number of time series to visualize
        lags (int): The number of lags to use
        alpha (float): The alpha value for the confidence interval
        method (str): The method to use for the confidence interval

    """

    # Extract variables
    vals = pivoted_df.values
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

            # Estiamte pacf
            pacf_x = pacf(y, nlags=lags, alpha=alpha, method=method)
            pacf_x, confint = pacf_x[:2]
            pacf_x = pacf_x[1:]
            confint = confint[1:]

            lags_x = np.arange(0, lags)

            ax.vlines(lags_x, [0], pacf_x)
            ax.axhline()
            ax.margins(0.05)
            title = "Partial Autocorrelation " + str(titles[series_index])
            ax.plot(
                lags_x,
                pacf_x,
                marker="o",
                markersize=5,
                markerfacecolor="red",
                markeredgecolor="red",
            )

            # ax.set_ylim(-1, 1)
            # Setting the limits
            ax.set_ylim(
                1.25 * np.minimum(min(pacf_x), min(confint[:, 0] - pacf_x)),
                1.25 * np.maximum(max(pacf_x), max(confint[:, 1] - pacf_x)),
            )

            lags_x[0] -= 0.5
            lags_x[-1] += 0.5
            ax.fill_between(
                lags_x, confint[:, 0] - pacf_x, confint[:, 1] - pacf_x, alpha=0.25
            )

            gray_scale = 0.93
            ax.set_facecolor((gray_scale, gray_scale, gray_scale))
            ax.grid()

            plt.title(title)

            ax.set_facecolor((gray_scale, gray_scale, gray_scale))
        plt.gcf().tight_layout()
        plt.show()


def visualize_acf(
    pivoted_df,
    n,
    lags,
    alpha,
):
    """
    Visualize the autocorrelation function for the given time series

    Args:
        pivoted_df (pd.DataFrame): The dataframe with the time series
        n (int): The number of time series to visualize
        lags (int): The number of lags to visualize
        alpha (float): The confidence level for the confidence interval

    """

    # Extract variables
    vals = pivoted_df.values
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

            acf_ = acf(y, nlags=lags, alpha=alpha)

            # splitting acf and the intervals
            acf_x, confint = acf_[:2]
            acf_x = acf_x[1:]
            confint = confint[1:]

            lags_x = np.arange(0, lags)

            ax.vlines(lags_x, [0], acf_x)
            ax.axhline()
            ax.margins(0.05)
            title = "Autocorrelation " + str(titles[series_index])
            ax.plot(
                lags_x,
                acf_x,
                marker="o",
                markersize=5,
                markerfacecolor="red",
                markeredgecolor="red",
            )

            # ax.set_ylim(-1, 1)
            # Setting the limits
            ax.set_ylim(
                1.25 * np.minimum(min(acf_x), min(confint[:, 0] - acf_x)),
                1.25 * np.maximum(max(acf_x), max(confint[:, 1] - acf_x)),
            )

            lags_x[0] -= 0.5
            lags_x[-1] += 0.5
            ax.fill_between(
                lags_x, confint[:, 0] - acf_x, confint[:, 1] - acf_x, alpha=0.25
            )

            gray_scale = 0.93
            ax.set_facecolor((gray_scale, gray_scale, gray_scale))
            ax.grid()

            plt.title(title)

            ax.set_facecolor((gray_scale, gray_scale, gray_scale))
        plt.gcf().tight_layout()
        plt.show()


def create_features(df, format="transaction"):
    """
    Creates time series features from datetime index.
    The function creates several new columns based on the 'date' column in the input DataFrame. The new columns include:
        - 'hour': the hour of the day
        - 'dayofweek': the day of the week (integer value)
        - 'weekday': the name of the day of the week
        - 'quarter': the quarter of the year
        - 'month': the month of the year
        - 'year': the year
        - 'dayofyear': the day of the year
        - 'dayofmonth': the day of the month
        - 'weekofyear': the week of the year
        - 'date_offset': a calculated value based on the date
        - 'season': the season of the year as determined by the 'date_offset' value
    :param df: DataFrame which contains a 'date' column in datetime format
    :type df: pd.DataFrame
    :return: the input DataFrame with the added columns
    :rtype: pd.DataFrame
    """

    cat_type = CategoricalDtype(
        categories=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        ordered=True,
    )

    # Convert to transaction
    if format == "pivot":
        df = transaction_df(df, drop_zeros=False)

    # Convert to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Extract features
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekday"] = df["date"].dt.day_name()
    df["weekday"] = df["weekday"].astype(cat_type)
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofmonth"] = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.weekofyear
    df["date_offset"] = (df.date.dt.month * 100 + df.date.dt.day - 320) % 1300
    df["season"] = pd.cut(
        df["date_offset"],
        [0, 300, 602, 900, 1300],
        labels=["Spring", "Summer", "Fall", "Winter"],
    )
    return df


def plot_seasonal_features(df, type, format, by):
    """
    This function plots a boxplot of the seasonal features of a given dataframe.
    It uses the create_features function to create new time features in the dataframe.
    Then plots using seaborn.
     The plot shows the average value of the 'y' column by the values in the x_axis and hue columns.

    Args:
    df (DataFrame): The input dataframe to plot.
    type (str): The type of plot to create. Accepts 'boxplot' or 'lineplot'.
    format (str): The format of the data in the dataframe.
    by (tuple): A tuple of strings containing the x_axis and hue columns to group the data by.
    """
    # Extract
    x_axis, hue = by

    # Create features
    df = create_features(df, format=format)

    # Initialie for plot
    fig, ax = plt.subplots(figsize=(12, 7))
    gray_scale = 0.93
    ax.set_facecolor((gray_scale, gray_scale, gray_scale))

    # Plot
    if type == "boxplot":
        plot_seasonal_boxplot(df, x_axis, hue, ax)
    elif type == "lineplot":
        plot_seasonal_lineplot(df, x_axis, hue, ax)
    else:
        print('Invalid plot type. Please enter "boxplot" or "lineplot".')


def plot_seasonal_boxplot(df, x_axis, hue, ax):
    """
    Plot a seasonal boxplot of the data

    Args:
        df (pandas DataFrame): The dataframe to be plotted
        x_axis (str): The x-axis column name
        hue (str): The hue column name
        ax (matplotlib.axes._subplots.AxesSubplot): The axis to plot on
    """
    # Plot
    sns.boxplot(
        data=df.dropna(), x=x_axis, y="y", hue=hue, ax=ax, showfliers=False, linewidth=1
    )

    # Edit format
    title = f"Seasonal Boxplot: {x_axis} by {hue} average"
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel("y")
    ax.grid(axis="y")
    ax.legend(bbox_to_anchor=(1, 1))

    plt.show()


def plot_seasonal_lineplot(df, x_axis, hue, ax):
    """
    Plot a seasonal lineplot of the data

    Args:
        df (pandas DataFrame): The dataframe to be plotted
        x_axis (str): The x-axis column name
        hue (str): The hue column name
        ax (matplotlib.axes._subplots.AxesSubplot): The axis to plot on
    """
    # Plot
    temp = df.groupby(["dayofweek", "month"])["y"].mean().reset_index()
    sns.lineplot(data=temp.dropna(), x=x_axis, y="y", hue=hue, ax=ax, linewidth=1)

    # Edit format
    title = f"Seasonal Lineplot: {x_axis} by {hue} average"
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel("y")
    ax.grid(axis="y")
    ax.legend(bbox_to_anchor=(1, 1))

    plt.show()


def calendar_heatmap(df, format, by):
    """
    Creates a heatmap representation of the data in a calendar format.
    Each year is being plotted in a separate subplot.

    Args:
    df (pandas DataFrame): The data frame to be plotted.
    format (str): The format of the calendar features to be added (eg. 'month_day', 'day_of_week').
    by (tuple of str): The columns to be used for the x and y axis of the heatmap.

    Returns:
    None. Plots the calendar heatmap representation of the data.
    """
    # Ensure columns are datetime
    df.columns = pd.to_datetime(df.columns.values)

    # Take the total years
    total_years = np.unique(df.columns.year)

    # Initialize the plot
    fig, axes = plt.subplots(
        nrows=len(total_years),
        ncols=1,
        squeeze=False,
        figsize=(16, len(total_years) * 3),
    )
    # Initialize the cbar
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    gray_scale = 0.93

    # Unfold
    x_axis, y_axis = by

    # Itterate over plots
    for year, ax in zip(total_years, axes):
        # Filre columns on sample_df to keep only dates on the first year
        temp_df = df[df.columns[df.columns.year == year]]

        # Normalize all values into (-1,1) to have identical scales
        temp_df = pd.DataFrame(
            StandardScaler_custom(temp_df.values),
            index=temp_df.index,
            columns=temp_df.columns,
        )

        # Add the calendar features
        temp_df = create_features(temp_df, format=format)

        # Pivot
        temp_df = pd.pivot_table(
            temp_df, index=x_axis, values="y", columns=y_axis, aggfunc="mean"
        )

        # Add the plot
        sns.heatmap(
            temp_df,
            annot=False,
            cmap="seismic",
            linewidth=0.5,
            linecolor="white",
            cbar=False if year != total_years[-1] else True,
            cbar_ax=None if year != total_years[-1] else cbar_ax,
            ax=ax[0],
        )

        title = f"Year: {year}"
        ax[0].set_title(title, pad=20)

        ax[0].set_xlabel(y_axis, labelpad=30)
        ax[0].set_ylabel(x_axis, labelpad=30)
        ax[0].set_facecolor((gray_scale, gray_scale, gray_scale))

    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.show()
