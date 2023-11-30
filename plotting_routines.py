import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from matplotlib import colormaps
from astropy import units as u
from scipy import stats
import pandas as pd
import h5py
import os
from utils import get_snap
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter
from config import config


def plot_histogram(
    df, props, labels=None, bins=100, log=True, legend_labels=None, xlim=None
):
    scale = 0.7
    labelsize = 55 * scale
    x_tick_major_size = 16 * scale
    x_tick_major_width = 4 * scale
    x_tick_minor_size = 8 * scale
    x_tick_minor_width = 3 * scale
    fig_width = 25 * scale
    fig_height = 15 * scale
    axes_width = 3 * scale
    tick_labelsize = 35 * scale
    legendsize = 50 * scale

    if len(props) > 1:
        alpha = 0.5
    else:
        alpha = 1

    plt.rcParams["figure.figsize"] = (fig_width, fig_height)
    plt.rc("axes", linewidth=axes_width)
    plt.rc("xtick", labelsize=tick_labelsize)
    plt.rc("ytick", labelsize=tick_labelsize)

    ax = plt.axes()
    ax.tick_params(length=x_tick_major_size, width=x_tick_major_width)
    ax.tick_params(
        length=x_tick_minor_size, width=x_tick_minor_width, which="minor"
    )
    for i, prop in enumerate(props):
        if log:
            input_val = np.log10(df[prop])
        else:
            input_val = df[prop]
        if legend_labels is None:
            ax.hist(input_val, bins=100, density=True, alpha=alpha, range=xlim)
        else:
            ax.hist(
                input_val,
                bins=100,
                density=True,
                alpha=alpha,
                label=legend_labels[i],
                range=xlim,
            )

    if labels is not None:
        ax.set_xlabel(labels["x"], size=labelsize)
        ax.set_ylabel(labels["y"], size=labelsize)
    if legend_labels is not None:
        plt.legend(fontsize=legendsize)
    plt.show()
    return


def plot_fesc_dependence(
    df, prop, labels=None, log=False, modes=None, plt_labels=None
):
    scale = 0.5
    labelsize = 55 * scale
    x_tick_major_size = 16 * scale
    x_tick_major_width = 4 * scale
    x_tick_minor_size = 8 * scale
    x_tick_minor_width = 3 * scale
    fig_width = 25 * scale
    fig_height = 15 * scale
    axes_width = 3 * scale
    tick_labelsize = 35 * scale
    scattersize = 30 * scale
    legendsize = 50 * scale

    plt.rcParams["figure.figsize"] = (fig_width, fig_height)
    plt.rc("axes", linewidth=axes_width)
    plt.rc("xtick", labelsize=tick_labelsize)
    plt.rc("ytick", labelsize=tick_labelsize)

    ax = plt.axes()
    ax.tick_params(length=x_tick_major_size, width=x_tick_major_width)
    ax.tick_params(
        length=x_tick_minor_size, width=x_tick_minor_width, which="minor"
    )

    if modes is None:
        modes = ["r", "2r", "sf_r", "sf_2r"]
        plt_labels = modes

    if log:
        input_val = np.log10(df[prop])
    else:
        input_val = df[prop]

    for mode, label in zip(modes, plt_labels):
        ax.scatter(
            input_val,
            df["f_esc_" + mode],
            s=scattersize,
            label=label,
            alpha=0.5,
        )

    if labels is not None:
        ax.set_xlabel(labels["x"], size=labelsize)
        ax.set_ylabel(labels["y"], size=labelsize)
    plt.legend(fontsize=legendsize)
    plt.ylim(0, 1.05)
    plt.show()
    return


def plot_prop_dependence(
    df,
    prop_x,
    prop_y,
    labels=None,
    log_x=False,
    log_y=False,
    xlim=None,
    ylim=None,
    lin_fit=False,
):
    scale = 0.5
    labelsize = 55 * scale
    x_tick_major_size = 16 * scale
    x_tick_major_width = 4 * scale
    x_tick_minor_size = 8 * scale
    x_tick_minor_width = 3 * scale
    fig_width = 25 * scale
    fig_height = 15 * scale
    axes_width = 3 * scale
    tick_labelsize = 35 * scale
    scattersize = 30 * scale
    legendsize = 50 * scale

    plt.rcParams["figure.figsize"] = (fig_width, fig_height)
    plt.rc("axes", linewidth=axes_width)
    plt.rc("xtick", labelsize=tick_labelsize)
    plt.rc("ytick", labelsize=tick_labelsize)

    ax = plt.axes()
    ax.tick_params(length=x_tick_major_size, width=x_tick_major_width)
    ax.tick_params(
        length=x_tick_minor_size, width=x_tick_minor_width, which="minor"
    )

    df = df.dropna(subset=[prop_x, prop_y])

    if log_x and log_y:
        df = df[(df[prop_x] != 0) & (df[prop_y] != 0)]
    if log_x:
        x_val = np.log10(df[prop_x])
    else:
        x_val = df[prop_x]

    if log_y:
        y_val = np.log10(df[prop_y])
    else:
        y_val = df[prop_y]

    ax.scatter(x_val, y_val, s=scattersize)
    if lin_fit:
        m, b = np.polyfit(x_val, y_val, 1)
        x_grid = np.linspace(np.min(x_val), np.max(x_val), 100)
        ax.plot(x_grid, m * x_grid + b, color="black", linewidth=4)

    if labels is not None:
        ax.set_xlabel(labels["x"], size=labelsize)
        ax.set_ylabel(labels["y"], size=labelsize)

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.show()
    return


def describe_to_latex(desc_obj):
    print(r"\begin{tabular}{|c|c|}")
    for name, element in zip(desc_obj.index, desc_obj):
        print(r"\hline")
        name = name.replace(r"%", r"\%")
        print(name, "&", "{:.2f}".format(element), r"\\")
    print(r"\hline")
    print(r"\end{tabular}")
    return


def get_scatter(
    df,
    halo_prop="M_star_sun",
    bins=30,
    threshold=1e-2,
    y_axis="f_esc",
    lum_weighted=False,
):
    x_values = df.loc[:, halo_prop]
    edges = np.logspace(
        np.log10(x_values.min()), np.log10(x_values.max()), bins
    )

    means = []
    quantile16 = []
    quantile84 = []
    error = []
    centers = []
    variance = []
    frac_small_arr = []

    skip_next = 0
    for i in range(len(edges) - 1):
        if skip_next > 0:
            skip_next -= 1
            continue
        sub_fesc = df[
            (edges[i] * (1 - 1e-10) < df[halo_prop])
            & (df[halo_prop] < edges[i + 1])
        ][y_axis]
        center = np.exp((np.log(edges[i + 1]) + np.log(edges[i])) / 2.0)

        add_bins = 1
        while len(sub_fesc) < 10:
            if (i + add_bins) >= (len(edges)):
                break
            sub_fesc = df[
                (edges[i] * (1 - 1e-10) < df[halo_prop])
                & (df[halo_prop] < edges[i + add_bins])
            ][y_axis]
            center = np.exp(
                (np.log(edges[i + add_bins]) + np.log(edges[i])) / 2.0
            )
            add_bins += 1
            skip_next += 1

        means.append(sub_fesc.mean())
        centers.append(center)
        quantile16.append(sub_fesc.quantile(0.16))
        quantile84.append(sub_fesc.quantile(0.84))
        error.append(sub_fesc.std() / np.sqrt(sub_fesc.shape[0]))
        variance.append(sub_fesc.var())
        frac_small_arr.append((sub_fesc < threshold).sum() / len(sub_fesc))

    means = np.array(means)
    quantile16 = np.array(quantile16)
    quantile84 = np.array(quantile84)
    error = np.array(error)
    centers = np.array(centers)
    variance = np.array(variance)
    frac_small_arr = np.array(frac_small_arr)
    return (
        centers,
        means,
        quantile16,
        quantile84,
        error,
        variance,
        frac_small_arr,
    )


def plot_scatter(
    df,
    halo_prop="M_star_sun",
    bins=30,
    threshold=1e-2,
    include_frac=False,
    lum_weighted=False,
    include_var=False,
    lin=False,
):
    linewidth = 4
    upper_y_threshold = 1.0
    lower_y_threshold_lin = 0.0
    size_label = 50
    x_tick_major_size = 16
    x_tick_major_width = 4
    x_tick_minor_size = 8
    x_tick_minor_width = 3
    legend_fontsize = 35
    legend_loc = "upper right"
    fig_width = 15
    fig_height = 15
    axes_width = 3
    tick_labelsize = 35
    y_lim_low = 0
    y_lim_high = 1.2

    scatter_size = 10
    alpha = 0.5

    var_lim_low = 1e-3
    var_lim_high = 0.3

    if halo_prop == "M_star_sun":
        x_label = "$M_{\star} [\log(M_{\u2609})]$"
        plot_col = 1
        prop = halo_prop
    else:
        raise ValueError("Only defined for the properties 'M_star_sun'")

    y_axis_sfr = "f_esc_r"
    y_axis_bpass = "f_esc_sf_r"
    median_label_sfr = r"$\langle f_\mathrm{esc}(\mathrm{SFR}) \rangle $"
    median_label_bpass = r"$\langle f_\mathrm{esc}(\mathrm{BPASS}) \rangle $"
    y_label = "$f_\mathrm{esc}$"
    var_label = "$\mathrm{var}(f_\mathrm{esc})$"
    low_esc_label = "$P(f_\mathrm{esc}<10^{-2})$"

    (
        centers_sfr,
        means_sfr,
        _,
        _,
        error_sfr,
        variance_sfr,
        frac_small_arr_sfr,
    ) = get_scatter(
        df, halo_prop=prop, bins=bins, y_axis=y_axis_sfr, threshold=threshold
    )

    (
        centers_bpass,
        means_bpass,
        _,
        _,
        error_bpass,
        variance_bpass,
        frac_small_arr_bpass,
    ) = get_scatter(
        df, halo_prop=prop, bins=bins, y_axis=y_axis_bpass, threshold=threshold
    )

    _ = plt.figure()
    plt.subplots_adjust(hspace=0.001)
    plt.subplots_adjust(wspace=0.001)

    if include_var and include_frac:
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    elif include_var or include_frac:
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    else:
        ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=2)

    x_bins_sfr = np.log10(centers_sfr)
    ax1.plot(
        x_bins_sfr,
        means_sfr,
        linewidth=linewidth,
        color="black",
        label=median_label_sfr,
    )
    ax1.scatter(
        np.log10(df["M_star_sun"]),
        df["f_esc_r"],
        s=scatter_size,
        alpha=alpha,
        label="SFR",
    )
    x_bins_bpass = np.log10(centers_bpass)
    ax1.plot(
        x_bins_bpass,
        means_bpass,
        linewidth=linewidth,
        color="black",
        linestyle="--",
        label=median_label_bpass,
    )
    ax1.scatter(
        np.log10(df["M_star_sun"]),
        df["f_esc_sf_r"],
        s=scatter_size,
        alpha=alpha,
        label="BPASS",
    )

    if lin:
        ax1.set_ylim(lower_y_threshold_lin, upper_y_threshold)
    else:
        ax1.set_yscale("log")
        # ax1.set_ylim(threshold, upper_y_threshold)
    ax1.set_ylabel(y_label, size=size_label)
    ax1.tick_params(length=x_tick_major_size, width=x_tick_major_width)
    ax1.tick_params(
        length=x_tick_minor_size, width=x_tick_minor_width, which="minor"
    )
    ax1.legend(fontsize=legend_fontsize)  # , loc=legend_loc, ncol=2)
    ax1.set_rasterization_zorder(-15)
    ax1.set_xlim(5)
    ax1.axes.xaxis.set_visible(False)

    if include_var:
        if include_frac:
            ax2 = plt.subplot2grid((4, 1), (2, 0), sharex=ax1)
        else:
            ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
        ax2.plot(
            np.log10(centers_sfr),
            variance_sfr,
            linewidth=linewidth,
            color="black",
        )
        ax2.plot(
            np.log10(centers_bpass),
            variance_bpass,
            linewidth=linewidth,
            color="black",
            linestyle="--",
        )
        ax2.set_ylabel(var_label, size=size_label)
        ax2.set_xlabel(x_label, size=size_label)
        ax2.set_yscale("log")
        ax2.set_ylim(var_lim_low, var_lim_high)
        ax2.tick_params(length=x_tick_major_size, width=x_tick_major_width)
        ax2.tick_params(
            length=x_tick_minor_size, width=x_tick_minor_width, which="minor"
        )

    if include_frac:
        if include_var:
            ax3 = plt.subplot2grid((4, plot_col), (3, 0), sharex=ax2)
        else:
            ax3 = plt.subplot2grid((3, plot_col), (2, 0), sharex=ax1)
        ax3.plot(
            np.log10(centers_sfr),
            frac_small_arr_sfr,
            linewidth=linewidth,
            color="black",
        )
        ax3.plot(
            np.log10(centers_bpass),
            frac_small_arr_bpass,
            linewidth=linewidth,
            color="black",
            linestyle="--",
        )
        ax3.legend(fontsize=legend_fontsize, loc=legend_loc)
        ax3.set_ylabel(low_esc_label, size=size_label)
        ax3.set_xlabel(x_label, size=size_label)
        ax3.tick_params(length=x_tick_major_size, width=x_tick_major_width)
        ax3.tick_params(
            length=x_tick_minor_size, width=x_tick_minor_width, which="minor"
        )
        ax3.set_ylim(y_lim_low, y_lim_high)

    plt.rcParams["figure.figsize"] = (fig_width, fig_height)
    plt.rc("axes", linewidth=axes_width)
    plt.rc("xtick", labelsize=tick_labelsize)
    plt.rc("ytick", labelsize=tick_labelsize)
    plt.show()
    return


def get_hist_scatter(df, prop, y_axis="f_esc_r", bin_width=0.1, mode="median"):
    f_esc = df.loc[:, y_axis]
    x_values = df.loc[:, "M_star_sun"]
    x_prop = "M_star_sun"

    x_values_log = np.log10(x_values)
    log_edges = np.arange(x_values_log.min(), x_values_log.max(), bin_width)
    edges = np.power(10, log_edges)
    prop_norm = {}
    prop_norm[prop] = np.zeros(df[x_prop].size, dtype="float32")
    start_index = 0
    end_index = 0
    for i in range(len(edges) - 1):
        column = df[
            (edges[i] * (1 - 1e-10) < df[x_prop]) & (df[x_prop] < edges[i + 1])
        ][prop]
        end_index += len(column)
        if mode == "median":
            norm_val = column.median()
        elif mode == "mean":
            norm_val = column.mean()

        new_data = column / norm_val

        prop_norm[prop][start_index:end_index] = new_data
        start_index = end_index
    prop_norm[prop] = np.array(prop_norm[prop])
    return x_values, f_esc, prop_norm


def filter_function(array):
    if len(array) > 0:
        return np.median(array)
    else:
        return np.nan


def get_levels(hist_cont, thresholds):
    levels = []
    counts = np.sort(hist_cont.flatten())[::-1]
    value_thresholds = counts.sum() * np.array(thresholds)
    for threshold in value_thresholds:
        count_sum = 0
        i = 0
        while count_sum < threshold:
            count_sum += counts[i]
            i += 1
        levels.append(counts[i])
    return levels


def plot_hist_color(
    df,
    prop,
    y_axis="f_esc_r",
    bin_width=0.1,
    mode="median",
    label=None,
    small_scale=False,
    bins=[50, 50],
    levels=[10, 30],
):
    x_labelsize = 50
    y_labelsize = 50

    length_major_ticks = 16
    length_minor_ticks = 8
    width_minor_ticks = 3
    width_major_ticks = 4
    labelsize_x_ticks = 35
    labelsize_y_ticks = 35

    colorbar_labelsize = 50
    colorbar_ticklabelsize = 35

    axes_width = 3

    figure_width = 20
    figure_height = 20
    x_prop = "M_star_sun"

    if small_scale:
        v_min = 0.9
        v_max = 1.1
    else:
        v_min = 0.5
        v_max = 2.0

    v_center = 1.0

    x_label = "$M_{\star} [\log(M_{\u2609})]$"

    df.sort_values(by=x_prop, inplace=True)
    x_values, f_esc, prop_norm = get_hist_scatter(
        df, prop=prop, bin_width=bin_width, mode=mode, y_axis=y_axis
    )

    y_label = "$f_\mathrm{esc}$"

    f, ax = plt.subplots()

    color_data = prop_norm[prop]
    if label is None:
        bar_label = f"$\Delta ${prop}"
    else:
        bar_label = label

    col_norm = colors.TwoSlopeNorm(vmin=v_min, vcenter=v_center, vmax=v_max)

    hist, xedges, yedges, binnumber = stats.binned_statistic_2d(
        np.log10(x_values),
        f_esc,
        values=color_data,
        statistic=filter_function,
        bins=[bins[0], bins[1]],
    )  # , range=[[6,8],[0,1]])
    (
        hist_cont,
        xedges_cont,
        yedges_cont,
        binnumber_cont,
    ) = stats.binned_statistic_2d(
        np.log10(x_values),
        f_esc,
        values=color_data,
        statistic="count",
        bins=[bins[0], bins[1]],
    )  # , range=[[6,8],[0,1]])
    levels = get_levels(hist_cont, thresholds=[0.954, 0.683])
    x_grid, y_grid = np.meshgrid(xedges, yedges)
    cont_centers_x = (xedges_cont[1:] + xedges_cont[:-1]) / 2
    cont_centers_y = (yedges_cont[1:] + yedges_cont[:-1]) / 2

    x_grid_cont, y_grid_cont = np.meshgrid(cont_centers_x, cont_centers_y)
    subfig = ax.pcolormesh(
        x_grid, y_grid, hist.T, norm=col_norm, cmap=plt.get_cmap("coolwarm")
    )
    levels = levels
    ax.contour(
        cont_centers_x,
        cont_centers_y,
        hist_cont.T,
        levels=levels,
        linewidths=5,
        linestyles=["solid", "dashed"],
        colors="black",
    )

    ax.set_xlabel(x_label, size=x_labelsize)
    ax.set_ylabel(y_label, size=y_labelsize)

    # Save the scatter as a rasterization graphic to make the plots smaller
    ax.set_rasterization_zorder(-15)

    ax.tick_params(length=length_major_ticks, width=width_major_ticks)
    ax.tick_params(
        length=length_minor_ticks, width=width_minor_ticks, which="minor"
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = f.colorbar(subfig, cax=cax)
    cbar.set_label(
        bar_label, size=colorbar_labelsize
    )  # Save the scatter as a rasterization graphic to make the plots smaller
    ax.set_rasterization_zorder(-15)
    cbar.ax.tick_params(labelsize=colorbar_ticklabelsize)

    xticklabels = ax.get_xticklabels()
    plt.setp(xticklabels, visible=True)
    plt.rc("axes", linewidth=axes_width)
    plt.rc("xtick", labelsize=labelsize_x_ticks)
    plt.rc("ytick", labelsize=labelsize_y_ticks)

    plt.rcParams["figure.figsize"] = (figure_width, figure_height)
    plt.tight_layout(rect=(0, 0, 1, 0.7))

    plt.show()
    return


def plot_parameters(params, multiple=False):
    parameters = {}
    parameters["labelsize"] = 50

    parameters["titlesize"] = 30
    parameters["multiple_titlesize"] = 35
    parameters["length_major_ticks"] = 16
    parameters["length_minor_ticks"] = 8
    parameters["width_minor_ticks"] = 3
    parameters["width_major_ticks"] = 4
    parameters["labelsize_ticks"] = 35

    parameters["colorbar_labelsize"] = 50
    parameters["colorbar_ticklabelsize"] = 30
    parameters["colorbar_labelsize_multiple"] = 40
    parameters["colorbar_ticklabelsize_multiple"] = 30

    parameters["axes_width"] = 3

    parameters["figure_width"] = 15
    parameters["figure_height"] = 15

    parameters["height_per_image"] = 6
    parameters["width_per_image"] = 6

    # parameters["x_label"] = r"$\log(n) [\mathrm{cm}^{-3}]$"
    # parameters["y_label"] = r"$\log(T) [\mathrm{K}]$"
    # parameters["bar_label"] = r"$\log(\frac{M}{M_\mathrm{max}})$"

    # parameters[
    #     "x_label_convergence"
    # ] = r"Grid size $[\log(\mathrm{avg\_separation})]$"
    parameters["x_label_convergence"] = r"Grid cells"
    parameters["y_label_convergence"] = r"$f_\mathrm{esc}$"

    parameters["nx"] = 45
    parameters["ny"] = 30

    parameters["v_min"] = -4
    parameters["v_max"] = 0

    parameters["x_lim_min"] = -4.8
    parameters["x_lim_max"] = 0

    parameters["y_lim_min"] = 2.8
    parameters["y_lim_max"] = 6.4

    parameters["legendsize"] = 30

    parameters["linewidth"] = 3
    parameters["capsize"] = 10
    parameters["capwidth"] = 3

    if params is not None:
        for element in params:
            parameters[element] = params[element]
    return parameters


def get_col_norm(parameters):
    # v_min = np.log10(hist[hist > 0].min())
    # v_max = np.log10(hist.max())
    v_min = parameters["v_min"]
    v_max = parameters["v_max"]
    v_center = (v_max + v_min) / 2

    col_norm = colors.TwoSlopeNorm(vmin=v_min, vcenter=v_center, vmax=v_max)
    return col_norm


def set_ax_params(ax, parameters, multiple=False):
    # ax.set_xlabel(parameters["x_label"], size=parameters["x_labelsize"])
    # ax.set_ylabel(parameters["y_label"], size=parameters["y_labelsize"])

    # ax.set_xlim(parameters["x_lim_min"], parameters["x_lim_max"])
    # ax.set_ylim(parameters["y_lim_min"], parameters["y_lim_max"])

    ax.tick_params(
        length=parameters["length_major_ticks"],
        width=parameters["width_major_ticks"],
    )
    ax.tick_params(
        length=parameters["length_minor_ticks"],
        width=parameters["width_minor_ticks"],
        which="minor",
    )
    if multiple:
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=parameters["colorbar_ticklabelsize_multiple"],
        )
    else:
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=parameters["colorbar_ticklabelsize"],
        )

    # change all spines
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(parameters["axes_width"])
    return


def create_color_bar(
    f,
    ax,
    parameters,
    subfig,
    label=None,
    multiple=False,
    ax_is_cbar=False,
    horizontal=False,
    gap=False,
    prop="f_esc",
):
    if ax_is_cbar:
        if horizontal:
            cbar = f.colorbar(subfig, cax=ax)
            # ax.xaxis.set_ticks_position("top")
            # ax.xaxis.set_label_position("top")

        else:
            cbar = f.colorbar(subfig, cax=ax, orientation="horizontal")
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")

    else:
        divider = make_axes_locatable(ax)
        if gap:
            pad = 10
        else:
            pad = 0.15
        cax = divider.append_axes("right", size="5%", pad=pad)
        cbar = f.colorbar(subfig, cax=cax)

    if label is not None:
        if multiple:
            size = parameters["colorbar_labelsize_multiple"]
        else:
            size = parameters["colorbar_labelsize"]
        cbar.set_label(label, size=size, labelpad=18)

    if multiple:
        ticksize = parameters["colorbar_ticklabelsize_multiple"]
    else:
        ticksize = parameters["colorbar_ticklabelsize"]
    cbar.ax.tick_params(labelsize=ticksize)
    if prop != "f_esc":
        if horizontal:
            cbar.ax.set_yticks([0.5, 0.75, 1, 1.5, 2], labelsize=ticksize)
        else:
            cbar.ax.set_xticks([0.5, 0.75, 1, 1.5, 2], labelsize=ticksize)

    return


def plot_multiple_histograms(maps, params=None):
    parameters = plot_parameters(params, multiple=True)

    col_norm = get_col_norm(parameters)
    n_columns = 4

    image_columns = n_columns
    image_rows = int(np.ceil(len(maps.keys()) / n_columns))
    figsize = (
        parameters["width_per_image"] * image_columns,
        parameters["height_per_image"] * image_rows,
    )
    fig, axs = plt.subplots(
        ncols=image_columns,
        nrows=image_rows,
        gridspec_kw={"hspace": 0.2, "wspace": 0.2},
        figsize=figsize,
    )
    lin_props = ["f_esc", "f_g", "f_g_crit"]
    for i, prop in enumerate(maps.keys()):
        column = int(i % n_columns)
        row = int(i // n_columns)
        ax = axs[row, column]
        if prop in lin_props:
            quant = maps[prop]
        else:
            quant = np.log10(maps[prop])
        subfig = ax.pcolormesh(quant, cmap=colormaps["inferno"])
        # subfig = ax.pcolormesh(quant, cmap=colormaps["hotcold"])

        set_ax_params(ax, parameters)
        ax.set_title(prop, fontsize=parameters["titlesize"])
        create_color_bar(fig, ax, parameters, subfig, ax_is_cbar=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # if len(maps.keys()) % 2 == 1:
    #     fig.delaxes(axs[image_rows - 1, 1])
    return


def get_quantity_array(df, prop, scales):
    prop_dict = {}
    for idx in df.index:
        prop_dict[idx] = []
        for scale in scales:
            column = prop + "_" + str(scale)
            quant = df.loc[idx, column]
            prop_dict[idx].append(quant)
        prop_dict[idx] = np.array(prop_dict[idx])
    return prop_dict


def get_scales(df, prop):
    scales = []
    scale_names = []
    for column in df.columns:
        if column.startswith(prop):
            try:
                scales.append(float(column[len(prop) + 1 :]))
                scale_names.append(column[len(prop) + 1 :])
            except ValueError:
                continue
    zipped = sorted(zip(scales, scale_names), key=lambda x: x[0])
    scales, scale_names = zip(*zipped)
    return list(scales), list(scale_names)


def plot_convergence(df, prop="f_esc", params=None, log=True, weights=None):
    scales, scale_names = get_scales(df, prop)
    parameters = plot_parameters(params)
    prop_dict = get_quantity_array(df, prop, scale_names)
    ion_dict = get_quantity_array(df, "Ion_em", scale_names)
    f, ax = plt.subplots(figsize=(20, 20))
    all_esc = []
    all_ion = []
    scales = sorted(scales)
    for idx in prop_dict:
        ax.scatter(scales, prop_dict[idx], s=100)
        all_esc.append(prop_dict[idx])
        all_ion.append(ion_dict[idx])

    all_esc = np.array(all_esc)
    mean_values = np.nanmean(all_esc, axis=0)
    all_ion = np.array(all_ion)
    label_mean = r"$<f_\mathrm{esc}>$"

    ax.plot(scales, mean_values, linewidth=5, color="black", label=label_mean)
    if weights is not None:
        all_ion_weighted = (all_ion.T * weights).T
        mean_weighted = np.nansum(all_esc * all_ion_weighted, axis=0) / np.sum(
            all_ion_weighted, axis=0
        )
        label_mean_weighted = r"$<f_\mathrm{esc}>_{n*Q_0}$"
        ax.plot(
            scales,
            mean_weighted,
            linewidth=5,
            linestyle="--",
            color="black",
            label=label_mean_weighted,
        )
    ax.set_xlabel(
        parameters["x_label_convergence"], size=parameters["labelsize"]
    )
    ax.set_ylabel(
        parameters["y_label_convergence"], size=parameters["labelsize"]
    )
    set_ax_params(ax, parameters)

    # ax.set_xscale("log")
    if log:
        ax.set_ylim(1e-3, 1)
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 0.15)
    ax.set_xlim(
        0.3,
    )

    ax.legend(fontsize=parameters["legendsize"])
    return


def get_z_edges(df):
    z = np.array(df["z"].unique())
    pre_bin_edges = list((z[:-1] + z[1:]) / 2)
    bin_edge_0 = 2 * z[0] - pre_bin_edges[0]
    bin_edge_f = 2 * z[-1] - pre_bin_edges[-1]
    bin_edges = [bin_edge_0]
    bin_edges.extend(pre_bin_edges)
    bin_edges.append(bin_edge_f)
    return bin_edges


def log_mean(quant):
    return np.log10(np.mean(quant))


def get_label(prop):
    prop_labels = {
        "M_gas_sun_log": r"$\log \left(M_\mathrm{gas}/\mathrm{M}_\odot \right)$",
        "M_star_sun_log": r"$\log \left( M_\star/\mathrm{M}_\odot \right)$",
        "SFR": r"$\log \left( \mathrm{SFR}/\mathrm{M}_\odot \mathrm{yr}^{-1} \right)$",
        "f_g": r"$f_g$",
        "f_g_crit": r"$f_{g, \mathrm{crit}}$",
        "Column_height": r"$\log(H/\mathrm{cm})$",
        "Columns_dens": r"$\log \left(N_0/\mathrm{cm}^{-2} \right)$",
        "Columns_dens_stroemgren": r"$\log \left(N_S/\mathrm{cm}^{-2} \right)$",
        "Metallicity": r"$\log(Z)$",
        "U": r"$\log(U)$",
        "N_d": r"$N_d$",
        # "N_ratio": r"$N_0/N_d$",
        "N_ratio": r"$\tau_\mathrm{d}$",
        "z": "z",
        "f_esc": r"$\langle f_\mathrm{esc} \rangle$",
        "f_esc_model": r"$\langle f_\mathrm{esc,fit} \rangle$",
        "n_gas": r"$\log \left( \frac{n_\mathrm{gas}}{\mathrm{cm}^{-3}} \right)$",
        "Sigma_SFR": r"\log \left( \frac{\rangle \Sigma_\mathrm{SFR} \langle}{M_\odot \mathrm{yr}^{-1} \mathrm{kpc}^{-2}} \right)",
        "U1": r"$U_1$",
        # "N_S_ratio": r"$N_0/N_S$",
        "N_S_ratio": r"$\tau_\mathrm{HI}$",
        "Ion_flux": r"$\log (F_i/\mathrm{cm}^{-2})$",
        "TimeMajorMerger": r"$t_\mathrm{merger}[\mathrm{Myr}]$",
        "TimeMinorMerger": r"$t_\mathrm{merger}[\mathrm{Myr}]$",
        "TimeRecentMerger": r"$t_\mathrm{merger}[\mathrm{Myr}]$",
        "sSFR": r"$\log \left( \frac{\mathrm{sSFR}}{\mathrm{yr}^{-1}} \right)$",
        "sZ": r"$\log(\frac{Z/M_\star}{\mathrm{M}_\odot^{-1}})$",
        "MgasMstar": r"$\log(M_\mathrm{gas}/ M_\star)$",
        "color_prop": r"$\log(H/\mathrm{cm})$",
        "MassStarLog": r"$\log \left(\frac{M_\star}{M_\odot} \right)$",
        "AverageColumnDens": r"$\log(N_0/\mathrm{cm}^{-2})$",
        "Dist_5": r"$\log( d_5 / \mathrm{kpc})$ ",
        "Dist_5_sim": r"$\log( d_5 / \mathrm{kpc})$ ",
        "v_sigma": r"$v_\mathrm{max}/\sigma_v$",
        "flow": r"$\mathcal{F}$",
        "L_M": r"$L_\mathrm{gas}/M_\mathrm{gas}[\mathrm{cm}^2\mathrm{s}^{-1}]$",
        "Offset_pc": r"$\log( \Delta_c/\mathrm{pc})$",
        "Q0": r"$\log(Q_{0,\mathrm{RT}}/\mathrm{s}^{-1})$",
        "analytic_Q0": r"$\log(Q_{0}/\mathrm{s}^{-1})$",
        # "f_esc": r"$f_\mathrm{esc, RT}$",
        "analytic_fesc": r"$f_\mathrm{esc}$",
    }
    if prop in prop_labels:
        return prop_labels[prop]
    else:
        return None


def log_count(prop):
    return np.log10(len(prop))


def masked_mean(prop):
    return np.mean(np.ma.masked_invalid(prop))


def get_histogram(
    df,
    x_values,
    y_values,
    bins,
    color_prop="f_esc",
    em_weighted=False,
    statistic="mean",
):
    if em_weighted:
        df["weighted"] = df[color_prop] * df["Ion_em"]
        hist_esc, *_ = stats.binned_statistic_2d(
            x_values,
            y_values,
            values=df["weighted"],
            statistic="sum",
            bins=bins,
        )
        hist_em, *_ = stats.binned_statistic_2d(
            x_values,
            y_values,
            values=df["Ion_em"],
            statistic="sum",
            bins=bins,
        )
        hist = hist_esc / hist_em
    else:
        hist, *_ = stats.binned_statistic_2d(
            x_values,
            y_values,
            values=df[color_prop],
            statistic=statistic,
            bins=bins,
        )

    hist_cont, xedges_cont, yedges_cont, _ = stats.binned_statistic_2d(
        x_values,
        y_values,
        values=df[color_prop],
        statistic="count",
        bins=bins,
    )

    if statistic == "count":
        pass
        # hist = np.log10(hist)
    return hist, hist_cont, xedges_cont, yedges_cont


def get_color_limits(prop, statistic="mean", maps=False):
    limits = {
        "f_esc_model": (0.0, 0.1, 0.2),
        "f_esc": (0.0, 0.1, 0.2),
        "f_g_crit": (0.0, 0.5, 1.0),
        "M_star_sun_log": (5.8, 8, 10),
        # Only works for column height
        # "color_prop": (21, 21.3, 21.6),
        "color_prop": (-4, -3, -2),
        "TimeMajorMerger": (0, 150, 300),
        "Column_height": (1.9e21, 2.15e21, 2.4e21),
        "Metallicity": (1e-6, 1e-4, 1e-2),
    }
    limits_maps = {
        "f_esc": (0.0, 0.5, 1.0),
        "f_g_crit": (0.0, 0.5, 1.0),
        "SFR": (-8, -6, -4),
        "N_S_ratio": (0.5, 1, 2),
        "N_ratio": (0.5, 1, 2),
        "Ion_flux": (5, 6.5, 8),
    }
    if maps:
        if prop in limits_maps:
            return limits_maps[prop]
        else:
            return (None, None, None)
    else:
        if statistic == "count":
            return (0, 500, 1000)
            # return (0, 2.5, 5)
        elif prop in limits:
            return limits[prop]
        else:
            return (None, None, None)


def prop_prop_histogram(
    df,
    prop_x,
    prop_y,
    color_prop="f_esc",
    statistic="mean",
    em_weighted=False,
    log_x=True,
    log_y=True,
    bins_x=30,
    bins_y=30,
    params=None,
    color_log=False,
    contour=True,
    add_line=False,
    line_log=False,
    line_x_log=False,
):
    if color_log:
        df["color_prop"] = np.log10(df[color_prop])
        color_prop = "color_prop"
    parameters = plot_parameters(params)

    # filter = (df.M_star_sun_log > 6.8) & (df.M_star_sun_log < 7.2)
    # df = df[filter]
    # Clean df
    # df.dropna(subset="f_g_crit", inplace=True)

    if (prop_x == "TimeMajorMerger") or (prop_x == "TimeMajorMerger"):
        df = df.replace({"TimeMajorMerger": [np.inf, -np.inf]}, np.nan).dropna(
            subset="TimeMajorMerger", axis=0
        )
    # df = df.replace([np.inf, -np.inf], np.nan).dropna(
    #     subset="TimeMajorMerger", axis=0
    # )
    x_values = df[prop_x]
    y_values = df[prop_y]
    if log_x:
        x_values = np.ma.masked_invalid(np.log10(x_values))
    if log_y:
        y_values = np.ma.masked_invalid(np.log10(y_values))

    if prop_x == "z":
        x_edges = get_z_edges(df)[::-1]
    else:
        x_edges = np.linspace(x_values.min(), x_values.max(), bins_x)
    y_edges = np.linspace(y_values.min(), y_values.max(), bins_y)

    hist, hist_cont, xedges_cont, yedges_cont = get_histogram(
        df,
        x_values,
        y_values,
        bins=[x_edges, y_edges],
        color_prop=color_prop,
        em_weighted=em_weighted,
        statistic=statistic,
    )
    if prop_x == "z":
        x_edges = np.arange(len(x_edges))
        cont_centers_x = np.arange(len(xedges_cont) - 1)
    else:
        cont_centers_x = (xedges_cont[1:] + xedges_cont[:-1]) / 2
    cont_centers_y = (yedges_cont[1:] + yedges_cont[:-1]) / 2
    x_grid, y_grid = np.meshgrid(x_edges, y_edges)
    vmin, vcenter, vmax = get_color_limits(color_prop, statistic)
    col_norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    # f, ax = plt.subplots(
    #     figsize=[parameters["figure_width"], parameters["figure_height"]]
    # )
    f, axs = plt.subplots(
        ncols=1,
        nrows=2,
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.1,  # 0.3 * 0.75 * len(props_of_interest),
            # "width_ratios": [24, 1],
            "height_ratios": [1.5, 24],
        },
        figsize=[parameters["figure_width"], parameters["figure_height"]],
    )
    ax = axs[1]
    cax = axs[0]
    subfig = ax.pcolormesh(
        x_grid, y_grid, hist.T, norm=col_norm, cmap=plt.get_cmap("inferno")
    )
    # subfig = ax.pcolormesh(
    #     x_grid, y_grid, hist.T, norm=col_norm, cmap=plt.get_cmap("plasma")
    # )
    levels = get_levels(hist_cont, thresholds=[0.954, 0.683])
    if contour:
        ax.contour(
            cont_centers_x,
            cont_centers_y,
            hist_cont.T,
            levels=levels,
            linewidths=4,
            linestyles=["dotted", "dashed", "solid"],
            colors="lightblue",
        )
    if statistic == "count":
        color_label = r"$\log(\mathrm{counts})$"
    else:
        color_label = get_label(color_prop)
    create_color_bar(
        f,
        cax,
        parameters,
        subfig,
        label=color_label,
        gap=True,
        ax_is_cbar=True,
        horizontal=False,
    )

    if add_line:
        lower_edge = x_values.min()
        upper_edge = x_values.max()
        x_edges = np.linspace(lower_edge, upper_edge, 20)
        if line_x_log:
            log_edges = np.linspace(lower_edge, upper_edge, 20)
            x_edges = 10**log_edges
            x_centers = (log_edges[1:] + log_edges[:-1]) / 2
        else:
            x_edges = np.linspace(lower_edge, upper_edge, 20)
            x_centers = (x_edges[1:] + x_edges[:-1]) / 2

        # df.replace([0, -np.inf], np.nan).dropna(subset="Metallicity", axis=0)
        df["prop_bins"] = pd.cut(df[prop_x], x_edges, include_lowest=True)
        groups = df.groupby(["prop_bins"])[prop_y]

        y_prop_std = groups.std()
        y_prop_count = groups.count()
        if line_log:
            y_prop_means = groups.apply(
                lambda x: np.log10(x.mean()) if x.count() > 5 else np.nan
            )
            y_prop_err = 1 / groups.mean() * y_prop_std / np.sqrt(y_prop_count)

        else:
            y_prop_means = groups.apply(
                lambda x: x.mean() if x.count() > 5 else np.nan
            )
            y_prop_err = y_prop_std / np.sqrt(y_prop_count)
        ax2 = ax.twinx()
        # prop = r"$M_\mathrm{gas}/M_\star$"
        prop = r"$\langle (\log( d_5 ) \rangle$"
        # prop = r"$\langle \Delta_c \rangle$"

        color = "lime"
        ax2.errorbar(
            x_centers,
            y_prop_means,
            yerr=y_prop_err,
            capsize=parameters["capsize"],
            capthick=parameters["capwidth"],
            linewidth=parameters["linewidth"] * 1.5,
            color=color,
            label=rf"$\langle${prop}$\rangle(T_\mathrm{{merger}})$",
        )
        ax2.set_ylabel(
            prop,  # /\mathrm{{cm}})$",
            size=parameters["labelsize"],
        )
        ax2.set_ylim(2.7, 3.2)
        ax2_color = "green"
        ax2.yaxis.label.set_color(ax2_color)
        ax2.tick_params(axis="y", colors=ax2_color)
        # ax2.set_ylim(
        #     y_prop_means.min() - 0.2 * y_range,
        #     y_prop_means.max() + 0.2 * y_range,
        # )

        set_ax_params(ax2, parameters)
        # ax2.legend(fontsize=parameters["legendsize"])
        ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax2.yaxis.offsetText.set_fontsize(parameters["colorbar_ticklabelsize"])

    if prop_x == "z":
        labels = [f"{x:.1f}" for x in df.z.unique()[::-1]]
        ax.set_xticks(np.arange(len(x_edges) - 1)[::2] + 0.5)
        ax.set_xticklabels(labels[::2])

    ax.set_xlabel(get_label(prop_x), size=parameters["labelsize"])
    ax.set_ylabel(get_label(prop_y), size=parameters["labelsize"])
    # ax.legend(fontsize=parameters["legendsize"])
    set_ax_params(ax, parameters)

    # ax.set_xlim(-9, -7.5)
    # ax.set_ylim(-1, 3.6)
    return


def fesc_Mstar(
    df,
    mass_bins=30,
    em_weighted=False,
    x_prop="M_star_sun_log",
    skip=1,
    params=None,
):
    parameters = plot_parameters(params)
    g_to_msun = (1 * u.g).to(u.M_sun)
    avg_prop = "f_esc_model"
    df.dropna(subset=avg_prop, inplace=True)
    if x_prop == "TimeMajorMerger":
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    x_values = df[x_prop]
    x_edges = np.linspace(x_values.min(), x_values.max(), mass_bins)
    x_centers = (x_edges[1:] + x_edges[:-1]) / 2
    df["mass_bins"] = pd.cut(df[x_prop], x_edges)

    z_values = df["z"].unique()[::-1]
    if em_weighted:
        df["f_esc_weighted"] = df[avg_prop] * df["Ion_em"]
        f_esc_weighted = df.groupby(["z", "mass_bins"])["f_esc_weighted"].sum()
        Ion_em = df.groupby(["z", "mass_bins"])["Ion_em"].sum()
        f_esc_means = f_esc_weighted / Ion_em
    else:
        groups = df.groupby(["z", "mass_bins"])[avg_prop]
        f_esc_means = groups.mean()
        f_esc_std = groups.std()
        f_esc_count = groups.count()
        f_esc_err = f_esc_std / np.sqrt(f_esc_count)

    f, ax = plt.subplots(
        figsize=[parameters["figure_width"], parameters["figure_height"]]
    )
    for i, z in enumerate(z_values):
        if i % skip == 0:
            if em_weighted:
                ax.plot(
                    x_centers,
                    f_esc_means[z],
                    label=f"z={z:.1f}",
                    linewidth=parameters["linewidth"],
                )

            else:
                ax.errorbar(
                    x_centers,
                    f_esc_means[z],
                    yerr=f_esc_err[z],
                    label=f"z={z:.1f}",
                    capsize=parameters["capsize"],
                    linewidth=parameters["linewidth"],
                )

    label_x = get_label(x_prop)
    label_y = r"$\langle f_\mathrm{esc, fit} \rangle$"

    ax.set_xlabel(label_x, size=parameters["labelsize"])
    ax.set_ylabel(label_y, size=parameters["labelsize"])
    set_ax_params(ax, parameters)
    ax.legend(fontsize=parameters["legendsize"])
    if x_prop == "M_star_sun_log":
        ax.set_xlim(5.8, 10.5)
        ax.set_ylim(0, 0.25)
        ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    elif x_prop == "M_gas_sun_log":
        ax.set_xlim(6.7, 9.7)
        ax.set_ylim(0, 0.25)
    return


def get_hdf(df, z, hdf_prefix):
    all_z = df["z"].unique()
    snap_num = int(np.where(all_z == z)[0])
    base_path = config["base_path"]
    snap = get_snap(snap_num)
    hdf_name = f"{hdf_prefix}{snap_num}.hdf5"
    origin_path_hdf = os.path.join(base_path, snap, hdf_name)
    f = h5py.File(origin_path_hdf)
    return f


def get_convergence_maps(hdf, halo_idx, prop):
    maps = {}
    for key in hdf.keys():
        if key == "3":
            continue
        if key == "None":
            continue
        maps[float(key)] = hdf[key][str(halo_idx)][prop]
    return maps


def get_convergence_maps(hdf, halo_idx, prop):
    maps = {}
    for key in hdf.keys():
        if key == "3":
            continue
        if key == "None":
            continue
        maps[float(key)] = hdf[key][str(halo_idx)][prop]
    return maps


def trim_axes(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def get_range(maps):
    low = []
    high = []
    for key in maps.keys():
        low.append(np.min(maps[key]))
        high.append(np.max(maps[key]))
    return np.min(low), np.max(high)


def get_image(df, halo_num, hdf_prefix, prop, grid_size):
    halo_idx = df.loc[halo_num, "idx"]
    halo_z = df.loc[halo_num, "z"]
    hdf = get_hdf(df, halo_z, hdf_prefix)
    image = hdf[grid_size][str(halo_idx)][prop]
    return image


def get_map_sample(df, prop, hdf_prefix, grid_size, n, halo_nums):
    sample = df.sample(n)
    maps = {}
    if halo_nums is None:
        for num in sample.index:
            maps[num] = get_image(df, num, hdf_prefix, prop, grid_size)
    else:
        for num in halo_nums:
            if prop == "N_S_ratio":
                map_ns = get_image(
                    df, num, hdf_prefix, "Column_dens_stroemgren", grid_size
                )
                map_n0 = get_image(
                    df, num, hdf_prefix, "Column_dens", grid_size
                )
                maps[num] = np.array(map_n0) / np.array(map_ns)
            else:
                maps[num] = get_image(df, num, hdf_prefix, prop, grid_size)
    return maps


def set_title(
    df, type, prop, halo_num, ax, scale, props_of_interest, parameters
):
    if type == "convergence":
        if prop == "f_esc":
            f_esc = df.loc[halo_num, f"f_esc_{int(scale)}"]
            ax.set_title(
                rf"$\lambda = ${scale:.2f}, $f_\mathrm{{esc}} = ${f_esc:.2f}",
                fontsize=parameters["titlesize"],
            )
        elif prop == "Ion_flux":
            ax.set_title(
                rf"$\lambda = ${scale:.2f}: $F_i$",
                fontsize=parameters["titlesize"],
            )
    if type == "sample":
        title = ""

        for new_prop in props_of_interest:
            value = df.loc[halo_num, new_prop]
            if new_prop != "f_esc":
                label = get_label(new_prop)
                title += f"{label} = {np.log10(value):.2f}\n"
            else:
                label = r"$f_\mathrm{esc}$"
                title += f"{label} = {value*100:.0f}%\n"

        ax.set_title(title, fontsize=parameters["multiple_titlesize"], y=0.9)
    return


def plot_prop_maps(
    df,
    halo_num=None,
    hdf=None,
    hdf_prefix="gridded_maps_",
    grid_size="100",
    type="single_halo",
    skip=1,
    params=None,
    prop="f_esc",
    log=False,
    props_of_interest=["f_esc"],
    n_maps=16,
    horizontal=True,
):
    if type == "convergence":
        maps = get_convergence_maps(hdf, halo_num, prop=prop)
        maps = dict(sorted(maps.items()))

    if type == "sample":
        maps = get_map_sample(
            df, prop, hdf_prefix, grid_size, n=n_maps, halo_nums=halo_num
        )

    if type == "single_halo":
        maps = {}
        maps[halo_num] = get_image(df, halo_num, hdf_prefix, prop, grid_size)

    vmin, vcenter, vmax = get_color_limits(prop, maps=True)

    if log:
        for key in maps.keys():
            maps[key] = np.log10(maps[key])

    parameters = plot_parameters(params, multiple=True)
    col_norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    if type == "single_halo":
        figsize = (
            parameters["width_per_image"],
            parameters["height_per_image"],
        )
        fig, ax = plt.subplots(figsize=figsize)
        subfig = ax.pcolormesh(
            maps[halo_num],
            cmap=colormaps["inferno"],
            norm=col_norm,
            # cmap=colormaps["coolwarm"],
        )
        create_color_bar(
            fig,
            ax,
            parameters,
            subfig,
            multiple=False,
            ax_is_cbar=False,
            horizontal=True,
        )
    else:
        image_columns = 3
        image_rows = int(np.ceil(len(maps.keys()) / image_columns / skip))
        if horizontal:
            figsize = (
                parameters["width_per_image"] * image_columns,
                parameters["height_per_image"] * image_rows,
            )
            fig, axs = plt.subplots(
                ncols=image_columns + 1,
                nrows=image_rows,
                gridspec_kw={
                    "hspace": 0.01 * len(props_of_interest),
                    "wspace": 0.05,  # 0.3 * 0.75 * len(props_of_interest),
                    "width_ratios": [12, 12, 12, 1],
                },
                figsize=figsize,
            )
        else:
            figsize = (
                parameters["width_per_image"] * 1,
                parameters["height_per_image"] * image_columns,
            )
            fig, axs = plt.subplots(
                ncols=image_rows,
                nrows=image_columns + 1,
                gridspec_kw={
                    "hspace": 0.05,  # * len(props_of_interest),
                    "wspace": 0.05,  # 0.3 * 0.75 * len(props_of_interest),
                    "height_ratios": [12, 12, 12, 1],
                },
                figsize=figsize,
            )

        counter = 0
        for i, scale in enumerate(maps.keys()):
            if i % skip == 0:
                if type != "single_halo":
                    if horizontal:
                        column = int(i // skip % image_columns)
                        row = int(i // skip // image_columns)
                        if image_rows > 1:
                            ax = axs[row, column]
                            ax_col = axs[row, image_columns]
                        else:
                            ax = axs[column]
                            ax_col = axs[image_columns]
                    else:
                        ax = axs[i]
                        ax_col = axs[image_columns]

                subfig = ax.pcolormesh(
                    maps[scale],
                    cmap=colormaps["inferno"],
                    norm=col_norm,
                    # cmap=colormaps["coolwarm"],
                )

                set_ax_params(ax, parameters, multiple=True)
                set_title(
                    df,
                    type,
                    prop,
                    scale,
                    ax,
                    scale,
                    props_of_interest,
                    parameters,
                )
                if int(i // skip % image_columns) == (image_columns - 1):
                    if prop != "f_esc":
                        label = get_label(prop)
                    else:
                        label = r"$f_\mathrm{esc, cell}$"
                    create_color_bar(
                        fig,
                        ax_col,
                        parameters,
                        subfig,
                        label=label,
                        multiple=True,
                        ax_is_cbar=True,
                        horizontal=horizontal,
                        prop=prop,
                    )
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                counter += 1
        trim_axes(axs, counter + 1)
    return


def plot_z_histogram(df):
    parameters = plot_parameters(params=None)

    f, ax = plt.subplots(
        figsize=[parameters["figure_width"], parameters["figure_height"]]
    )

    values = np.array(df.groupby("z").count()["r"])

    ax.bar(np.arange(len(values)), values, width=0.8, color="darkgreen")
    labels = [f"{x:.1f}" for x in df.z.unique()[::-1]]
    ax.set_xticks(np.arange(len(labels))[::2] - 0.1)
    ax.set_xticklabels(labels[::2])
    ax.set_xlabel("z", size=parameters["labelsize"])
    ax.set_ylabel(r"$N_\mathrm{galaxies}$", size=parameters["labelsize"])
    set_ax_params(ax, parameters)
    return


def get_average_values(groups, log=False, em_weighted=False, df=None):
    if log:
        y_prop_means = np.log10(groups.mean())
        y_prop_std = groups.std()
        y_prop_count = groups.count()
        y_prop_err = 1 / groups.mean() * y_prop_std / np.sqrt(y_prop_count)

    elif em_weighted:
        Ion_em_groups = df.groupby(["prop_bins"])["Ion_em"]
        df["weights"] = Ion_em_groups.transform(lambda x: x / x.sum())

        df["f_esc_weighted"] = df["f_esc"] * df["weights"]
        df["weighted_means"] = df.groupby(["prop_bins"])[
            "f_esc_weighted"
        ].transform(lambda x: x.sum())

        y_prop_means = df.groupby(["prop_bins"])["f_esc_weighted"].sum()

        df["delta_fesc_weighted"] = (
            df["weights"] * (df["f_esc"] - df["weighted_means"]) ** 2
        )

        y_prop_var = (
            df.groupby(["prop_bins"])["weights"].apply(lambda x: x**2).sum()
            * df.groupby(["prop_bins"])["f_esc"].var()
        )

        y_prop_err = np.sqrt(y_prop_var)

    else:
        y_prop_means = groups.apply(
            lambda x: x.mean() if x.count() > 5 else np.nan
        )
        y_prop_std = groups.std()
        y_prop_count = groups.count()
        y_prop_err = y_prop_std / np.sqrt(y_prop_count)

    # for key in y_prop_means.keys():
    #     print(len(y_prop_means[key]))

    return y_prop_means, y_prop_err


def lineplots(
    df,
    mass_bins=30,
    em_weighted=False,
    x_prop="M_star_sun_log",
    y_prop="f_esc",
    skip=1,
    params=None,
    log=False,
    individual_z=True,
    x_log=False,
    all=False,
    with_mass_bins=False,
    two_modes=False,
    filters=None,
):
    parameters = plot_parameters(params)
    df.dropna(subset="f_esc", inplace=True)
    if x_prop == "TimeMajorMerger":
        df = df.replace({"TimeMajorMerger": [np.inf, -np.inf]}, np.nan).dropna(
            subset="TimeMajorMerger", axis=0
        )
    elif x_prop == "TimeMinorMerger":
        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset="TimeMinorMerger", axis=0
        )
    elif x_prop == "TimeRecentMerger":
        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset="TimeRecentMerger", axis=0
        )

    if x_log:
        df["prop_log"] = np.log10(df[x_prop])
        # Hacky solution to deepcopy the string
        x_prop_org = (x_prop + ".")[:-1]
        x_prop = "prop_log"
        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset="prop_log", axis=0
        )

    x_values = df[x_prop]
    x_edges = np.linspace(x_values.min(), x_values.max(), mass_bins)
    x_centers = (x_edges[1:] + x_edges[:-1]) / 2

    df["prop_bins"] = pd.cut(df[x_prop], x_edges, include_lowest=True)
    df = df[(df.M_star_sun_log > 6) & (df.M_star_sun_log < 9)]
    if individual_z:
        z_values = df["z"].unique()[::-1]
        groups = df.groupby(["z", "prop_bins"])[y_prop]

    elif with_mass_bins:
        stellar_mass_bins = pd.cut(df.M_star_sun_log, bins=20)
        df["mass_bins"] = stellar_mass_bins
        bin_groups = np.sort(stellar_mass_bins.unique())
        groups = df.groupby(["mass_bins", "prop_bins"])[y_prop]

    else:
        groups = df.groupby(["prop_bins"])[y_prop]

    if all:
        groups_all = df.groupby(["prop_bins"])[y_prop]

    y_prop_means, y_prop_err = get_average_values(
        groups, log=log, em_weighted=em_weighted, df=df
    )
    if two_modes:
        df1 = df[filters[0]]
        df2 = df[filters[1]]

        groups1 = df1.groupby(["prop_bins"])[y_prop]
        groups2 = df2.groupby(["prop_bins"])[y_prop]
        y_prop_means1, y_prop_err1 = get_average_values(
            groups1, log=log, em_weighted=em_weighted, df=df
        )
        y_prop_means2, y_prop_err2 = get_average_values(
            groups2, log=log, em_weighted=em_weighted, df=df
        )

    if all:
        y_prop_means_all, y_prop_err_all = get_average_values(
            groups_all, log=log, em_weighted=em_weighted, df=df
        )
        # print(groups_all.count())
        # print(y_prop_means_all)
        print(len(df))

    f, ax = plt.subplots(
        figsize=[parameters["figure_width"] * 1.3, parameters["figure_height"]]
    )

    if individual_z:
        norm = plt.Normalize(5, 11)
        for i, z in enumerate(z_values):
            if i % skip == 0:
                ax.errorbar(
                    x_centers,
                    y_prop_means[z],
                    yerr=y_prop_err[z],
                    label=f"z={z:.1f}",
                    color=plt.cm.viridis(norm(z)),
                    capsize=parameters["capsize"],
                    capthick=parameters["capwidth"],
                    linewidth=parameters["linewidth"],
                )
        # cbar = plt.colorbar(
        #     plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax
        # )
        # cbar.set_label("$z$", size=parameters["labelsize"])
        # cbar.ax.tick_params(labelsize=30)
    if with_mass_bins:
        norm = plt.Normalize(df.M_star_sun_log.min(), df.M_star_sun_log.max())
        for i, bin in enumerate(bin_groups):
            if i % skip == 0:
                lower = f"{bin.left:.1f}"
                upper = f"{bin.right:.1f}"
                label = (
                    "$M_\star=10^{"
                    + lower
                    + "}-10^{"
                    + upper
                    + "} \mathrm{{M}}_\odot$"
                )

                mass = (bin.right + bin.left) / 2
                ax.errorbar(
                    x_centers,
                    y_prop_means[bin],
                    yerr=y_prop_err[bin],
                    color=plt.cm.viridis(norm(mass)),
                    label=label,
                    capsize=parameters["capsize"],
                    capthick=parameters["capwidth"],
                    linewidth=parameters["linewidth"],
                )
        # cbar = plt.colorbar(
        #     plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax
        # )
        # cbar.set_label("$\log(M_\star/M_\odot)$", size=parameters["labelsize"])
        # cbar.ax.tick_params(labelsize=30)
    if all:
        factor = 1.5
        ax.errorbar(
            x_centers,
            y_prop_means_all,
            yerr=y_prop_err_all,
            capsize=parameters["capsize"] * factor,
            capthick=parameters["capwidth"] * factor,
            linewidth=parameters["linewidth"] * factor,
            color="black",
            label="all galaxies",
        )

    if two_modes:
        ax.errorbar(
            x_centers,
            y_prop_means1,
            yerr=y_prop_err1,
            capsize=parameters["capsize"],
            capthick=parameters["capwidth"],
            linewidth=parameters["linewidth"],
            color="red",
            label=r"$H<10^{21}\mathrm{cm}$",
        )
        ax.errorbar(
            x_centers,
            y_prop_means2,
            yerr=y_prop_err2,
            capsize=parameters["capsize"],
            capthick=parameters["capwidth"],
            linewidth=parameters["linewidth"],
            color="blue",
            label=r"$H>10^{21}\mathrm{cm}$",
        )

    # else:
    #     if all:
    #         ax.errorbar(
    #             x_centers,
    #             y_prop_means,
    #             yerr=y_prop_err,
    #             capsize=parameters["capsize"],
    #             capthick=parameters["capwidth"],
    #             linewidth=parameters["linewidth"],
    #             color="red",
    #         )

    if x_log:
        label_x = get_label(x_prop_org)
    else:
        label_x = get_label(x_prop)
    label_y = get_label(y_prop)

    ax.set_xlabel(label_x, size=parameters["labelsize"])
    ax.set_ylabel(label_y, size=parameters["labelsize"])
    set_ax_params(ax, parameters)
    ax.legend(fontsize=parameters["legendsize"])
    # ax.set_xlim(5.8)
    ax.set_ylim(0, 0.18)
    return


# Fixing the plot sample to specific galaxy IDs for quick updating
def plot_sample(df, mode, prop="f_esc", props_of_interest=[]):
    sample_dict = {
        "low_sfr": [630154, 533087, 546897],
        "high_sfr": [109259, 413819, 559133],
        "center": [514694, 32983, 417607],
        "extensive_escape": [347141, 345611, 355525],
        "localized_escape": [213376, 604648, 592453],
    }
    plot_prop_maps(
        df,
        halo_num=sample_dict[mode],
        prop=prop,
        hdf=None,
        hdf_prefix="gridded_maps_",
        grid_size="100",
        type="sample",
        skip=1,
        params=None,
        log=False,
        props_of_interest=props_of_interest,
        # n_maps=16,
        horizontal=True,
    )
    return


def modes_plot(
    full_df,
    color_prop="f_esc",
    statistic="mean",
    bins_x=30,
    bins_y=30,
    params=None,
    contour=True,
    add_line=False,
):
    prop_x = "TimeMajorMerger"
    mass_filter = (full_df.M_star_sun_log > 6.8) & (
        full_df.M_star_sun_log < 7.2
    )

    # mass_filter = (full_df.M_star_sun_log > 6.0) & (
    #     full_df.M_star_sun_log < 6.2
    # )

    full_df = full_df[mass_filter]
    filter_low = np.log10(full_df["Column_height"]) < 21
    filter_high = np.log10(full_df["Column_height"]) > 21

    y_dict = {
        "Column_height_low": {
            "filter": filter_low,
            "prop": "Column_height",
            "ax": "height_low",
            "line_log": True,
            "label_prop": r"$\log(\langle H \rangle)$",
            "yrange": (20.1, 22.2),
            "grid": (20, 7),
            "avg_lim": (20.69, 21.5),
        },
        "sSFR_low": {
            "filter": filter_low,
            "prop": "sSFR",
            "ax": "sSFR_low",
            "line_log": True,
            "label_prop": r"$\log(\langle \mathrm{sSFR}\rangle)$",
            "yrange": (-9.5, -7.6),
            "grid": (25, 25),
            "avg_lim": (-8.9, -8.05),
        },
        "MgasMstar_low": {
            "filter": filter_low,
            "prop": "MgasMstar",
            "ax": "Ratio_low",
            "line_log": True,
            "label_prop": r"$\log(\langle M_\mathrm{gas}/M_\star \rangle )$",
            "yrange": (-0.6, 2.4),
            "grid": (20, 15),
            "avg_lim": (0.45, 1.55),
        },
        "sZ_low": {
            "filter": filter_low,
            "prop": "sZ",
            "ax": "sZ_low",
            "line_log": True,
            "label_prop": r"$\log(\langle Z/M_\star \rangle)$",
            "yrange": (-10.6, -8.9),
            "grid": (20, 20),
            # "avg_lim": (-9.95, -9.15),
            "avg_lim": (-10, -9.3),
        },
        "Column_height_high": {
            "filter": filter_high,
            "prop": "Column_height",
            "ax": "height_high",
            "line_log": True,
            "label_prop": r"$\log(\langle H \rangle)$",
            "yrange": (20.1, 22.2),
            "grid": (20, 10),
            "avg_lim": (20.69, 21.5),
        },
        "sSFR_high": {
            "filter": filter_high,
            "prop": "sSFR",
            "ax": "sSFR_high",
            "line_log": True,
            "label_prop": r"$\log(\langle \mathrm{sSFR}\rangle)$",
            "yrange": (-9.5, -7.6),
            "grid": (25, 25),
            "avg_lim": (-8.9, -8.05),
        },
        "MgasMstar_high": {
            "filter": filter_high,
            "prop": "MgasMstar",
            "ax": "Ratio_high",
            "line_log": True,
            "label_prop": r"$\log(\langle M_\mathrm{gas}/M_\star \rangle )$",
            "yrange": (-0.6, 2.4),
            "grid": (20, 20),
            "avg_lim": (0.45, 1.55),
        },
        "sZ_high": {
            "filter": filter_high,
            "prop": "sZ",
            "ax": "sZ_high",
            "line_log": True,
            "label_prop": r"$\log(\langle Z/M_\star \rangle)$",
            "yrange": (-10.6, -8.9),
            "grid": (20, 20),
            # "avg_lim": (-9.95, -9.15),
            "avg_lim": (-10, -9.3),
        },
    }

    parameters = plot_parameters(params)
    parameters["colorbar_ticklabelsize"] = 20
    parameters["colorbar_labelsize"] = 30

    full_df = full_df.replace(
        {"TimeMajorMerger": [np.inf, -np.inf]}, np.nan
    ).dropna(subset="TimeMajorMerger", axis=0)

    f, axs = plt.subplot_mosaic(
        [
            ["colorbar", "colorbar"],
            ["ghost", "ghost"],
            ["height_low", "height_high"],
            # ["sSFR_low", "sSFR_high"],
            # ["Ratio_low", "Ratio_high"],
            # ["sZ_low", "sZ_high"],
        ],
        # sharey=False,
        sharex=False,
        gridspec_kw={
            "hspace": 0,
            "width_ratios": [24, 24],
            "height_ratios": [1.5, 1.0, 24],  # , 24, 24, 24],
            "wspace": 0,
        },
        figsize=[15, 8.5],
    )
    # axs["height_low"].get_shared_x_axes().join(
    #     axs["height_low"], axs["height_high"]
    # )

    axs["ghost"].axis("off")

    for y_prop in y_dict:
        if (y_prop == "Column_height_low") or (y_prop == "Column_height_high"):
            prop_y = y_dict[y_prop]["prop"]
            df = full_df.loc[y_dict[y_prop]["filter"]]
            y_values = df[y_dict[y_prop]["prop"]]
            y_values = np.ma.masked_invalid(np.log10(y_values))
            x_values = df.loc[:, prop_x]
            bins_x, bins_y = y_dict[y_prop]["grid"]
            x_edges = np.linspace(x_values.min(), x_values.max(), bins_x)
            y_edges = np.linspace(y_values.min(), y_values.max(), bins_y)

            hist, hist_cont, xedges_cont, yedges_cont = get_histogram(
                df,
                x_values,
                y_values,
                bins=[x_edges, y_edges],
                color_prop=color_prop,
                statistic=statistic,
            )
            cont_centers_x = (xedges_cont[1:] + xedges_cont[:-1]) / 2
            cont_centers_y = (yedges_cont[1:] + yedges_cont[:-1]) / 2
            x_grid, y_grid = np.meshgrid(x_edges, y_edges)
            vmin, vcenter, vmax = get_color_limits(color_prop, statistic)
            col_norm = colors.TwoSlopeNorm(
                vmin=vmin, vcenter=vcenter, vmax=vmax
            )

            ax = axs[y_dict[y_prop]["ax"]]
            if "high" in y_prop:
                ax.yaxis.set_tick_params(labelleft=False)
                ax.set_yticks([])

            if "Column_height" not in y_prop:
                ax.xaxis.set_tick_params(labelbottom=False, bottom=False)

            subfig = ax.pcolormesh(
                x_grid,
                y_grid,
                hist.T,
                norm=col_norm,
                cmap=plt.get_cmap("inferno"),
            )

            levels = get_levels(hist_cont, thresholds=[0.954, 0.683])
            if contour:
                ax.contour(
                    cont_centers_x,
                    cont_centers_y,
                    hist_cont.T,
                    levels=levels,
                    linewidths=4,
                    linestyles=["dotted", "dashed"],  # , "solid"],
                    colors="lightblue",
                )

            if add_line:
                x_edges = np.linspace(x_values.min(), x_values.max(), 20)
                x_centers = (x_edges[1:] + x_edges[:-1]) / 2
                df["prop_bins"] = pd.cut(
                    df.loc[:, prop_x], x_edges, include_lowest=True
                )
                groups = df.groupby(["prop_bins"])[prop_y]

                y_prop_std = groups.std()
                y_prop_count = groups.count()
                if y_dict[y_prop]["line_log"]:
                    y_prop_means = groups.apply(
                        lambda x: np.log10(x.mean())
                        if x.count() > 5
                        else np.nan
                    )
                    y_prop_err = (
                        1 / groups.mean() * y_prop_std / np.sqrt(y_prop_count)
                    )

                else:
                    y_prop_means = groups.apply(
                        lambda x: x.mean() if x.count() > 5 else np.nan
                    )
                    y_prop_err = y_prop_std / np.sqrt(y_prop_count)

                ax2 = ax.twinx()
                prop = y_dict[y_prop]["label_prop"]
                color = "lime"
                ax2.errorbar(
                    x_centers,
                    y_prop_means,
                    yerr=y_prop_err,
                    capthick=parameters["capwidth"],
                    capsize=5,
                    linewidth=parameters["linewidth"],
                    color=color,
                )
                ax2.set_ylabel(
                    prop,
                    size=30,
                )
                ax2_color = "green"
                ax2.set_ylim(y_dict[y_prop]["avg_lim"])
                ax2.yaxis.label.set_color(ax2_color)
                ax2.tick_params(axis="y", colors=ax2_color)
                if "low" in y_prop:
                    ax2.yaxis.set_tick_params(labelright=False)
                    ax2.set_yticks([])
                    ax2.set_ylabel("")

                set_ax_params(ax2, parameters)
        else:
            continue

        labelsize = 30
        ax.set_xlabel(get_label(prop_x), size=labelsize)
        if "low" in y_prop:
            ax.set_ylabel(get_label(y_dict[y_prop]["prop"]), size=labelsize)
        set_ax_params(ax, parameters)
        # ax.set_yticks([19, 20, 21, 22])
        ax.set_ylim(y_dict[y_prop]["yrange"])
        ax.set_xlim(0, 770)

        ax.set_xticks([0, 200, 400, 600])

    color_label = get_label(color_prop)
    create_color_bar(
        f,
        axs["colorbar"],
        parameters,
        subfig,
        label=color_label,
        gap=True,
        ax_is_cbar=True,
        horizontal=False,
    )
    return


def get_convergence_maps(hdf, halo_idx, prop):
    maps = {}
    for key in hdf.keys():
        if key == "3":
            continue
        if key == "None":
            continue
        maps[float(key)] = hdf[key][str(halo_idx)][prop]
    return maps


def histograms_plot(
    full_df,
    color_prop="f_esc",
    statistic="mean",
    bins_x=30,
    bins_y=30,
    params=None,
    contour=True,
    add_line=False,
):
    prop_x = "TimeMajorMerger"
    mass_filter = (full_df.M_star_sun_log > 6.8) & (
        full_df.M_star_sun_log < 7.2
    )

    full_df = full_df[mass_filter]
    filter_high = np.log10(full_df["Column_height"]) > 21
    filter_low = np.log10(full_df["Column_height"]) < 21

    y_dict = {
        # "Column_height_low": {
        #     "filter": filter_low,
        #     "prop": "Column_height",
        #     "ax": "height_low",
        #     "line_log": True,
        #     "label_prop": r"$\log(\langle H \rangle)$",
        #     "yrange": (20.1, 21.0),
        #     "grid": (20, 7),
        #     "avg_lim": (20.8, 21.0),
        # },
        # "Column_height_high": {
        #     "filter": filter_high,
        #     "prop": "Column_height",
        #     "ax": "height_high",
        #     "line_log": True,
        #     "label_prop": r"$\log(\langle H \rangle)$",
        #     "yrange": (21, 22.2),
        #     "grid": (20, 10),
        #     "avg_lim": (21.0, 21.5),
        # },
        "sSFR": {
            "filter": filter_high,
            "prop": "sSFR",
            "ax": "sSFR",
            "line_log": True,
            "label_prop": r"$\log(\langle \mathrm{sSFR}\rangle)$",
            "yrange": (-9.4, -7.6),
            "grid": (25, 25),
            "avg_lim": (-8.6, -8.05),
        },
        "MgasMstar": {
            "filter": filter_high,
            "prop": "MgasMstar",
            "ax": "Ratio",
            "line_log": True,
            "label_prop": r"$\log(\langle M_\mathrm{gas}/M_\star \rangle )$",
            "yrange": (0.3, 2.4),
            "grid": (20, 15),
            "avg_lim": (0.91, 1.55),
        },
        "sZ": {
            "filter": filter_high,
            "prop": "sZ",
            "ax": "sZ",
            "line_log": True,
            "label_prop": r"$\log(\langle Z/M_\star \rangle)$",
            "yrange": (-10.6, -9.1),
            "grid": (20, 20),
            # "avg_lim": (-9.95, -9.15),
            "avg_lim": (-9.9, -9.54),
        },
        "v_sigma": {
            "filter": filter_high,
            "prop": "v_sigma",
            "ax": "v_sigma",
            "line_log": False,
            "label_prop": r"$\langle v_\mathrm{max}/\sigma_v \rangle$",
            "yrange": (1.1, 6.1),
            "grid": (20, 20),
            "avg_lim": (2.91, 3.22),
        },
        "flow": {
            "filter": filter_high,
            "prop": "flow",
            "ax": "flow",
            "line_log": False,
            "label_prop": r"$\langle \mathcal{F} \rangle$",
            "yrange": (-0.92, 0.63),
            "grid": (20, 15),
            "avg_lim": (-0.21, -0.062),
        },
        "Offset_pc": {
            "filter": filter_high,
            "prop": "Offset_pc",
            "ax": "Offset_pc",
            "line_log": True,
            "label_prop": r"$\langle \log(\Delta_C) \rangle$",
            "yrange": (0.5, 3.45),
            "grid": (20, 20),
            # "avg_lim": (-9.95, -9.15),
            "avg_lim": (1.7, 2.45),
        },
    }
    parameters = plot_parameters(params)
    parameters["colorbar_ticklabelsize"] = 20
    parameters["colorbar_labelsize"] = 30

    full_df = full_df.replace(
        {"TimeMajorMerger": [np.inf, -np.inf]}, np.nan
    ).dropna(subset="TimeMajorMerger", axis=0)

    f, axs = plt.subplot_mosaic(
        [
            ["colorbar", "colorbar", "colorbar"],
            [".", ".", "."],
            # ["sSFR"],
            # ["Ratio"],
            # ["sZ"],
            # ["height_high"],
            # ["height_low"],
            # ["v_sigma"],
            # ["flow"],
            # ["Offset_pc"],
            ["flow", ".", "sSFR"],
            ["Ratio", ".", "v_sigma"],
            ["sZ", ".", "Offset_pc"],
        ],
        sharey=False,
        sharex=False,
        gridspec_kw={
            "hspace": 0,
            "width_ratios": [24, 17, 24],
            "height_ratios": [1.5, 1.0, 24, 24, 24],
            "wspace": 0,
        },
        figsize=[18.0, 20.5],
    )
    # axs["height_low"].get_shared_x_axes().join(
    #     axs["height_low"], axs["height_high"]
    # )

    # axs["ghost"].axis("off")

    for y_prop in y_dict:
        prop_y = y_dict[y_prop]["prop"]
        df = full_df.loc[y_dict[y_prop]["filter"]]
        y_values = df[y_dict[y_prop]["prop"]]
        if (prop_y == "flow") or (prop_y == "v_sigma"):
            y_values = np.ma.masked_invalid(y_values)
        else:
            y_values = np.ma.masked_invalid(np.log10(y_values))
        x_values = df.loc[:, prop_x]
        bins_x, bins_y = y_dict[y_prop]["grid"]
        x_edges = np.linspace(x_values.min(), x_values.max(), bins_x)
        y_edges = np.linspace(y_values.min(), y_values.max(), bins_y)

        hist, hist_cont, xedges_cont, yedges_cont = get_histogram(
            df,
            x_values,
            y_values,
            bins=[x_edges, y_edges],
            color_prop=color_prop,
            statistic=statistic,
        )
        cont_centers_x = (xedges_cont[1:] + xedges_cont[:-1]) / 2
        cont_centers_y = (yedges_cont[1:] + yedges_cont[:-1]) / 2
        x_grid, y_grid = np.meshgrid(x_edges, y_edges)
        vmin, vcenter, vmax = get_color_limits(color_prop, statistic)
        col_norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        ax = axs[y_dict[y_prop]["ax"]]

        if ("Offset_pc" not in y_prop) and ("sZ" not in y_prop):
            ax.xaxis.set_tick_params(labelbottom=False, bottom=False)

        subfig = ax.pcolormesh(
            x_grid,
            y_grid,
            hist.T,
            norm=col_norm,
            cmap=plt.get_cmap("inferno"),
        )

        levels = get_levels(hist_cont, thresholds=[0.954, 0.683])
        if contour:
            ax.contour(
                cont_centers_x,
                cont_centers_y,
                hist_cont.T,
                levels=levels,
                linewidths=4,
                linestyles=["dotted", "dashed"],  # , "solid"],
                colors="lightblue",
            )

        if add_line:
            x_edges = np.linspace(x_values.min(), x_values.max(), 20)
            x_centers = (x_edges[1:] + x_edges[:-1]) / 2
            df["prop_bins"] = pd.cut(
                df.loc[:, prop_x], x_edges, include_lowest=True
            )
            groups = df.groupby(["prop_bins"])[prop_y]

            y_prop_std = groups.std()
            y_prop_count = groups.count()
            if y_dict[y_prop]["line_log"]:
                y_prop_means = groups.apply(
                    lambda x: np.log10(x.mean()) if x.count() > 5 else np.nan
                )
                y_prop_err = (
                    1 / groups.mean() * y_prop_std / np.sqrt(y_prop_count)
                )

            else:
                y_prop_means = groups.apply(
                    lambda x: x.mean() if x.count() > 5 else np.nan
                )
                y_prop_err = y_prop_std / np.sqrt(y_prop_count)

            ax2 = ax.twinx()
            prop = y_dict[y_prop]["label_prop"]
            color = "lime"
            ax2.errorbar(
                x_centers,
                y_prop_means,
                yerr=y_prop_err,
                capthick=parameters["capwidth"],
                capsize=5,
                linewidth=parameters["linewidth"],
                color=color,
            )
            if "height_low" not in y_prop:
                ax2.set_ylabel(
                    prop,
                    size=30,
                )
            ax2_color = "green"
            ax2.set_ylim(y_dict[y_prop]["avg_lim"])
            ax2.yaxis.label.set_color(ax2_color)
            ax2.tick_params(axis="y", colors=ax2_color)

            # ax2.yaxis.set_label_coords(1.2, -0.0)

            set_ax_params(ax2, parameters)

        labelsize = 30
        ax.set_xlabel(get_label(prop_x), size=labelsize)
        if "height_low" not in y_prop:
            ax.set_ylabel(get_label(y_dict[y_prop]["prop"]), size=labelsize)
        set_ax_params(ax, parameters)
        # ax.set_yticks([19, 20, 21, 22])
        ax.set_ylim(y_dict[y_prop]["yrange"])
        ax.set_xlim(0, 750)
        # ax.yaxis.set_label_coords(-0.18, 0.0)

        if y_prop == "Column_height_high":
            ax.spines["bottom"].set_color("white")
            ax2.spines["bottom"].set_color("white")

        if y_prop == "Column_height_low":
            ax.spines["top"].set_color("white")
            ax2.spines["top"].set_color("white")

        ax.set_xticks([0, 200, 400, 600])

    color_label = get_label(color_prop)
    create_color_bar(
        f,
        axs["colorbar"],
        parameters,
        subfig,
        label=color_label,
        gap=True,
        ax_is_cbar=True,
        horizontal=False,
    )
    return


def scatter_plot(
    df,
    prop_x,
    prop_y,
    log_x=True,
    log_y=True,
    params=None,
):
    parameters = plot_parameters(params)
    df = df.dropna(subset=[prop_x, prop_y])
    if log_x:
        df = df[df[prop_x] > 0]
    if log_y:
        df = df[df[prop_y] > 0]

    if log_x:
        x_values = np.log10(df[prop_x])
    else:
        x_values = df[prop_x]

    if log_y:
        y_values = np.log10(df[prop_y])

    else:
        y_values = df[prop_y]
    # if log_x:
    #     x_values = np.ma.masked_invalid(np.log10(x_values))
    # if log_y:
    #     y_values = np.ma.masked_invalid(np.log10(y_values))

    f, ax = plt.subplots(
        figsize=[parameters["figure_width"], parameters["figure_height"]]
    )
    ax.scatter(x_values, y_values, s=5, alpha=0.7, color="grey")

    a, b = np.polyfit(x_values, y_values, 1)

    ax.plot(
        x_values,
        a * x_values + b,
        linewidth=parameters["linewidth"],
        color="black",
    )

    ax.set_xlabel(get_label(prop_x), size=parameters["labelsize"])
    ax.set_ylabel(get_label(prop_y), size=parameters["labelsize"])
    # ax.legend(fontsize=parameters["legendsize"])
    set_ax_params(ax, parameters)

    # ax.set_xlim(-9, -7.5)
    # ax.set_ylim(-1, 3.6)
    return
