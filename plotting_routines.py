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
    parameters["x_labelsize"] = 50
    parameters["y_labelsize"] = 50

    parameters["titlesize"] = 30
    parameters["length_major_ticks"] = 16
    parameters["length_minor_ticks"] = 8
    parameters["width_minor_ticks"] = 3
    parameters["width_major_ticks"] = 4
    parameters["labelsize_ticks"] = 35

    parameters["colorbar_labelsize"] = 50
    parameters["colorbar_ticklabelsize"] = 35

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


def set_ax_params(ax, parameters):
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
    ax.tick_params(
        axis="both", which="both", labelsize=parameters["labelsize_ticks"]
    )

    # change all spines
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(parameters["axes_width"])
    return


def create_color_bar(f, ax, parameters, subfig, label=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = f.colorbar(subfig, cax=cax)
    if label is not None:
        cbar.set_label(label, size=parameters["colorbar_labelsize"])
    cbar.ax.tick_params(labelsize=parameters["colorbar_ticklabelsize"])
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
        set_ax_params(ax, parameters)
        ax.set_title(prop, fontsize=parameters["titlesize"])
        create_color_bar(fig, ax, parameters, subfig)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # if len(maps.keys()) % 2 == 1:
    #     fig.delaxes(axs[image_rows - 1, 1])
    return


def get_convergence_maps(hdf, halo_idx, prop):
    maps = {}
    for key in hdf.keys():
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


def plot_prop_maps(
    df, hdf, halo_idx, skip=1, params=None, prop="f_esc", log=False, grid=False
):
    maps = get_convergence_maps(hdf, halo_idx, prop=prop)
    maps = dict(sorted(maps.items()))
    if prop == "f_esc":
        vmin, vmax = 0, 1
    else:
        vmin, vmax = None, None
    if log:
        # vmin, vmax = np.log10(vmin), np.log10(vmax)
        for key in maps.keys():
            maps[key] = np.log10(maps[key])

    parameters = plot_parameters(params, multiple=True)

    image_columns = 4
    image_rows = int(np.ceil(len(maps.keys()) / image_columns / skip))
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

    counter = 0
    for i, scale in enumerate(maps.keys()):
        if i % skip == 0:
            column = int(i // skip % image_columns)
            row = int(i // skip // image_columns)
            ax = axs[row, column]

            subfig = ax.pcolormesh(
                maps[scale], cmap=colormaps["inferno"], vmin=vmin, vmax=vmax
            )

            set_ax_params(ax, parameters)
            if prop == "f_esc":
                f_esc = df.loc[halo_idx, f"f_esc_{int(scale)}"]
                ax.set_title(
                    rf"$\lambda = ${scale:.2f}, $f_\mathrm{{esc}} = ${f_esc:.2f}",
                    fontsize=parameters["titlesize"],
                )
            elif prop == "Ion_flux":
                ax.set_title(
                    rf"$\lambda = ${scale:.2f}: $F_i$",
                    fontsize=parameters["titlesize"],
                )
            create_color_bar(fig, ax, parameters, subfig)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            counter += 1
    trim_axes(axs, counter)
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
        parameters["x_label_convergence"], size=parameters["y_labelsize"]
    )
    ax.set_ylabel(
        parameters["y_label_convergence"], size=parameters["y_labelsize"]
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


def fesc_histogram(df, em_weighted=False, mass_bins=30, params=None):
    parameters = plot_parameters(params)
    g_to_msun = (1 * u.g).to(u.M_sun)
    df.dropna(subset="f_esc", inplace=True)
    df["M_star_sun_log"] = np.log10(df["M_star"] * g_to_msun)
    x_values = df["z"]
    y_values = df["M_star_sun_log"]

    x_edges = get_z_edges(df)[::-1]
    y_edges = np.linspace(y_values.min(), y_values.max(), mass_bins)

    if em_weighted:
        df["f_esc_weighted"] = df["f_esc"] * df["Ion_em"]
        hist_esc, *_ = stats.binned_statistic_2d(
            x_values,
            y_values,
            values=df["f_esc_weighted"],
            statistic="sum",
            bins=[x_edges, y_edges],
        )
        hist_em, *_ = stats.binned_statistic_2d(
            x_values,
            y_values,
            values=df["Ion_em"],
            statistic="sum",
            bins=[x_edges, y_edges],
        )
        hist = hist_esc / hist_em
    else:
        hist, *_ = stats.binned_statistic_2d(
            x_values,
            y_values,
            values=df["f_esc"],
            statistic="mean",
            bins=[x_edges, y_edges],
        )
    hist_cont, xedges_cont, yedges_cont, _ = stats.binned_statistic_2d(
        x_values,
        y_values,
        values=df["f_esc"],
        statistic="count",
        bins=[x_edges, y_edges],
    )
    cont_centers_x = (xedges_cont[1:] + xedges_cont[:-1]) / 2
    cont_centers_y = (yedges_cont[1:] + yedges_cont[:-1]) / 2
    x_grid, y_grid = np.meshgrid(x_edges, y_edges)
    col_norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.1, vmax=0.2)
    f, ax = plt.subplots(
        figsize=[parameters["figure_width"], parameters["figure_height"]]
    )
    subfig = ax.pcolormesh(
        x_grid, y_grid, hist.T, norm=col_norm, cmap=plt.get_cmap("inferno")
    )
    levels = get_levels(hist_cont, thresholds=[0.954, 0.683])
    ax.contour(
        cont_centers_x,
        cont_centers_y,
        hist_cont.T,
        levels=levels,
        linewidths=5,
        linestyles=["dashed", "solid"],
        colors="black",
    )
    create_color_bar(f, ax, parameters, subfig, label=r"$f_\mathrm{esc}$")
    ax.set_xlabel("z", size=parameters["y_labelsize"])
    ax.set_ylabel(
        r"$\log(M_\star)[\log(M_\odot)]$", size=parameters["y_labelsize"]
    )
    set_ax_params(ax, parameters)

    return


def log_count(prop):
    return np.log10(len(prop))


def masked_mean(prop):
    return np.mean(np.ma.masked_invalid(prop))


def count_histogram(df, mass_bins=30, fesc_bins=30, params=None):
    parameters = plot_parameters(params)
    g_to_msun = (1 * u.g).to(u.M_sun)
    df.dropna(subset="f_esc", inplace=True)
    df["M_star_sun_log"] = np.log10(df["M_star"] * g_to_msun)
    x_values = df["M_star_sun_log"]
    y_values = df["f_esc"]

    x_edges = np.linspace(x_values.min(), x_values.max(), fesc_bins)
    y_edges = np.linspace(y_values.min(), y_values.max(), mass_bins)

    hist, *_ = stats.binned_statistic_2d(
        x_values,
        y_values,
        values=df["f_esc"],
        statistic=log_count,
        bins=[x_edges, y_edges],
    )
    hist_cont, xedges_cont, yedges_cont, _ = stats.binned_statistic_2d(
        x_values,
        y_values,
        values=df["f_esc"],
        statistic="count",
        bins=[x_edges, y_edges],
    )
    cont_centers_x = (xedges_cont[1:] + xedges_cont[:-1]) / 2
    cont_centers_y = (yedges_cont[1:] + yedges_cont[:-1]) / 2

    x_grid, y_grid = np.meshgrid(x_edges, y_edges)
    col_norm = colors.TwoSlopeNorm(vmin=0, vcenter=2.5, vmax=5)
    f, ax = plt.subplots(
        figsize=[parameters["figure_width"], parameters["figure_height"]]
    )
    subfig = ax.pcolormesh(
        x_grid, y_grid, hist.T, norm=col_norm, cmap=plt.get_cmap("inferno")
    )
    levels = get_levels(hist_cont, thresholds=[0.954, 0.683])
    ax.contour(
        cont_centers_x,
        cont_centers_y,
        hist_cont.T,
        levels=levels,
        linewidths=5,
        linestyles=["dashed", "solid"],
        colors="black",
    )
    create_color_bar(f, ax, parameters, subfig, label=r"log(counts)")
    ax.set_ylabel(r"$f_\mathrm{esc}$", size=parameters["y_labelsize"])
    ax.set_xlabel(
        r"$\log(M_\star)[\log(M_\odot)]$", size=parameters["y_labelsize"]
    )
    set_ax_params(ax, parameters)

    return


def prop_prop_histogram(
    df,
    prop_x,
    prop_y,
    em_weighted=False,
    log_x=True,
    log_y=True,
    bins_x=30,
    bins_y=30,
    label_x=None,
    label_y=None,
    params=None,
):
    parameters = plot_parameters(params)
    df.dropna(subset="f_esc", inplace=True)
    df.dropna(subset="f_g_crit", inplace=True)
    x_values = df[prop_x]
    y_values = df[prop_y]
    if log_x:
        x_values = np.ma.masked_invalid(np.log10(x_values))
    if log_y:
        y_values = np.ma.masked_invalid(np.log10(y_values))

    x_edges = np.linspace(x_values.min(), x_values.max(), bins_x)
    y_edges = np.linspace(y_values.min(), y_values.max(), bins_y)

    if em_weighted:
        df["f_esc_weighted"] = df["f_esc"] * df["Ion_em"]
        hist_esc, *_ = stats.binned_statistic_2d(
            x_values,
            y_values,
            values=df["f_esc_weighted"],
            statistic="sum",
            bins=[x_edges, y_edges],
        )
        hist_em, *_ = stats.binned_statistic_2d(
            x_values,
            y_values,
            values=df["Ion_em"],
            statistic="sum",
            bins=[x_edges, y_edges],
        )
        hist = hist_esc / hist_em
    else:
        hist, *_ = stats.binned_statistic_2d(
            x_values,
            y_values,
            values=df["f_esc"],
            statistic="mean",
            bins=[x_edges, y_edges],
        )
        hist, *_ = stats.binned_statistic_2d(
            x_values,
            y_values,
            values=df["M_star_sun_log"],
            statistic="mean",
            bins=[x_edges, y_edges],
        )
    hist_cont, xedges_cont, yedges_cont, _ = stats.binned_statistic_2d(
        x_values,
        y_values,
        values=df["f_esc"],
        statistic="count",
        bins=[x_edges, y_edges],
    )
    cont_centers_x = (xedges_cont[1:] + xedges_cont[:-1]) / 2
    cont_centers_y = (yedges_cont[1:] + yedges_cont[:-1]) / 2
    x_grid, y_grid = np.meshgrid(x_edges, y_edges)
    vmin = 5.8
    vcenter = 6.3
    vmax = 7.0
    col_norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    f, ax = plt.subplots(
        figsize=[parameters["figure_width"], parameters["figure_height"]]
    )
    subfig = ax.pcolormesh(
        x_grid, y_grid, hist.T, norm=col_norm, cmap=plt.get_cmap("inferno")
    )
    levels = get_levels(hist_cont, thresholds=[0.997, 0.954, 0.683])
    ax.contour(
        cont_centers_x,
        cont_centers_y,
        hist_cont.T,
        levels=levels,
        linewidths=3,
        linestyles=["dotted", "dashed", "solid"],
        colors="blue",
    )
    # bar_label = r"$\log(M_\star/M_\odot)$"
    bar_label = r"$f_\mathrm{esc}$"
    # bar_label = r"$f_{g,\mathrm{crit}}$"
    # bar_label = "log(counts)"
    create_color_bar(f, ax, parameters, subfig, label=bar_label)

    ax.set_xlabel(label_x, size=parameters["y_labelsize"])
    ax.set_ylabel(label_y, size=parameters["y_labelsize"])
    set_ax_params(ax, parameters)
    return


def fesc_Mstar(df, mass_bins=30, em_weighted=False, skip=1, params=None):
    parameters = plot_parameters(params)
    g_to_msun = (1 * u.g).to(u.M_sun)
    df.dropna(subset="f_esc", inplace=True)
    df["M_star_sun_log"] = np.log10(df["M_star"] * g_to_msun)
    x_values = df["M_star_sun_log"]
    x_edges = np.linspace(x_values.min(), x_values.max(), mass_bins)
    x_centers = (x_edges[1:] + x_edges[:-1]) / 2
    df["mass_bins"] = pd.cut(df["M_star_sun_log"], x_edges)

    z_values = df["z"].unique()[::-1]
    if em_weighted:
        df["f_esc_weighted"] = df["f_esc"] * df["Ion_em"]
        f_esc_weighted = df.groupby(["z", "mass_bins"])["f_esc"].sum()
        Ion_em = df.groupby(["z", "mass_bins"])["Ion_em"].sum()
        f_esc_means = f_esc_weighted / Ion_em
    else:
        groups = df.groupby(["z", "mass_bins"])["f_esc"]
        f_esc_means = groups.mean()
        f_esc_std = groups.std()
        f_esc_count = groups.count()
        f_esc_err = f_esc_std / np.sqrt(f_esc_count)

    f, ax = plt.subplots(
        figsize=[parameters["figure_width"], parameters["figure_height"]]
    )
    for i, z in enumerate(z_values):
        if i % skip == 0:
            ax.errorbar(
                x_centers,
                f_esc_means[z],
                yerr=f_esc_err[z],
                label=f"z={z:.1f}",
                capsize=parameters["capsize"],
                linewidth=parameters["linewidth"],
            )

    label_x = r"$\log(M_\star/M_\odot)$"
    label_y = r"$\langle f_\mathrm{esc} \rangle$"

    ax.set_xlabel(label_x, size=parameters["y_labelsize"])
    ax.set_ylabel(label_y, size=parameters["y_labelsize"])
    set_ax_params(ax, parameters)
    ax.legend(fontsize=parameters["legendsize"])
    ax.set_xlim(5.8)
    ax.set_ylim(0, 0.2)
    return


def get_hdf(df, z, hdf_prefix):
    all_z = df["z"].unique()
    snap_num = int(np.where(all_z == z)[0])
    base_path = "/ptmp/mpa/ivkos/semianalytic_fesc"
    snap = get_snap(snap_num)
    hdf_name = f"{hdf_prefix}{snap_num}.hdf5"
    origin_path_hdf = os.path.join(base_path, snap, hdf_name)
    f = h5py.File(origin_path_hdf)
    return f


def plot_prop_map(
    df,
    halo_num,
    prop="f_esc",
    log=False,
    grid_size="100",
    hdf_prefix="gridded_maps_",
    params=None,
):
    halo_idx = df.loc[halo_num, "idx"]
    halo_z = df.loc[halo_num, "z"]
    hdf = get_hdf(df, halo_z, hdf_prefix)
    image = hdf[grid_size][str(halo_idx)][prop]
    print(hdf[grid_size][str(halo_idx)].keys())
    if prop == "f_esc":
        vmin, vmax = 0, 1
    else:
        vmin, vmax = None, None
    if log:
        # vmin, vmax = np.log10(vmin), np.log10(vmax)
        image = np.log10(image)

    parameters = plot_parameters(params)
    figsize = (
        parameters["width_per_image"],
        parameters["height_per_image"],
    )
    fig, ax = plt.subplots(figsize=figsize)

    subfig = ax.pcolormesh(
        image, cmap=colormaps["inferno"], vmin=vmin, vmax=vmax
    )

    set_ax_params(ax, parameters)
    if prop == "f_esc":
        f_esc = df.loc[halo_num, "f_esc"]
        ax.set_title(
            rf"$f_\mathrm{{esc}} = ${f_esc:.2f}",
            fontsize=parameters["titlesize"],
        )
    elif prop == "Ion_flux":
        ax.set_title(r"$F_i$", fontsize=parameters["titlesize"])
    else:
        if log:
            ax.set_title(f"log({prop})", fontsize=parameters["titlesize"])
        else:
            ax.set_title(prop, fontsize=parameters["titlesize"])
    create_color_bar(fig, ax, parameters, subfig)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return
