import numpy as np
import pandas as pd
import h5py
import os
from utils import get_snap
import astropy.units as u

np.seterr(divide="ignore", invalid="ignore")


def get_average_N_d(maps):
    return 1 / np.sum(1 / np.array(maps["N_d"]))


def get_df_quantity(prop, hdf_file, df, index, scale):
    maps = hdf_file[scale][str(index)]
    flux_quantities = ["Ion_flux", "Bol_flux"]
    summed_quantities = [
        "M_gas",
        "M_star",
        "SFR",
    ]
    average_quantities = [
        "Dust_norm",
        "Column_dens",
        "Column_dens_stroemgren",
        "N_d",
        "N_ratio",
        "N_red",
        "U",
        "U1",
        "f_g",
        "f_g_crit",
        "p_r",
        "sigma_d_H",
        "n_gas",
    ]
    if prop in summed_quantities:
        quant = np.sum(maps[prop])
    elif prop in flux_quantities:
        grid_column = "Grid_cell_size"
        # This is only for convergence testing
        # grid_column = f"Grid_cell_size_{scale}"
        grid_size = df.loc[index, grid_column]
        quant = np.sum(maps[prop]) * grid_size**2
    elif prop in average_quantities:
        quant = np.mean(np.ma.masked_invalid(maps[prop]))
    elif prop == "Metallicity":
        quant = np.sum(
            np.array(maps["M_gas"])
            * np.ma.masked_invalid(np.array(maps["Metallicity"]))
        ) / np.sum(maps["M_gas"])
    elif prop == "f_esc":
        quant = np.sum(
            np.array(maps["f_esc"]) * np.array(maps["Ion_flux"])
        ) / np.sum(maps["Ion_flux"])
    else:
        return
    # The scale here is only for testing
    # if prop in flux_quantities:
    #     column_name = prop[:-4] + "em_" + str(scale)
    # else:
    #     column_name = prop + "_" + str(scale)
    if prop in flux_quantities:
        column_name = prop[:-4] + "em"
    else:
        column_name = prop
    df.loc[index, column_name] = quant
    return


def add_map_quantities(df, hdf_file, scale):
    for prop in hdf_file[scale][str(df.index[0])].keys():
        # for testing only
        # column_name = prop + "_" + scale
        column_name = scale
        df[column_name] = np.nan
        for idx in df.index:
            get_df_quantity(prop, hdf_file, df, idx, scale)
    return


def update_map_df(
    snap_num,
    scale,
    hdf_name,
    df_name,
    output_name=None,
    base="/ptmp/mpa/ivkos/semianalytic_fesc",
):
    if output_name is None:
        output_name = df_name
    snap = get_snap(snap_num)
    hdf_filename = hdf_name + ".hdf5"
    hdf_path = os.path.join(base, snap, hdf_filename)

    df_filename = df_name + ".pickle"
    output_filename = df_name + ".pickle"
    origin_path = os.path.join(base, snap, df_filename)
    destination_path = os.path.join(base, snap, output_filename)

    hdf_file = h5py.File(hdf_path, "a")
    df = pd.read_pickle(origin_path)
    add_map_quantities(df, hdf_file, str(scale))

    df.to_pickle(destination_path)
    hdf_file.close()
    return
