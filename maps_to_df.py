import numpy as np
import pandas as pd
import h5py
import os
from utils import get_snap
import astropy.units as u


def get_average_N_d(maps):
    return 1 / np.sum(1 / np.array(maps["N_d"]))


def get_df_quantity(prop, hdf_file, df, index, scale):
    maps = hdf_file[scale][str(index)]
    flux_quantities = ["Ion_flux", "Bol_flux"]
    summed_quantities = [
        "Dust_norm",
        "M_gas",
        "M_star",
        "SFR",
    ]
    average_quantities = [
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
        "sigma_d_H" "n_gas",
    ]
    if prop in summed_quantities:
        quant = np.sum(maps[prop])
    elif prop in flux_quantities:
        cm_to_kpc = (1 * u.cm).to(u.kpc).value
        grid_column = "Grid_cell_size_" + str(scale)
        grid_size = df.loc[index, grid_column]
        quant = np.sum(maps[prop]) * grid_size**2 * cm_to_kpc**2
    elif prop in average_quantities:
        quant = np.mean(maps[prop])
    elif prop == "Metallicity":
        quant = np.sum(
            np.array(maps["M_gas"]) * np.array(maps["Metallicity"])
        ) / np.sum(maps["M_gas"])
    elif prop == "f_esc":
        quant = np.sum(
            np.array(maps["f_esc"]) * np.array(maps["Ion_flux"])
        ) / np.sum(maps["Ion_flux"])
    else:
        return
    # The scale here is only for testing
    if prop in flux_quantities:
        column_name = prop[:-4] + "em_" + str(scale)
    else:
        column_name = prop + "_" + str(scale)
    df.loc[index, column_name] = quant
    return


def add_map_quantities(df, hdf_file, scale):
    for prop in hdf_file[scale][str(df.index[0])].keys():
        # to be corrected after scale is fixed
        column_name = prop + "_" + scale
        df[column_name] = np.nan
        for idx in df.index:
            get_df_quantity(prop, hdf_file, df, idx, scale)
    return


def update_map_df(
    snap_num,
    grid_scale,
    hdf_filename="maps.hdf5",
    df_name="df_full.pickle",
    output_name="df_fesc.pickle",
    base="/ptmp/mpa/ivkos/semianalytic_fesc",
):
    snap = get_snap(snap_num)
    hdf_path = os.path.join(base, snap, hdf_filename)
    origin_path = os.path.join(base, snap, df_name)
    destination_path = os.path.join(base, snap, output_name)

    hdf_file = h5py.File(hdf_path, "a")
    df = pd.read_pickle(origin_path)
    add_map_quantities(df, hdf_file, str(grid_scale))

    df.to_pickle(destination_path)
    hdf_file.close()
    return


if __name__ == "__main__":
    grid_sizes = [
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
        1.05,
        1.1,
        1.15,
        1.2,
        1.25,
        1.3,
        1.35,
        1.4,
        1.45,
        1.5,
        1.55,
        1.6,
        1.65,
        1.7,
        1.75,
        1.8,
        1.85,
        1.9,
        1.95,
        2.0,
    ]
    snap_num = 13
    for grid_size in grid_sizes:
        update_map_df(
            snap_num,
            grid_scale=grid_size,
            hdf_filename="maps2.hdf5",
            df_name="test_df2_updated.pickle",
            output_name="test_df2_updated.pickle",
            base="/ptmp/mpa/ivkos/semianalytic_fesc",
        )
        print(f"Done with {grid_size}")
