import numpy as np
import pandas as pd
import h5py
import os
from utils import get_snap


def get_average_N_d(maps):
    return 1 / np.sum(1 / np.array(maps["N_d"]))


def get_df_quantity(prop, hdf_file, df, index, scale):
    maps = hdf_file[scale][str(index)]
    summed_quantities = [
        "Bol_flux",
        "Dust_norm",
        "Ion_flux",
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
    add_map_quantities(df, hdf_file, grid_scale)

    df.to_pickle(destination_path)
    hdf_file.close()
    return


if __name__ == "__main__":
    grid_sizes = [
        "0.005",
        "0.01",
        "0.03",
        "0.05",
        "0.07",
        "0.09",
        "0.1",
        "0.2",
        "0.3",
        "0.5",
        "0.7",
        "1.0",
        "10",
        "3",
        "5",
        "7",
    ]
    snap_num = 13
    for grid_size in grid_sizes:
        update_map_df(
            snap_num,
            grid_scale=grid_size,
            hdf_filename="maps.hdf5",
            df_name="test_df_updated.pickle",
            output_name="test_df_updated.pickle",
            base="/ptmp/mpa/ivkos/semianalytic_fesc",
        )
        print(f"Done with {grid_size}")
