import os
import h5py
import pandas as pd
import numpy as np
from utils import get_snap
from config import config


def get_average_column_dens(maps):
    esc_mask = np.array(maps["f_esc"]) > 0
    masked_N = esc_mask * np.array(maps["Column_dens"])
    masked_Nd = esc_mask * np.array(maps["N_d"])
    masked_NS = esc_mask * np.array(maps["Column_dens_stroemgren"])

    relevant_N = masked_N[masked_N > 0]
    relevant_Nd = masked_Nd[masked_Nd > 0]
    relevant_NS = masked_NS[masked_NS > 0]

    average_N = relevant_N.sum() / len(relevant_N)
    average_Nd = relevant_Nd.sum() / len(relevant_Nd)
    average_NS = relevant_NS.sum() / len(relevant_NS)

    return average_N, average_Nd, average_NS


def update_average_column_dens(df_prefix, map_prefix, snap_num):
    base_path = config["base_path"]

    df_name = f"{df_prefix}_{snap_num}.pickle"
    maps_name = f"{map_prefix}_{snap_num}.hdf5"

    sub_dir = get_snap(snap_num)

    df_path = os.path.join(base_path, sub_dir, df_name)
    maps_name = os.path.join(base_path, sub_dir, maps_name)

    df = pd.read_pickle(df_path)
    map_file = h5py.File(maps_name, "r")

    average_column_dens = np.empty(len(df))
    average_column_dens[:] = np.nan

    average_column_dens_dust = np.empty(len(df))
    average_column_dens_dust[:] = np.nan

    average_ionizable_column_dens = np.empty(len(df))
    average_ionizable_column_dens[:] = np.nan

    for i, gal_id in enumerate(df.index):
        maps = map_file[str(config["grid_size"])][str(gal_id)]
        average_N, average_Nd, average_NS = get_average_column_dens(maps)

        average_column_dens[i] = average_N
        average_column_dens_dust[i] = average_Nd
        average_ionizable_column_dens[i] = average_NS

    map_file.close()

    df["AverageColumnDens"] = average_column_dens
    df["AverageColumnDensDust"] = average_column_dens_dust
    df["AverageIonizableColumnDens"] = average_ionizable_column_dens

    df.to_pickle(df_path)
    return


def update_column_dens_range(
    snap_min,
    snap_max,
    df_prefix=config["df_name"],
    map_prefix=config["maps_name"],
):
    for i in range(snap_min, snap_max):
        print(f"Working on snapshot {i}")
        update_average_column_dens(df_prefix, map_prefix, i)
    return


if __name__ == "__main__":
    update_column_dens_range(0, 17)
