import pandas as pd
import numpy as np
import h5py
from config import config
import os
from utils import get_snap


def get_gini(map):
    map = map.ravel()
    map = np.where(np.isnan(map), 0, map)
    map = np.sort(map)
    index = np.arange(1, len(map) + 1)
    n = len(map)
    return (np.sum((2 * index - n - 1) * map)) / (n * np.sum(map))


def update_gini_values(df_prefix, map_prefix, snap_num):
    base_path = config["base_path"]

    df_name = f"{df_prefix}_{snap_num}.pickle"
    maps_name = f"{map_prefix}_{snap_num}.hdf5"

    sub_dir = get_snap(snap_num)

    df_path = os.path.join(base_path, sub_dir, df_name)
    maps_name = os.path.join(base_path, sub_dir, maps_name)

    df = pd.read_pickle(df_path)
    map_file = h5py.File(maps_name, "r")

    gini_sfr_arr = np.empty(len(df))
    gini_sfr_arr[:] = np.nan

    gini_fesc_arr = np.empty(len(df))
    gini_fesc_arr[:] = np.nan

    for i, gal_id in enumerate(df.index):
        map_sfr = np.array(
            map_file[str(config["grid_size"])][str(gal_id)]["SFR"]
        )
        map_fesc = np.array(
            map_file[str(config["grid_size"])][str(gal_id)]["f_esc"]
        )
        map_sfr_fesc = map_sfr * map_fesc

        gini_sfr = get_gini(map_sfr)
        gini_fesc = get_gini(map_sfr_fesc)

        gini_sfr_arr[i] = gini_sfr
        gini_fesc_arr[i] = gini_fesc

    map_file.close()
    df["Gini_sfr"] = gini_sfr_arr
    df["Gini_fesc"] = gini_fesc_arr

    df.to_pickle(df_path)
    return


def update_gini_range(
    snap_min,
    snap_max,
    df_prefix=config["df_name"],
    map_prefix=config["maps_name"],
):
    for i in range(snap_min, snap_max):
        print(f"Working on snapshot {i}")
        update_gini_values(df_prefix, map_prefix, i)
    return


if __name__ == "__main__":
    update_gini_range(0, 17)
