import os
import h5py
import pandas as pd
import numpy as np
from utils import get_snap


def breakout_con(maps):
    return 6 * (
        np.array(maps["Dust_norm"])
        * np.array(maps["U"])
        * ((1 - np.array(maps["N_ratio"])) / np.array(maps["N_ratio"]))
    ) ** (1 / 3)


def get_breakout(maps):
    if "w_esc" in maps.keys():
        del maps["w_esc"]
    maps["w_esc"] = np.where(
        np.array(maps["N_ratio"]) < 1, breakout_con(maps), 0
    )
    return


def breakout_maps(
    snap_num,
    grid_size,
    hdf_name,
    df_name,
    base="/ptmp/mpa/ivkos/semianalytic_fesc",
):
    snap = get_snap(snap_num)
    hdf_filename = hdf_name + ".hdf5"
    hdf_path = os.path.join(base, snap, hdf_filename)

    df_filename = df_name + ".pickle"
    origin_path = os.path.join(base, snap, df_filename)

    hdf_file = h5py.File(hdf_path, "a")
    group = hdf_file[str(grid_size)]
    df = pd.read_pickle(origin_path)

    for idx in df.index:
        galaxy_name = str(idx)
        galaxy_group = group[galaxy_name]
        get_breakout(maps=galaxy_group)
    hdf_file.close()
    return


def add_breakout_frac(hdf_file, df, idx, scale, column_name):
    breakout_maps = hdf_file[str(scale)][str(idx)]["w_esc"]
    f_breakout = (
        np.sum(np.array(breakout_maps) > 1) / breakout_maps.shape[0] ** 2
    )
    df.loc[idx, column_name] = f_breakout
    return


def breakout_frac(
    snap_num,
    grid_size,
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
    column_name = "f_channel"
    df[column_name] = np.nan
    for idx in df.index:
        add_breakout_frac(hdf_file, df, idx, grid_size, column_name)

    df.to_pickle(destination_path)
    hdf_file.close()
    return


if __name__ == "__main__":
    snaps = np.arange(16, 17)
    grid_size = 100
    hdf_prefix = "gridded_maps_"
    df_prefix = "gridded_df_"
    for snap_num in snaps:
        hdf_name = hdf_prefix + str(snap_num)
        df_name = df_prefix + str(snap_num)
        print(f"Working on snap {snap_num}")
        breakout_maps(
            snap_num,
            grid_size,
            hdf_name,
            df_name,
            base="/ptmp/mpa/ivkos/semianalytic_fesc",
        )
        breakout_frac(
            snap_num,
            grid_size,
            hdf_name,
            df_name,
            output_name=None,
            base="/ptmp/mpa/ivkos/semianalytic_fesc",
        )
