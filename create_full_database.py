from build_initial_df_0 import build_new_df
from gridded_maps import grid_halos
from calculate_gridded_fesc_2 import update_maps
from maps_to_df_3 import update_map_df
from utils import get_snap
from config import config

import pandas as pd
import os


def create_database(
    num,
    df_name=config["df_name"],
    maps_name=config["maps_name"],
    scale=config["grid_size"],
    base=config["base_path"],
    with_breakout=config["with_breakout"],
):
    df_name_full = df_name + "_" + str(num)
    maps_name_full = maps_name + "_" + str(num)
    snap = get_snap(num)
    df_path = os.path.join(base, snap, df_name_full)
    try:
        df = pd.read_pickle(df_path)
        if not all(df["processed"]):
            build_new_df(snap_num=num, save_name=df_name_full, base=base)
    except FileNotFoundError:
        build_new_df(snap_num=num, save_name=df_name_full, base=base)
    grid_halos(
        df_name=df_name_full,
        snap_num=num,
        grid_scale=scale,
        hdf_name=maps_name_full,
    )
    update_maps(
        snap_num=num,
        grid_size=scale,
        hdf_name=maps_name_full,
        df_name=df_name_full,
        base=base,
        with_breakout=with_breakout,
    )
    update_map_df(
        snap_num=num,
        scale=scale,
        hdf_name=maps_name_full,
        df_name=df_name_full,
        base=base,
    )
    return
