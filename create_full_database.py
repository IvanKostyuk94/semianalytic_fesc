from build_initial_df_0 import build_new_df
from get_height_and_maps_1 import update_df_height_make_maps
from gridded_maps import grid_halos
from calculate_gridded_fesc_2 import update_maps
from maps_to_df_3 import update_map_df
from submit_runs import is_df_done
from utils import get_snap
import pandas as pd
import os
import config


def create_database(
    num,
    df_name=config.df_name,
    maps_name=config.maps_name,
    scale=config.grid_size,
    base=config.base_path,
    with_breakout=config.with_breakout,
):
    df_name_full = df_name + "_" + str(num)
    maps_name_full = maps_name + "_" + str(num)
    df_name_extension = df_name + "_" + str(num) + ".pickle"
    snap = get_snap(num)
    df_path = os.path.join(base, snap, df_name_full)
    try:
        df = pd.read_pickle(df_path)
        if not all(df["processed"]):
            build_new_df(snap_num=num, save_name=df_name_extension, base=base)
    except FileNotFoundError:
        build_new_df(snap_num=num, save_name=df_name_extension, base=base)
    grid_halos(
        df_name=df_name_full,
        snap_num=num,
        grid_scale=scale,
        hdf_name=maps_name_full,
    )
    # update_df_height_make_maps(
    #     snap_num=num,
    #     df_name=df_name_full,
    #     hdf_name=maps_name_full,
    #     to_hdf=True,
    #     adaptive=True,
    #     avg_dist_weighting=1,
    # )
    update_maps(
        snap_num=num,
        grid_size=scale,
        hdf_name=maps_name_full,
        df_name=df_name_full,
        with_breakout=with_breakout,
    )
    update_map_df(
        snap_num=num,
        scale=scale,
        hdf_name=maps_name_full,
        df_name=df_name_full,
    )
    return
