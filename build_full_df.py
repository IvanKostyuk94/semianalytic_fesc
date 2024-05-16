import pandas as pd
import numpy as np
import os
from utils import get_sim, get_redshift, get_snap
from config import config


def merge_dfs(
    snap_min,
    snap_max,
    additional_snaps=[],
    base=config["base_path"],
    destination_path=None,
    name=config["full_df_name"],
    name_prefix_dfs=config["df_name"],
):
    """
    Takes all individual dataframes for various snapshots and creates a
    dataframe containing all data and an additional redshift column

    Parameters
    ----------
    snap_min : int
        first snapshot number to add to the dataframe
    snap_max : int
        last snapshot number to add to the dataframe
    base : str, optional
        Path where the snapshot directories are stored, by default
        "/ptmp/mpa/ivkos/semianalytic_fesc"
    destination_path : str, optional
        Path where to save the full df, by default it is the directory
        containing the snapshot subdirectories, by default None
    name : str, optional
        Name of the final df file, by default "full_df"
    """
    if destination_path is None:
        destination_path = os.path.join(base, name + ".pickle")

    snaps = np.arange(snap_min, snap_max)
    for element in additional_snaps:
        snaps = np.append(snaps, element)
    for snap_num in snaps:
        if name_prefix_dfs is None:
            df_name = "df_" + str(snap_num) + ".pickle"
        else:
            df_name = f"{name_prefix_dfs}_{snap_num}.pickle"
        snap = get_snap(snap_num)
        sim, sim_path = get_sim()
        z = get_redshift(sim, snap_num)
        origin_path = os.path.join(base, snap, df_name)
        df = pd.read_pickle(origin_path)

        if snap_num == snap_min:
            df_dict = {}
            for key in df.columns:
                df_dict[key] = list(df[key].copy())
            df_dict["z"] = list(np.ones(len(df)) * z)
            df_dict["idx"] = list(df.index)
        else:
            for key in df.columns:
                try:
                    df_dict[key].extend(list(df[key].copy()))
                except:
                    # del df_dict[key]
                    # continue
                    # Just for debugging
                    print(key)
                    print(snap_num)
                    print(f"{key} will be dropped from this dataframe")
                    if key in df_dict.keys():
                        del df_dict[key]

            df_dict["z"].extend(np.ones(len(df)) * z)
            df_dict["idx"].extend(df.index)

    # Hacky temporary solution to work with dfs with different columns
    keys_to_delete = []
    for key in df_dict.keys():
        if len(df_dict[key]) < len(df_dict["z"]):
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del df_dict[key]

    # for key in df_dict.keys():
    #     print(f"{key}: {len(df_dict[key])}")

    full_df = pd.DataFrame.from_dict(df_dict)
    # full_df.drop(columns=["100", "processed"], inplace=True)
    full_df.to_pickle(destination_path)
    return


if __name__ == "__main__":
    merge_dfs(
        0,
        17,
        additional_snaps=[21, 25],
    )
