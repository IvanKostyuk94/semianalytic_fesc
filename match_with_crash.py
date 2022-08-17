import pandas as pd
import numpy as np
import os
from utils import get_sim


def get_parent_halo_ids(df, snap_num):
    sim, sim_path = get_sim()
    dataset = next(sim.group_cat[snap_num].chunk_generator("subhalo"))
    parent_halos = dataset["SubhaloGrNr"][np.array(df.index)]
    df["Parent"] = parent_halos
    return


def calculate_halo_fesc(df):
    fesc = df.groupby("Parent").apply(
        lambda gr: np.sum(gr["f_esc_sf_r"] * gr["Ion_em_sf_r"])
        / gr["Ion_em_sf_r"].sum()
    )
    return fesc


def get_crash_halos_z(z, df_name="full_esc_updated.pickle"):
    path_to_dfs = "/freya/u/ivkos/analysis/dfs"
    path_to_sel_df = os.path.join(path_to_dfs, df_name)
    full_df = pd.read_pickle(path_to_sel_df)
    df_z = full_df[full_df.z == z]
    return df_z


def add_semianalytic_fesc(df_z, fesc):
    df_z["fesc_semi"] = fesc.loc[df_z.index]
    return


def get_z(snap_num):
    z_values = {4: 10, 8: 8, 13: 6}
    try:
        return z_values[snap_num]
    except KeyError:
        raise KeyError("No Crash results are provided for this snapshot")


def get_crash_df_with_semi(df, snap_num, df_name="full_esc_updated.pickle"):
    get_parent_halo_ids(df, snap_num)
    fesc = calculate_halo_fesc(df)
    z = get_z(snap_num)
    df_z = get_crash_halos_z(z, df_name=df_name)
    add_semianalytic_fesc(df_z, fesc)
    return df_z
