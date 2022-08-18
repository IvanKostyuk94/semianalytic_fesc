from email.message import EmailMessage
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


def get_sfr_em(df):
    df["Ion_em_sfr_r"] = df["Ion_flux_sf_r"] * np.pi * df["r"] ** 2
    df["Ion_em_sfr_2r"] = df["Ion_flux_sf_2r"] * np.pi * (df["r"] * 2) ** 2
    return


def emissivity(fesc_prop):
    emissivity_dict = {
        "f_esc_r": "Ion_em_sfr_r",
        "f_esc_2r": "Ion_em_sfr_2r",
        "f_esc_sf_r": "Ion_em_sf_r",
        "f_esc_sf_2r": "Ion_em_sf_2r",
    }
    return emissivity_dict[fesc_prop]


def calculate_halo_fesc(df, fesc_types=None):
    if fesc_types == None:
        fesc_types = ["f_esc_r", "f_esc_2r", "f_esc_sf_r", "f_esc_sf_2r"]

    fesc = {}
    for type in fesc_types:
        fesc[type] = (
            df.fillna(0)
            .groupby("Parent")
            .apply(
                lambda gr: np.sum(gr[type] * gr[emissivity(type)])
                / gr[emissivity(type)].sum()
            )
        )
    return fesc


def calculate_halo_Q0(df):
    halo_Q0 = df.fillna(0).groupby("Parent").apply(lambda gr: gr["Ion_em_sf_2r"].sum())
    return halo_Q0


def get_crash_halos_z(z, df_name="full_esc_updated.pickle"):
    path_to_dfs = "/freya/u/ivkos/analysis/dfs"
    path_to_sel_df = os.path.join(path_to_dfs, df_name)
    full_df = pd.read_pickle(path_to_sel_df)
    df_z = full_df[full_df.z == z]
    df_z.set_index("ID", inplace=True)
    return df_z


def add_semianalytic_fesc(df_z, fesc):
    for key in fesc:
        df_z[key + "_semi"] = np.nan
        used_idx = list(set(df_z.index).intersection(set(fesc[key].index)))
        df_z.loc[used_idx, key + "_semi"] = fesc[key].loc[used_idx]
    return


def add_semianalytic_Q0(df_z, halo_Q0):
    df_z["halo_Q0_semi"] = np.nan
    used_idx = list(set(df_z.index).intersection(set(halo_Q0.index)))
    df_z.loc[used_idx, "halo_Q0_semi"] = halo_Q0.loc[used_idx]
    return


def get_z(snap_num):
    z_values = {4: 10, 8: 8, 13: 6}
    try:
        return z_values[snap_num]
    except KeyError:
        raise KeyError("No Crash results are provided for this snapshot")


def get_crash_df_with_semi(
    df, snap_num, df_name="full_esc_updated.pickle", fesc_types=None
):
    get_parent_halo_ids(df, snap_num)
    get_sfr_em(df)
    fesc = calculate_halo_fesc(df, fesc_types)
    halo_Q0 = calculate_halo_Q0(df)
    z = get_z(snap_num)
    df_z = get_crash_halos_z(z, df_name=df_name)
    add_semianalytic_fesc(df_z, fesc)
    add_semianalytic_Q0(df_z, halo_Q0)
    return df_z
