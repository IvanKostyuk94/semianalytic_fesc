import h5py
import os
import numpy as np
import pandas as pd
from config import config


def get_progenitor(idx, snap, history):
    ID_raw = int(snap * 1e12 + idx)
    id_array = np.array(history["SubhaloIDRaw"])
    subhalo_pos = np.where(id_array == ID_raw)
    return history["FirstProgenitorID"][subhalo_pos]


def create_z_snap_dict(df):
    z_snap = {}
    for i, z in enumerate(df.z.unique()):
        z_snap[z] = i
    return z_snap


def add_snap_column(df):
    z_snap = create_z_snap_dict(df)
    df["snap"] = [z_snap[z] for z in df.z]
    return


def add_progenitors(
    df_name=config["full_df_name"],
    base_path=config["base_path"],
    merger_history_path=config["merger_history_path"],
):
    df_path = os.path.join(base_path, df_name)
    df = pd.read_pickle(df_path)

    add_snap_column(df)
    history = h5py.File(merger_history_path)

    df["progenitor"] = None

    for i, element in df.iterrows():
        snap = element.snap
        idx = element.idx
        progenitor_id = get_progenitor(idx, snap, history)
        df.loc[i]["progenitor"] = progenitor_id

    df.to_pickle(df_path)
    return
