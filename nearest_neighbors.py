import pandas as pd
import numpy as np
from config import config
import os
from utils import get_snap


def add_neares_neighbors(df_prefix, snap_num):

    base_path = config["base_path"]
    df_name = f"{df_prefix}_{snap_num}.pickle"
    sub_dir = get_snap(snap_num)
    df_path = os.path.join(base_path, sub_dir, df_name)
    df = pd.read_pickle(df_path)

    x_coord = np.array(df.Halo_pos_x, dtype="float32").reshape(len(df), 1)
    y_coord = np.array(df.Halo_pos_y, dtype="float32").reshape(len(df), 1)
    z_coord = np.array(df.Halo_pos_z, dtype="float32").reshape(len(df), 1)
    coord = np.concatenate((x_coord, y_coord, z_coord), axis=1)

    differences = coord[:, np.newaxis, :] - coord[np.newaxis, :, :]

    distances = np.linalg.norm(differences, axis=2)
    distances = np.sort(distances, axis=1)

    nearest_neighbors = distances[:, 1]
    avg_5 = np.mean(distances[:, 1:6], axis=1)
    avg_10 = np.mean(distances[:, 1:11], axis=1)
    avg_32 = np.mean(distances[:, 1:33], axis=1)

    df["Dist_nearest"] = nearest_neighbors
    df["Dist_5"] = avg_5
    df["Dist_10"] = avg_10
    df["Dist_32"] = avg_32

    df.to_pickle(df_path)
    return


def update_neares_neighbors(
    snap_min,
    snap_max,
    df_prefix=config["df_name"],
):
    for i in range(snap_min, snap_max):
        print(f"Working on snapshot {i}")
        add_neares_neighbors(df_prefix, i)
    return


if __name__ == "__main__":
    update_neares_neighbors(0, 17)
