import pandas as pd
import numpy as np
from config import config
import os
from utils import get_snap
from utils import get_redshift_from_snap
from utils import scale_factor
from astropy import units as u
from pyTNG.cosmology import TNGcosmo


def select_similar_mass_df(df, binsize, stepsize):
    lower_mass = 6.0
    similar_mass_dfs = []
    while lower_mass < df["M_gas_sun_log"].max():
        upper_mass = lower_mass + binsize
        sub_df = df[
            (df.M_gas_sun_log > lower_mass) & (df.M_gas_sun_log < upper_mass)
        ].copy()
        similar_mass_dfs.append(sub_df)
        lower_mass += stepsize
    return similar_mass_dfs


def add_similar_mass_neighbor(df, z, binsize=0.7, stepsize=0.2):
    g_to_msun = (1 * u.g).to(u.M_sun)
    df["M_gas_sun_log"] = np.log10(df["M_gas"] * g_to_msun)
    similar_mass_dfs = select_similar_mass_df(df, binsize, stepsize)
    new_sub_dfs = []
    for sub_df in similar_mass_dfs:
        new_sub_dfs.append(add_nearest_neighbors(sub_df, z))

    df["Dist_nearest_sim"] = None
    df["Dist_5_sim"] = None
    df["Dist_10_sim"] = None
    df["Dist_32_sim"] = None

    lower_mass = 6.0
    upper_mass = 6.5

    while lower_mass < df["M_gas_sun_log"].max():
        for sub_df in new_sub_dfs:
            for idx, element in sub_df.iterrows():
                if (element.M_gas_sun_log < upper_mass) and (
                    element.M_gas_sun_log > lower_mass
                ):
                    # if idx < 100:
                    #     print(element.Dist_5)
                    df.loc[idx, "Dist_nearest_sim"] = element.Dist_nearest
                    df.loc[idx, "Dist_5_sim"] = element.Dist_5
                    df.loc[idx, "Dist_10_sim"] = element.Dist_10
                    df.loc[idx, "Dist_32_sim"] = element.Dist_32
        lower_mass = upper_mass
        upper_mass += stepsize

    return df


def add_nearest_neighbors(df, z):
    x_coord = np.array(df.Halo_pos_x, dtype="float32").reshape(len(df), 1)
    y_coord = np.array(df.Halo_pos_y, dtype="float32").reshape(len(df), 1)
    z_coord = np.array(df.Halo_pos_z, dtype="float32").reshape(len(df), 1)
    coord = np.concatenate((x_coord, y_coord, z_coord), axis=1)

    differences = coord[:, np.newaxis, :] - coord[np.newaxis, :, :]

    distances = (
        np.linalg.norm(differences, axis=2) / TNGcosmo.h * scale_factor(z)
    )
    distances = np.sort(distances, axis=1)

    try:
        nearest_neighbors = distances[:, 1]
    except:
        nearest_neighbors = [None] * len(distances)
    avg_5 = np.mean(distances[:, 1:6], axis=1)
    avg_10 = np.mean(distances[:, 1:11], axis=1)
    avg_32 = np.mean(distances[:, 1:33], axis=1)

    df["Dist_nearest"] = nearest_neighbors
    df["Dist_5"] = avg_5
    df["Dist_10"] = avg_10
    df["Dist_32"] = avg_32

    return df


def add_similar_mass_neighbor_count(df, z, ns, binsize=0.7, stepsize=0.2):
    g_to_msun = (1 * u.g).to(u.M_sun)
    df["M_gas_sun_log"] = np.log10(df["M_gas"] * g_to_msun)
    similar_mass_dfs = select_similar_mass_df(df, binsize, stepsize)
    for n in ns:
        new_sub_dfs = []
        for sub_df in similar_mass_dfs:
            new_sub_dfs.append(count_neighbors_in_nr(sub_df, z, n))

        sub_df_column_name = f"neighbors_{n}r"
        df_column_name = f"Neighbors_{n}r_sim"
        df[df_column_name] = None

        lower_mass = 6.0
        upper_mass = 6.5

        while lower_mass < df["M_gas_sun_log"].max():
            for sub_df in new_sub_dfs:
                for idx, element in sub_df.iterrows():
                    if (element.M_gas_sun_log < upper_mass) and (
                        element.M_gas_sun_log > lower_mass
                    ):
                        # if idx < 100:
                        #     print(element.Dist_5)
                        df.loc[idx, df_column_name] = element[
                            sub_df_column_name
                        ]
            lower_mass = upper_mass
            upper_mass += stepsize

    return df


def count_neighbors_in_nr(df, z, n):
    cm_to_kpc = (1 * u.cm).to(u.kpc).value
    nr_kpc = np.array(n * df["r"] * cm_to_kpc)
    x_coord = np.array(df.Halo_pos_x, dtype="float32").reshape(len(df), 1)
    y_coord = np.array(df.Halo_pos_y, dtype="float32").reshape(len(df), 1)
    z_coord = np.array(df.Halo_pos_z, dtype="float32").reshape(len(df), 1)
    coord = np.concatenate((x_coord, y_coord, z_coord), axis=1)

    differences = coord[:, np.newaxis, :] - coord[np.newaxis, :, :]

    distances = (
        np.linalg.norm(differences, axis=2) / TNGcosmo.h * scale_factor(z)
    )

    are_neighbors = np.less(distances, nr_kpc[:, None])
    neighbor_count = np.sum(are_neighbors, axis=1) - 1

    column_name = f"neighbors_{n}r"
    df[column_name] = neighbor_count
    return df


def get_df_path(df_prefix, snap_num):
    base_path = config["base_path"]
    df_name = f"{df_prefix}_{snap_num}.pickle"
    sub_dir = get_snap(snap_num)
    df_path = os.path.join(base_path, sub_dir, df_name)
    return df_path


def update_neares_neighbors(
    snap_min,
    snap_max,
    df_prefix=config["df_name"],
):
    for i in range(snap_min, snap_max):
        print(f"Working on snapshot {i}")
        df_path = get_df_path(df_prefix, i)
        df = pd.read_pickle(df_path)
        z = get_redshift_from_snap(i)
        # df = add_nearest_neighbors(df, z)
        # df = add_similar_mass_neighbor(df, z)
        df = add_similar_mass_neighbor_count(df, z, ns=[333, 666, 2000])
        df.to_pickle(df_path)
    return


if __name__ == "__main__":
    update_neares_neighbors(0, 17, df_prefix="new_df")
