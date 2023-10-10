import os
import h5py
import numpy as np
import pandas as pd
from pyTNG.cosmology import TNGcosmo
import pyTNG.utils as utils
from utils import get_sim
from utils import get_redshift
from utils import get_snap
from config import config


def get_merger_snap(current_snap):
    snap_num = current_snap[2:]
    basepath = config["tng_datapath"]
    sim_name = "L35n2160TNG"
    merge_history = "postprocessing/MergerHistory/"
    catalog_name = f"MergerHistory_{snap_num}.hdf5"
    catalog_path = os.path.join(
        basepath, sim_name, merge_history, catalog_name
    )
    hdf_file = h5py.File(catalog_path, "r")
    return hdf_file


def get_time_between_snapshot(snap1, snap2):
    sim, _ = get_sim()
    z1 = get_redshift(sim, snap1)
    z2 = get_redshift(sim, snap2)
    time_since_merger = TNGcosmo.age(z2) - TNGcosmo.age(z1)
    return time_since_merger.value * 1000


def most_recent_merger(df):
    df["TimeRecentMerger"] = df[["TimeMajorMerger", "TimeMinorMerger"]].min(
        axis=1
    )
    return


def get_time_since_merger(merger_file, galaxy_id, snap_num):
    snap_major_merger = merger_file["SnapNumLastMajorMerger"][galaxy_id]
    snap_minor_merger = merger_file["SnapNumLastMinorMerger"][galaxy_id]
    if snap_major_merger != -1:
        time_since_major = get_time_between_snapshot(
            snap_major_merger, snap_num
        )
    else:
        time_since_major = np.inf
    if snap_minor_merger != -1:
        time_since_minor = get_time_between_snapshot(
            snap_minor_merger, snap_num
        )
    else:
        time_since_minor = np.inf

    return time_since_major, time_since_minor


def update_merger_time(
    df_prefix,
    snap_range=(1, 17),
    base_path=config["base_path"],
):
    for i in range(*snap_range):
        snap = get_snap(i)
        dir_path = os.path.join(base_path, snap)
        df_name = f"{df_prefix}{str(i)}.pickle"
        df_path = os.path.join(dir_path, df_name)
        df = pd.read_pickle(df_path)
        merger_file = get_merger_snap(snap)
        merger_times_major = []
        merger_times_minor = []
        for galaxy in df.index:
            time_since_major, time_since_minor = get_time_since_merger(
                merger_file, galaxy, i
            )
            merger_times_major.append(time_since_major)
            merger_times_minor.append(time_since_minor)
        merger_file.close()
        df["TimeMajorMerger"] = merger_times_major
        df["TimeMinorMerger"] = merger_times_minor
        df.to_pickle(df_path)
    return


def get_subhalo_df(snap_num):
    sim, _ = get_sim()
    dataset = next(sim.group_cat[snap_num].chunk_generator("subhalo"))
    keys_needed = ["SubhaloMass", "SubhaloMassType", "SubhaloSFR"]
    sub_dict = {key: dataset[key] for key in keys_needed}
    dataset_df = utils.dfFromArrDict(sub_dict)

    new_df = pd.DataFrame().assign(
        M_tot=dataset_df[("SubhaloMass", 0)],
        M_gas=dataset_df[("SubhaloMassType", 0)],
        M_star=dataset_df[("SubhaloMassType", 4)],
        SFR=dataset_df[("SubhaloSFR", 0)],
    )
    return new_df


def get_subhalo_list(snap_num):
    sim, _ = get_sim()
    dataset = next(sim.group_cat[snap_num].chunk_generator("halo"))
    subhalo_list = dataset["GroupFirstSub"]
    return subhalo_list


# Note: This function only works for the three snapshots examined in the numerical study
def get_snap_num(z):
    z_to_snap = {6: 13, 8: 8, 10: 4}
    return z_to_snap[z]


def add_subhalo_info(df_name, base_path=config["base_path"]):
    df_path = os.path.join(base_path, df_name)
    df = pd.read_pickle(df_path)
    df["GalaxyStarMass"] = np.nan
    df["GalaxyGasMass"] = np.nan
    df["GalaxyMass"] = np.nan
    df["GalaxySFR"] = np.nan
    df["GalaxyID"] = np.nan

    redshifts = [6, 8, 10]
    for z in redshifts:
        snap_num = get_snap_num(z)
        subhalo_catalog = get_subhalo_df(snap_num)
        subhalo_list = get_subhalo_list(snap_num)
        sub_df = df[df.z == z]
        for _, element in sub_df.iterrows():
            subhalo_id = subhalo_list[element.ID]
            df.loc[
                (df.z == z) & (df.ID == element.ID), "GalaxyStarMass"
            ] = subhalo_catalog.loc[subhalo_id]["M_star"]
            df.loc[
                (df.z == z) & (df.ID == element.ID), "GalaxyGasMass"
            ] = subhalo_catalog.loc[subhalo_id]["M_gas"]
            df.loc[
                (df.z == z) & (df.ID == element.ID), "GalaxyMass"
            ] = subhalo_catalog.loc[subhalo_id]["M_tot"]
            df.loc[
                (df.z == z) & (df.ID == element.ID), "GalaxySFR"
            ] = subhalo_catalog.loc[subhalo_id]["SFR"]
            df.loc[
                (df.z == z) & (df.ID == element.ID), "GalaxyID"
            ] = subhalo_id
    df["GalaxyID"] = df["GalaxyID"].astype(int)
    df.to_pickle(df_path)
    return


def add_relative_fractions(df_name, base_path=config["base_path"]):
    df_path = os.path.join(base_path, df_name)
    df = pd.read_pickle(df_path)
    df["GalaxyStarFraction"] = df["GalaxyStarMass"] / (
        df["GalaxyMass"] * df["FractionStars"]
    )
    df["GalaxyGasFraction"] = df["GalaxyGasMass"] / (
        df["GalaxyMass"] * df["FractionGas"]
    )
    df["GalaxyMassFraction"] = df["GalaxyMass"] / df["HaloMass"]
    df["GalaxySFRFraction"] = df["GalaxySFR"] / df["SFR"]
    df.to_pickle(df_path)
    return


def add_merger_times_to_numerical_df(
    numerical_df_name,
    base_path=config["base_path"],
):
    numerical_df_path = os.path.join(base_path, numerical_df_name)
    numerical_df = pd.read_pickle(numerical_df_path)

    numerical_df["TimeMajorMerger"] = np.nan
    numerical_df["TimeMinorMerger"] = np.nan

    redshifts = [6, 8, 10]
    for z in redshifts:
        numerical_sub_df = numerical_df[numerical_df.z == z]
        snap_num = get_snap_num(z)
        snap = get_snap(snap_num)
        merger_file = get_merger_snap(snap)

        for _, element in numerical_sub_df.iterrows():
            time_since_major, time_since_minor = get_time_since_merger(
                merger_file, element.GalaxyID, snap_num
            )
            numerical_df.loc[
                (numerical_df.z == z) & (numerical_df.ID == element.ID),
                "TimeMajorMerger",
            ] = np.float64(time_since_major)
            numerical_df.loc[
                (numerical_df.z == z) & (numerical_df.ID == element.ID),
                "TimeMinorMerger",
            ] = np.float64(time_since_minor)

    numerical_df.to_pickle(numerical_df_path)


# if __name__ == "__main__":
#     update_merger_time("test_", snap_range=(0, 3))
