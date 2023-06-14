import os
import h5py
import numpy as np
import pandas as pd
from pyTNG.cosmology import TNGcosmo
import pyTNG.utils as utils
from pyTNG import data_interface as _data_interface
from utils import get_sim
from utils import get_redshift
from utils import get_snap


def get_merger_snap(current_snap):
    snap_num = current_snap[2:]
    basepath = "/virgotng/universe/IllustrisTNG/"
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


def update_merger_time(
    df_prefix,
    snap_range=(1, 17),
    base_path="/ptmp/mpa/ivkos/semianalytic_fesc",
):
    for i in range(*snap_range):
        snap = get_snap(i)
        dir_path = os.path.join(base_path, snap)
        df_name = df_prefix + str(i) + ".pickle"
        df_path = os.path.join(dir_path, df_name)
        df = pd.read_pickle(df_path)
        merger_file = get_merger_snap(snap)
        merger_times_major = []
        merger_times_minor = []
        for galaxy in df.index:
            snap_major_merger = merger_file["SnapNumLastMajorMerger"][galaxy]
            snap_minor_merger = merger_file["SnapNumLastMinorMerger"][galaxy]
            if snap_major_merger != -1:
                time_since_major = get_time_between_snapshot(
                    snap_major_merger, i
                )
            else:
                time_since_major = np.inf
            if snap_minor_merger != -1:
                time_since_minor = get_time_between_snapshot(
                    snap_minor_merger, i
                )
            else:
                time_since_minor = np.inf
            merger_times_major.append(time_since_major)
            merger_times_minor.append(time_since_minor)
        merger_file.close()
        df["TimeMajorMerger"] = merger_times_major
        df["TimeMinorMerger"] = merger_times_minor
        df.to_pickle(df_path)
    return


if __name__ == "__main__":
    update_merger_time("gridded_df_", snap_range=(0, 17))
