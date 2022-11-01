import pandas as pd
import numpy as np
import os
from utils import get_sim, get_redshift, get_snap


def merge_dfs(
    snap_min,
    snap_max,
    base="/ptmp/mpa/ivkos/semianalytic_fesc",
    destination_path=None,
    name="full_df",
):
    if destination_path is None:
        destination_path = os.join(base, name + ".pickle")

    for snap_num in range(snap_min, snap_max):
        df_name = "df_" + str(snap_num) + ".pickle"
        snap = get_snap(snap_num)
        sim, sim_path = get_sim()
        z = get_redshift(sim, snap_num)
        origin_path = os.path.join(base, snap, df_name)
        df = pd.read_pickle(origin_path)

        if snap_num == snap_min:
            df_dict = {}
            for key in df.columns:
                df_dict[key] = df[key].copy()
            df_dict["z"] = np.ones(len(df)) * z
        else:
            for key in df.columns:
                df_dict[key].extend(df[key].copy)
            df_dict["z"].extend(np.ones(len(df)) * z)

    full_df = pd.DataFrame.from_dict(df_dict)
    full_df.to_pickle(destination_path)
    return
