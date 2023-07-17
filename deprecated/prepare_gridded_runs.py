import pandas as pd
import os
import numpy as np
from utils import get_sim, get_redshift, get_snap


base = "/ptmp/mpa/ivkos/semianalytic_fesc"
for snap_num in range(17):
    df_name = "df_" + str(snap_num) + ".pickle"
    df_gridded_name = "df_" + str(snap_num) + "_gridded.pickle"
    snap = get_snap(snap_num)
    sim, sim_path = get_sim()
    z = get_redshift(sim, snap_num)
    origin_path = os.path.join(base, snap, df_name)
    gridded_path = os.path.join(base, snap, df_gridded_name)
    df = pd.read_pickle(origin_path)
    df_gridded = df[["Halo_pos_x", "Halo_pos_y", "Halo_pos_z", "r"]]
    false_values = np.zeros(len(df)).astype(np.bool8)
    df_gridded.loc[:, "processed"] = false_values
    df_gridded.to_pickle(gridded_path)
