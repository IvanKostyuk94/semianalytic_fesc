import pandas as pd
import numpy as np
from utils import get_sim


def get_parent_halo_ids(df, snap_num):
    sim, sim_path = get_sim()
    dataset = next(sim.group_cat[snap_num].chunk_generator("subhalo"))
    parent_halos = dataset["SubhaloGrNr"][np.array(df.index)]
    df["Parent"] = parent_halos
    return
