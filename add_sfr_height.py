import numpy as np
import os
import pandas as pd
from pyTNG import gridding
import illustris_python as il
import h5py
import pyTNG.utils as utils
from pyTNG.cosmology import TNGcosmo
from astropy import units as u
from utils import get_snap, get_sim, get_redshift
from gridded_maps_1 import create_particle_box
from config import config


def get_sfr_height(
    df,
    idx,
    sim_path,
    snap_num,
    grid_cen,
    z,
    grid_size,
):
    n_threads = 8
    gas = il.snapshot.loadSubhalo(sim_path, snap_num, idx, "gas")
    box_gas = create_particle_box(gas, df, idx, z)
    if box_gas == 0:
        print(f"Dropping halo {idx}: too few gas particles")
        df.drop(idx, inplace=True)
        return 0, 0, 0, 0, 0
    if box_gas["StarFormationRate"].sum() == 0:
        print(f"Dropping halo {idx}: no star formation")
        df.drop(idx, inplace=True)
        return 0, 0, 0, 0, 0
    utils.computeGasSmoothingLength(box_gas)

    dist_to_cm = (1 * u.kpc).to(u.cm).value / TNGcosmo.h / (1 + z)
    box_size = df.loc[idx, "r"] * 4 / dist_to_cm * np.ones(3)

    shape = (grid_size * np.ones(3)).astype(np.int64)

    grid_sfr = gridding.depositParticlesOnGrid(
        gas_parts=box_gas,
        method="sphKernelDep",
        quants=[],
        box_size_parts=[0, 0, 0],
        grid_shape=shape,
        grid_size=box_size,
        grid_cen=grid_cen,
        n_threads=n_threads,
        mass_key="StarFormationRate",
    )

    sfr_height = gridded_sfr_height(
        sfr_grid=grid_sfr["StarFormationRate"], radius=df.loc[idx, "r"]
    )
    return sfr_height


def gridded_sfr_height(sfr_grid, radius):
    grid_size = sfr_grid.shape[0]
    center = grid_size // 2
    ratio = 0
    interval = 1 - 1 / np.e
    grid_steps = 0
    total_sfr = np.sum(sfr_grid)
    while ratio < interval:
        grid_steps += 1
        enclosed_sfr = np.sum(
            sfr_grid[:, :, center - grid_steps : center + grid_steps]
        )
        ratio = enclosed_sfr / total_sfr
    column_height = 4 * radius / grid_size * grid_steps
    return column_height


def add_sfr_column(
    df_name,
    snap_num,
    base_path,
):

    snap = get_snap(snap_num)
    sim, sim_path = get_sim()
    z = get_redshift(sim, snap_num)

    df_filename = df_name + ".pickle"
    df_path = os.path.join(base_path, snap, df_filename)
    df = pd.read_pickle(df_path)

    grid_cen = np.zeros(3)

    sfr_heights = []
    for _, idx in enumerate(df.index):
        grid_size = 100
        sfr_height = get_sfr_height(
            df,
            idx,
            sim_path,
            snap_num,
            grid_cen,
            z,
            grid_size,
        )
        sfr_heights.append(sfr_height)

    sfr_column_name = "SFR_height"
    df[sfr_column_name] = sfr_heights
    df.to_pickle(df_path)
    return


if __name__ == "__main__":
    df_name = config["df_name"]
    scale = config["grid_size"]
    base = config["base_path"]
    df_ending = config["df_ending"]

    for snap in range(17):

        df_name_full = df_name + "_" + str(snap)

        add_sfr_column(
            df_name=df_name_full,
            snap_num=snap,
            base_path=base,
        )
        print(f"Finished working on snapshot {snap}")
