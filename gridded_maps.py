import pandas as pd
import numpy as np
import os
import h5py
import illustris_python as il
from get_height_and_maps_1 import create_particle_box, separate_wind_stars
from utils import get_snap, get_sim, get_redshift, save_to_hdf
import pyTNG.utils as utils
from pyTNG import gridding
from astropy import units as u
from pyTNG.cosmology import TNGcosmo

h = TNGcosmo.h


def get_gridded_quantities(
    df, idx, sim_path, snap_num, grid_cen, quants, z, grid_size
):
    gas = il.snapshot.loadSubhalo(sim_path, snap_num, idx, "gas")
    wind_stars = il.snapshot.loadSubhalo(sim_path, snap_num, idx, "stars")
    _, stars = separate_wind_stars(wind_stars)
    box_gas, box_stars = create_particle_box(gas, df, idx, z, stars)
    utils.computeGasSmoothingLength(box_gas)
    dist_to_cm = (1 * u.kpc).to(u.cm).value / h / (1 + z)
    box_size = df.loc[idx, "r"] * 4 / dist_to_cm * np.ones(3)
    # Convert back to cm as the grid_size is given in cm
    shape = np.ceil(box_size * dist_to_cm / grid_size).astype(np.int64)
    grids = gridding.depositParticlesOnGrid(
        gas_parts=box_gas,
        method="sphKernelDep",
        quants=quants,
        box_size_parts=box_size,
        grid_shape=shape,
        grid_size=box_size,
        grid_cen=grid_cen,
        n_threads=8,
    )
    grid_sfr = gridding.depositParticlesOnGrid(
        gas_parts=box_gas,
        method="sphKernelDep",
        quants=[],
        box_size_parts=box_size,
        grid_shape=shape,
        grid_size=box_size,
        grid_cen=grid_cen,
        n_threads=8,
        mass_key="StarFormationRate",
    )
    grid_star = gridding.depositParticlesOnGrid(
        gas_parts=box_stars,
        method="PIC",
        quants=[],
        box_size_parts=box_size,
        grid_shape=shape,
        grid_size=box_size,
        grid_cen=grid_cen,
        n_threads=8,
    )
    return grids, grid_sfr, grid_star


def gridded_to_maps(grids, grid_sfr, grid_star):
    mass_to_g = (1 * u.Msun).to(u.g).value * 1e10 / h
    maps = {}
    map_sfr = grid_sfr["StarFormationRate"].sum(axis=2)
    map_mass = grids["Masses"].sum(axis=2)
    map_metallicity = grids["GFM_Metallicity"].sum(axis=2) / map_mass
    map_star = grid_star["Masses"].sum(axis=2)
    maps["M_gas"] = map_mass * mass_to_g
    maps["M_star"] = map_star * mass_to_g
    maps["SFR"] = map_sfr
    maps["Metallicity"] = map_metallicity
    return maps


def grid_halos(
    df_name,
    snap_num,
    grid_scale,
    grid_sizes,
    hdf_name,
    base_path="/ptmp/mpa/ivkos/semianalytic_fesc",
):

    snap = get_snap(snap_num)
    sim, sim_path = get_sim()
    z = get_redshift(sim, snap_num)

    df_filename = df_name + ".pickle"
    df_path = os.path.join(base_path, snap, df_filename)
    df = pd.read_pickle(df_path)

    hdf_name_full = hdf_name + ".hdf5"
    hdf_path = os.path.join(base_path, snap, hdf_name_full)
    if not os.path.exists(hdf_path):
        hdf_file = h5py.File(hdf_path, "w")
        hdf_file.close()
    hdf5_file = h5py.File(hdf_path, "a")

    groupname = grid_scale
    if str(groupname) not in hdf5_file:
        print(f"Creating group for grid scale {groupname}")
        _ = hdf5_file.create_group(str(groupname))

    quants = ["GFM_Metallicity"]
    grid_cen = np.zeros(3)

    for idx, grid_size in zip(df.index, grid_sizes):
        grids, grid_sfr, grid_star = get_gridded_quantities(
            df,
            idx,
            sim_path,
            snap_num,
            grid_cen,
            quants,
            z,
            grid_size,
        )
        maps = gridded_to_maps(grids, grid_sfr, grid_star)
        save_to_hdf(hdf5_file, idx, grid_scale, maps)
