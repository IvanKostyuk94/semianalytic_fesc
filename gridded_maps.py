import pandas as pd
import numpy as np
import os
import h5py
import illustris_python as il
import pyTNG.utils as utils

from utils import (
    get_snap,
    get_sim,
    get_redshift,
    save_to_hdf,
    get_particle_dist,
)
from pyTNG import gridding
from astropy import units as u
from pyTNG.cosmology import TNGcosmo
from sklearn.decomposition import PCA


h = TNGcosmo.h


def get_normed_coord(particles, df, index, z, is_relative=False):
    gal_center = np.array(
        [
            df.loc[index]["Halo_pos_x"],
            df.loc[index]["Halo_pos_y"],
            df.loc[index]["Halo_pos_z"],
        ]
    )
    if is_relative:
        rel_pos = particles["Coordinates"]
    else:
        rel_pos = particles["Coordinates"] - gal_center
    dist_to_cm = (1 * u.kpc).to(u.cm).value / h / (1 + z)
    radius = df.loc[index]["r"] * 2 / dist_to_cm
    rel_pos_norm = rel_pos / radius
    particles["rel_pos_norm"] = rel_pos_norm
    return


def get_relative_coord(particles, df, idx):
    gal_center = np.array(
        [
            df.loc[idx]["Halo_pos_x"],
            df.loc[idx]["Halo_pos_y"],
            df.loc[idx]["Halo_pos_z"],
        ]
    )
    particles["Coordinates"] = particles["Coordinates"] - gal_center
    return


# Corrects the particle dictionary to only contain the particles in relevant
def map_to_new_dict(particles, relevant):
    rel_particles = {}
    newcount_particles = (relevant).sum()
    for key, value in particles.items():
        try:
            rel_particles[key] = value[relevant]
        # for Python scalars
        except TypeError as e:
            if "not subscriptable" in str(e):
                pass
            else:
                raise
        # for numpy scalars
        except IndexError as e:
            if "invalid index to scalar variable" in str(e):
                pass
            else:
                raise
    if "count" in particles:
        rel_particles["count"] = newcount_particles
    return rel_particles


def select_sphere_gas(particles, df, idx, z, is_relative):
    if particles["count"] == 0:
        return particles
    else:
        get_particle_dist(particles, df, idx, z, is_relative=is_relative)
        idces_rel_particles = particles["rel_dist"] < 1
        rel_particles = map_to_new_dict(particles, idces_rel_particles)
    return rel_particles


def select_box_gas(particles):
    if particles["count"] == 0:
        return particles
    else:
        idces_rel_particles = ~np.any(
            np.abs(particles["rel_pos_norm"]) > 1, axis=1
        )
        rel_particles = map_to_new_dict(particles, idces_rel_particles)

    return rel_particles


def separate_wind_stars(starAndWindParts):
    try:
        idces_wind = starAndWindParts["GFM_StellarFormationTime"] < 0
        idces_stars = starAndWindParts["GFM_StellarFormationTime"] > 0

        newcount_wind = (idces_wind).sum()
        newcount_stars = (idces_stars).sum()

        wind = {}
        stars = {}
        for key, value in starAndWindParts.items():
            try:
                wind[key] = value[idces_wind]
                stars[key] = value[idces_stars]
            # for Python scalars
            except TypeError as e:
                if "not subscriptable" in str(e):
                    pass
                else:
                    raise
            # for numpy scalars
            except IndexError as e:
                if "invalid index to scalar variable" in str(e):
                    pass
                else:
                    raise
        if "count" in starAndWindParts:
            wind["count"] = newcount_wind
            stars["count"] = newcount_stars
    except KeyError as e:
        if (
            str(e) == "GFM_StellarFormationTime"
            and starAndWindParts["count"] == 0
        ):
            pass
        else:
            raise

    return wind, stars


def create_particle_box(gas, df, idx, z, stars=None):
    get_relative_coord(gas, df, idx)
    sphere_gas = select_sphere_gas(gas, df, idx, z, is_relative=True)
    # Send a flag identifying that there are too few gas gas
    # for a proper resolution
    if sphere_gas["count"] < 5:
        if stars is not None:
            return 0, 0
        else:
            return 0
    pca = PCA(3)
    pca.fit(sphere_gas["Coordinates"])
    gas["Coordinates"] = pca.transform(gas["Coordinates"])
    get_normed_coord(gas, df, idx, z, is_relative=True)
    box_gas = select_box_gas(gas)

    if stars is not None:
        get_relative_coord(stars, df, idx)
        stars["Coordinates"] = pca.transform(stars["Coordinates"])
        get_normed_coord(stars, df, idx, z, is_relative=True)
        box_stars = select_box_gas(stars)
        return box_gas, box_stars
    else:
        return box_gas


def get_gridded_quantities(
    df,
    idx,
    sim_path,
    snap_num,
    grid_cen,
    quants,
    z,
    grid_size,
    fixed_grid=True,
):
    gas = il.snapshot.loadSubhalo(sim_path, snap_num, idx, "gas")
    wind_stars = il.snapshot.loadSubhalo(sim_path, snap_num, idx, "stars")
    _, stars = separate_wind_stars(wind_stars)
    box_gas, box_stars = create_particle_box(gas, df, idx, z, stars)
    if box_gas == 0:
        print(f"Dropping halo {idx}: too few gas particles")
        df.drop(idx, inplace=True)
        return 0, 0, 0, 0, 0
    if box_gas["StarFormationRate"].sum() == 0:
        print(f"Dropping halo {idx}: no star formation")
        df.drop(idx, inplace=True)
        return 0, 0, 0, 0, 0
    utils.computeGasSmoothingLength(box_gas)

    dist_to_cm = (1 * u.kpc).to(u.cm).value / h / (1 + z)
    box_size = df.loc[idx, "r"] * 4 / dist_to_cm * np.ones(3)

    if fixed_grid:
        shape = (grid_size * np.ones(3)).astype(np.int64)
        physical_grid_size = df.loc[idx, "r"] * 4 / grid_size
    else:
        shape = np.ceil(box_size * dist_to_cm / grid_size).astype(np.int64)

    n_threads = 40
    grids = gridding.depositParticlesOnGrid(
        gas_parts=box_gas,
        method="sphKernelDep",
        quants=quants,
        box_size_parts=[0, 0, 0],
        grid_shape=shape,
        grid_size=box_size,
        grid_cen=grid_cen,
        n_threads=n_threads,
    )
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
    grid_star = gridding.depositParticlesOnGrid(
        gas_parts=box_stars,
        method="PIC",
        quants=[],
        box_size_parts=[0, 0, 0],
        grid_shape=shape,
        grid_size=box_size,
        grid_cen=grid_cen,
        n_threads=n_threads,
    )
    column_height = gridded_column_height(
        mass_grid=grids["Masses"], radius=df.loc[idx, "r"]
    )
    if fixed_grid:
        return grids, grid_sfr, grid_star, physical_grid_size, column_height
    else:
        return grids, grid_sfr, grid_star, 0, column_height


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


def gridded_column_height(mass_grid, radius):
    grid_size = mass_grid.shape[0]
    center = grid_size // 2
    ratio = 0
    interval = 1 - 1 / np.e
    grid_steps = 0
    total_mass = np.sum(mass_grid)
    while ratio < interval:
        grid_steps += 1
        enclosed_mass = np.sum(
            mass_grid[:, :, center - grid_steps : center + grid_steps]
        )
        ratio = enclosed_mass / total_mass
    column_height = 4 * radius / grid_size * grid_steps
    return column_height


def grid_halos(
    df_name,
    snap_num,
    grid_scale,
    hdf_name,
    physical_grid_sizes=None,
    base_path="/ptmp/mpa/ivkos/semianalytic_fesc",
    testing=False,
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

    grid_sizes = []
    column_heights = []
    for i, idx in enumerate(df.index):
        # This is for batch runs not able to finish a full df in 24h
        if "processed" in df.columns:
            if df.loc[idx, "processed"]:
                grid_sizes.append(df.loc[idx, "Grid_cell_size"])
                column_heights.append(df.loc[idx, "Column_height"])
                continue
        if physical_grid_sizes is None:
            grid_size = grid_scale
            fixed_grid = True
        else:
            grid_size = physical_grid_sizes[i]
            fixed_grid = False

        (
            grids,
            grid_sfr,
            grid_star,
            physical_grid_size,
            column_height,
        ) = get_gridded_quantities(
            df,
            idx,
            sim_path,
            snap_num,
            grid_cen,
            quants,
            z,
            grid_size,
            fixed_grid=fixed_grid,
        )
        if grids == 0:
            continue
        grid_sizes.append(physical_grid_size)
        column_heights.append(column_height)
        maps = gridded_to_maps(grids, grid_sfr, grid_star)
        save_to_hdf(hdf5_file, idx, grid_scale, maps)
        df.loc[idx, "processed"] = True

    if testing:
        grid_column_name = "Grid_cell_size_" + str(grid_scale)
        height_column_name = "Column_height_" + str(grid_scale)
    else:
        grid_column_name = "Grid_cell_size"
        height_column_name = "Column_height"
    if physical_grid_sizes is None:
        df[grid_column_name] = grid_sizes
    df[height_column_name] = column_heights
    df.to_pickle(df_path)
    return
