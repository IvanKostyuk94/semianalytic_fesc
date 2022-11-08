import pandas as pd
import numpy as np
import os
import h5py
import illustris_python as il
import astropy.units as u
from pyTNG.cosmology import TNGcosmo
from utils import (
    get_particle_dist,
    get_sim,
    get_redshift,
    get_snap,
    dist_to_cm,
    save_to_hdf,
)
from sklearn.decomposition import PCA
from scipy.stats import binned_statistic_2d
import sys


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


def select_box_gas(particles):
    if particles["count"] == 0:
        return particles
    else:
        idces_rel_particles = ~np.any(
            np.abs(particles["rel_pos_norm"]) > 1, axis=1
        )
        rel_particles = map_to_new_dict(particles, idces_rel_particles)

    return rel_particles


def select_sphere_gas(particles, df, idx, z, is_relative):
    if particles["count"] == 0:
        return particles
    else:
        get_particle_dist(particles, df, idx, z, is_relative=is_relative)
        idces_rel_particles = particles["rel_dist"] < 1
        rel_particles = map_to_new_dict(particles, idces_rel_particles)
    return rel_particles


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


def get_scale_height(box_particles, interval=None):
    data = box_particles["Coordinates"].T[2]
    if interval is None:
        interval = 1 - 1 / np.e
    data = sorted(data)
    lower = int(len(data) * (1 - interval) / 2)
    upper = int(len(data) * (1 + interval) / 2)
    scale_height = (data[upper] - data[lower]) / 2

    return scale_height


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


def merge_gas_wind(gas, wind):
    all_gas = {}
    if (gas["count"] > 0) and (wind["count"] > 0):
        for key in gas.keys():
            if (key in wind.keys()) and (key != "count"):
                all_gas[key] = np.append(gas[key], wind[key], axis=0)
        all_gas["count"] = gas["count"] + wind["count"]
    elif wind["count"] == 0:
        all_gas = gas.copy()
    return all_gas


def get_grid_cell_num(radius, approx_grid_size, z):
    approx_grid_size_cm = approx_grid_size * dist_to_cm(z)
    grid_cell_num = int(2 * radius / approx_grid_size_cm)
    # Set the cell number to one if the radius is very small
    if grid_cell_num == 0:
        grid_cell_num = 1
    grid_cell_size = 2 * radius / grid_cell_num
    return grid_cell_num, grid_cell_size


def get_adaptive_grid_cell_num(gas_box, avg_dist_weighting, radius):
    avg_dist = 2 * radius / np.sqrt(gas_box["count"])
    approx_grid_size = avg_dist * avg_dist_weighting
    grid_cell_num = int(2 * radius / approx_grid_size)
    if grid_cell_num == 0:
        grid_cell_num = 1
    grid_cell_size = 2 * radius / grid_cell_num
    return grid_cell_num, grid_cell_size


def get_gridded_surface_data(box_gas, box_stars, grid_cell_num):
    x_gas = box_gas["Coordinates"][:, 0]
    y_gas = box_gas["Coordinates"][:, 1]
    masses = box_gas["Masses"]
    binned_masses, *_ = binned_statistic_2d(
        x_gas,
        y_gas,
        values=masses,
        statistic="sum",
        bins=grid_cell_num,
    )
    sfr = box_gas["StarFormationRate"]
    binned_sfr, *_ = binned_statistic_2d(
        x_gas, y_gas, values=sfr, statistic="sum", bins=grid_cell_num
    )

    metallicities_masses = box_gas["GFM_Metallicity"] * box_gas["Masses"]
    binned_metallicities_masses, *_ = binned_statistic_2d(
        x_gas,
        y_gas,
        values=metallicities_masses,
        statistic="sum",
        bins=grid_cell_num,
    )

    binned_metallicities = binned_metallicities_masses / binned_masses

    x_stars = box_stars["Coordinates"][:, 0]
    y_stars = box_stars["Coordinates"][:, 1]
    stars = box_stars["Masses"]
    binned_stars, *_ = binned_statistic_2d(
        x_stars, y_stars, values=stars, statistic="sum", bins=grid_cell_num
    )

    mass_to_g = (1 * u.Msun).to(u.g).value * 1e10 / h
    maps = {}
    maps["M_gas"] = binned_masses * mass_to_g
    maps["M_star"] = binned_stars * mass_to_g
    maps["SFR"] = binned_sfr
    maps["Metallicity"] = binned_metallicities
    return maps


# Adds scale height, gas and star-masses
# Created a dict of maps of surface properties
def update_df_columns(
    df,
    sim_path,
    snap_num,
    z,
    approx_grid_size=None,
    to_hdf=False,
    hdf5_file=None,
    adaptive=False,
    avg_dist_weighting=None,
):
    mass_to_g = (1 * u.Msun).to(u.g).value * 1e10 / h

    scale_heights = []
    gas_masses = []
    star_masses = []
    grid_cell_sizes = []

    if adaptive:
        groupname = avg_dist_weighting
    else:
        groupname = approx_grid_size

    if to_hdf:
        if str(groupname) not in hdf5_file:
            print(f"Creating group for grid scale {groupname}")
            _ = hdf5_file.create_group(str(groupname))

    surface_maps = {}
    counter = 0
    for idx in df.index:
        gas = il.snapshot.loadSubhalo(sim_path, snap_num, idx, "gas")
        wind_stars = il.snapshot.loadSubhalo(sim_path, snap_num, idx, "stars")
        _, stars = separate_wind_stars(wind_stars)
        # Do not merge gas and wind since wind is mostly ionized
        # gas_wind = merge_gas_wind(gas, wind)
        box_gas, box_stars = create_particle_box(gas, df, idx, z, stars)
        if box_gas == 0:
            print(f"Dropping halo {idx}: too few gas particles")
            df.drop(idx, inplace=True)
            continue
        if box_gas["StarFormationRate"].sum() == 0:
            print(f"Dropping halo {idx}: no star formation")
            df.drop(idx, inplace=True)
            continue
        scale_height = get_scale_height(box_gas)
        scale_heights.append(scale_height)
        gas_masses.append(box_gas["Masses"].sum())
        star_masses.append(box_stars["Masses"].sum())

        radius = 2 * df.loc[idx, "r"]
        if adaptive:
            grid_cell_num, grid_cell_size = get_adaptive_grid_cell_num(
                box_gas, avg_dist_weighting, radius
            )
        else:
            grid_cell_num, grid_cell_size = get_grid_cell_num(
                radius, approx_grid_size, z
            )
        maps = get_gridded_surface_data(box_gas, box_stars, grid_cell_num)
        grid_cell_sizes.append(grid_cell_size)
        surface_maps[idx] = maps

        if to_hdf:
            if adaptive:
                save_to_hdf(hdf5_file, idx, avg_dist_weighting, maps)
            else:
                save_to_hdf(hdf5_file, idx, approx_grid_size, maps)

        counter += 1
        if counter % 100 == 0:
            sys.stdout.write(f"\r{counter/len(df)*100:.2f}% done")
            sys.stdout.flush()

    # This is temporarily for testing different grid_sizes
    # if adaptive:
    #     grid_column_name = "Grid_cell_size_" + str(avg_dist_weighting)
    # else:
    #     grid_column_name = "Grid_cell_size_" + str(approx_grid_size)
    grid_column_name = "Grid_cell_size"
    df["Column_height"] = np.array(scale_heights) * dist_to_cm(z)
    df["Gas_mass"] = np.array(gas_masses) * mass_to_g
    df["Star_mass"] = np.array(star_masses) * mass_to_g
    df[grid_column_name] = np.array(grid_cell_sizes)
    return


# Both the maps and the height are calculated at the same time
# in order to be able to use the particle box
def update_df_height_make_maps(
    snap_num,
    df_name,
    hdf_name,
    to_hdf=False,
    adaptive=False,
    base_path="/ptmp/mpa/ivkos/semianalytic_fesc",
    avg_dist_weighting=None,
    approx_grid_size=None,
):
    snap = get_snap(snap_num)
    sim, sim_path = get_sim()
    z = get_redshift(sim, snap_num)

    df_filename = df_name + ".pickle"
    origin_path = os.path.join(base_path, snap, df_filename)
    destination_path = os.path.join(base_path, snap, df_filename)
    if to_hdf:
        hdf_name_full = hdf_name + ".hdf5"
        hdf_path = os.path.join(base_path, snap, hdf_name_full)
        if not os.path.exists(hdf_path):
            hdf_file = h5py.File(hdf_path, "w")
            hdf_file.close()
        hdf_file = h5py.File(hdf_path, "a")
    else:
        hdf_file = None

    df = pd.read_pickle(origin_path)
    update_df_columns(
        df,
        sim_path,
        snap_num,
        z,
        approx_grid_size,
        to_hdf=to_hdf,
        hdf5_file=hdf_file,
        adaptive=adaptive,
        avg_dist_weighting=avg_dist_weighting,
    )
    if to_hdf:
        hdf_file.close()
    df.to_pickle(destination_path)
    return
