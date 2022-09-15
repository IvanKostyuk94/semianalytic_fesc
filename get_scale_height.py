import pandas as pd
import numpy as np
import os
import illustris_python as il
import astropy.units as u
from pyTNG.cosmology import TNGcosmo
from utils import get_particle_dist, get_sim, get_redshift, get_snap
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


def map_to_new_dict(particles, relevant):
    rel_particles = {}
    newcount_particles = (relevant == True).sum()
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


def select_box_particles(particles):
    if particles["count"] == 0:
        return particles
    else:
        idces_rel_particles = ~np.any(
            np.abs(particles["rel_pos_norm"]) > 1, axis=1
        )
        rel_particles = map_to_new_dict(particles, idces_rel_particles)

    return rel_particles


def select_sphere_particles(particles, df, idx, z, is_relative):
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


def create_particle_box(particles, df, idx, z, stars=None):
    get_relative_coord(particles, df, idx)
    sphere_particles = select_sphere_particles(
        particles, df, idx, z, is_relative=True
    )
    # Send a flag identifying that there are too few gas particles
    # for a proper resolution
    if sphere_particles["count"] < 5:
        if stars is not None:
            return 0, 0
        else:
            return 0
    pca = PCA(3)
    pca.fit(sphere_particles["Coordinates"])
    particles["Coordinates"] = pca.transform(particles["Coordinates"])
    get_normed_coord(particles, df, idx, z, is_relative=True)
    box_particles = select_box_particles(particles)
    if stars is not None:
        get_relative_coord(stars, df, idx)
        stars["Coordinates"] = pca.transform(stars["Coordinates"])
        get_normed_coord(stars, df, idx, z, is_relative=True)
        box_stars = select_box_particles(stars)
        return box_particles, box_stars
    else:
        return box_particles


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

        newcount_wind = (idces_wind == True).sum()
        newcount_stars = (idces_stars == True).sum()

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
        all_gas = gas
    return all_gas


# Adds scale height, gas and star-masses
def update_df_columns(df, sim_path, snap_num, z):
    dist_to_cm = (1 * u.kpc).to(u.cm).value / h / (1 + z)
    mass_to_g = (1 * u.Msun).to(u.g).value * 1e10 / h
    df.loc[:, "r"] *= dist_to_cm

    scale_heights = []
    gas_masses = []
    star_masses = []

    for idx in df.index:
        gas = il.snapshot.loadSubhalo(sim_path, snap_num, idx, "gas")
        wind_stars = il.snapshot.loadSubhalo(sim_path, snap_num, idx, "stars")
        wind, stars = separate_wind_stars(wind_stars)
        gas_wind = merge_gas_wind(gas, wind)
        box_particles, box_stars = create_particle_box(
            gas_wind, df, idx, z, stars
        )
        if box_particles == 0:
            print(f"Dropping halo {idx}: too few particles")
            df.drop(idx, inplace=True)
            continue
        scale_height = get_scale_height(box_particles)
        scale_heights.append(scale_height)
        gas_masses.append(box_particles["Masses"].sum())
        star_masses.append(box_stars["Masses"].sum())

    df["Column_height"] = np.array(scale_heights) * dist_to_cm
    df["Gas_mass"] = np.array(gas_masses) * mass_to_g
    df["Star_mass"] = np.array(star_masses) * mass_to_g
    return


def update_df_height(
    snap_num,
    df_name="0_df.pickle",
    base_path="/ptmp/mpa/ivkos/semianalytic_fesc",
    output_name="1_df.pickle",
):
    snap = get_snap(snap_num)
    sim, sim_path = get_sim()
    z = get_redshift(sim, snap_num)
    origin_path = os.path.join(base_path, snap, df_name)
    destination_path = os.path.join(base_path, snap, output_name)
    df = pd.read_pickle(origin_path)
    update_df_columns(df, sim_path, snap_num, z)
    df.to_pickle(destination_path)
    return


if __name__ == "__main__":
    snap_num = 13
    update_df_height(snap_num)
