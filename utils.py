import os
import pyTNG.utils as utils
from pyTNG import data_interface as _data_interface
import pandas as pd
import numpy as np
from pyTNG.cosmology import TNGcosmo
from astropy import units as u
import random


h = TNGcosmo.h


def dist_to_cm(z):
    return (1 * u.kpc).to(u.cm).value / h / (1 + z)


def get_sim():
    basepath = "/virgotng/universe/IllustrisTNG/"
    sim_name = "L35n2160TNG"
    sim = _data_interface.TNG50Simulation(os.path.join(basepath, sim_name))
    sim_path = os.path.join(basepath, sim_name, "output")
    return sim, sim_path


def get_dataset_df(sim, snap_num):
    dataset = next(sim.group_cat[snap_num].chunk_generator("subhalo"))
    keys_needed = [
        # "SubhaloGasMetallicity",
        # "SubhaloGasMetallicityHalfRad",
        "SubhaloHalfmassRadType",
        # "SubhaloMassInHalfRad",
        # "SubhaloMassInHalfRadType",
        # "SubhaloMassInRad",
        "SubhaloMassInRadType",
        # "SubhaloSFRinHalfRad",
        "SubhaloSFRinRad",
        "SubhaloPos",
    ]
    sub_dict = {key: dataset[key] for key in keys_needed}
    dataset_df = utils.dfFromArrDict(sub_dict)
    return dataset_df


def reduce_df(df):
    filt = (
        (df[("SubhaloMassInRadType", 4)] * 1e10 / h > 5e5)
        & (df[("SubhaloSFRinRad", 0)] > 0)
        & (df[("SubhaloMassInRadType", 0)] * 1e10 / h > 2e5)
    )
    reduced_df = df[filt]

    new_df = pd.DataFrame().assign(
        # Z_2r=reduced_df[("SubhaloGasMetallicity", 0)],
        # Z_r=reduced_df[("SubhaloGasMetallicityHalfRad", 0)],
        r=reduced_df[("SubhaloHalfmassRadType", 4)],
        # M_gas_r=reduced_df[("SubhaloMassInHalfRadType", 0)],
        # M_gas_2r=reduced_df[("SubhaloMassInRadType", 0)],
        # M_star_r=reduced_df[("SubhaloMassInHalfRadType", 4)],
        # M_star_2r=reduced_df[("SubhaloMassInRadType", 4)],
        # SFR_r=reduced_df[("SubhaloSFRinHalfRad", 0)],
        # SFR_2r=reduced_df[("SubhaloSFRinRad", 0)],
        Halo_pos_x=reduced_df[("SubhaloPos", 0)],
        Halo_pos_y=reduced_df[("SubhaloPos", 1)],
        Halo_pos_z=reduced_df[("SubhaloPos", 2)],
        processed=np.zeros(len(reduced_df)).astype(np.bool8),
    )
    return new_df


# Note that this function assumes that df['r'] is in units of cm
# and is therefore converted to kpc/h
def get_particle_dist(particles, df, index, z, is_relative=False):
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
    dist = np.sqrt(np.sum(np.square(rel_pos), axis=1))
    rel_dist = dist / radius
    particles["rel_dist"] = rel_dist
    return


def get_stellar_dist_gas(stars, gas_coord, gas_rad):
    rel_pos = stars["Coordinates"] - gas_coord
    dist = np.sqrt(np.sum(np.square(rel_pos), axis=1))
    rel_dist = dist / gas_rad
    stars["rel_dist"] = rel_dist
    return


def get_redshift(sim, snap_num):
    z = sim.snap_cat[snap_num].header["Redshift"]
    return z


def get_snap(snap_num):
    if snap_num < 10:
        return f"sn00{snap_num}"
    else:
        return f"sn0{snap_num}"


# Returns a subdf of galaxies with various masses
# Returned df does not necessarely contain 'sample_size'
# elements as some bins do not contain a galaxy
def select_sample_df(
    df, column="M_star_2r", sample_size=100, log=True, with_weights=False
):
    if log:
        group_values = np.log10(df[column])
    else:
        group_values = df[column]
    bins = pd.cut(group_values, sample_size)
    selected_idces = []
    num_per_bin = []
    for idces in group_values.groupby(bins).groups.values():
        if len(idces) > 0:
            num_per_bin.append(len(idces))
            selected_idces.append(random.choice(idces))
    selected_idces = selected_idces[::-1]
    test_df = df.loc[selected_idces].copy()
    if with_weights:
        return test_df, np.array(num_per_bin) / np.sum(num_per_bin)
    else:
        return test_df


def save_to_hdf(hdf_file, idx, approx_grid_size, maps):
    galaxy = str(idx)
    group = hdf_file[str(approx_grid_size)]
    if galaxy not in group:
        galaxy_group = group.create_group(galaxy)
    else:
        galaxy_group = group[galaxy]

    for map_name in maps:
        if map_name not in galaxy_group:
            _ = galaxy_group.create_dataset(map_name, data=maps[map_name])
    return


def load_df(path):
    df_full = pd.read_pickle(path)
    df_full.dropna(inplace=True, subset="f_esc")
    g_to_msun = (1 * u.g).to(u.M_sun)
    df_full["M_gas_sun_log"] = np.log10(df_full["M_gas"] * g_to_msun)
    df_full["M_star_sun_log"] = np.log10(df_full["M_star"] * g_to_msun)
    df_full.dropna(subset="f_esc", inplace=True)
    df_full.dropna(subset="f_g_crit", inplace=True)
    df_full.drop(df_full[df_full["M_star_sun_log"] < 5.55].index, inplace=True)
    return df_full
