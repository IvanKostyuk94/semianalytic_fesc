import os
import numpy as np
import pandas as pd
import illustris_python as il
from astropy import units as u
from pyTNG.cosmology import TNGcosmo
from config import config
from utils import scale_factor, get_redshift, get_sim, get_snap
from gridded_maps import select_sphere_gas, get_relative_coord

h = TNGcosmo.h


def get_pec_vel(gas, z):
    km_to_cm = 1e5
    gas["PeculiarVelocities"] = (
        gas["Velocities"] * np.sqrt(scale_factor(z)) * km_to_cm
    )
    return


def get_pos_cm(gas, z):
    pos_to_cm = (1 * u.kpc).to(u.cm).value / h / (1 + z)
    gas["RelativePosition_cm"] = gas["Coordinates"] * pos_to_cm
    return


def mass_to_g(gas):
    mass_to_g = (1 * u.Msun).to(u.g).value * 1e10 / h
    gas["Mass_g"] = gas["Masses"] * mass_to_g
    return


def angular_momentum(gas):
    r_cross_v = np.cross(gas["RelativePosition_cm"], gas["PeculiarVelocities"])
    gas["AngularMomenta"] = (gas["Mass_g"] * r_cross_v.T).T
    tot_angular_momentum = gas["AngularMomenta"].sum(axis=0)
    return tot_angular_momentum


def center_of_mass(gas):
    mass_weighted_coord = (gas["Mass_g"] * gas["RelativePosition_cm"].T).T
    center_of_mass = mass_weighted_coord.sum(axis=0) / np.sum(gas["Mass_g"])
    return center_of_mass


def center_of_sfr_mass(gas):
    gas_used = gas["StarFormationRate"] > 0
    gas_sfr = gas["StarFormationRate"][gas_used]
    gas_masses = gas["Mass_g"][gas_used]
    gas_coord = gas["RelativePosition_cm"][gas_used]

    center_of_mass = np.average(gas_coord, axis=0, weights=gas_masses)
    center_of_sfr = np.average(gas_coord, axis=0, weights=gas_sfr)

    return center_of_mass, center_of_sfr


def gas_flow(gas):
    gas_momentum = (gas["Mass_g"] * gas["PeculiarVelocities"].T).T
    directions = (
        gas["RelativePosition_cm"].T
        / np.linalg.norm(gas["RelativePosition_cm"], axis=1)
    ).T
    directional_momentum = np.multiply(gas_momentum, directions).sum(axis=1)
    flow = directional_momentum.sum() / gas_momentum.sum()
    return flow


def velocities(gas, angular_momentum):
    gas_used = gas["StarFormationRate"] > 0
    gas_sfr = gas["StarFormationRate"][gas_used]
    gas_velocities = gas["PeculiarVelocities"][gas_used]
    gas_masses = gas["Mass_g"][gas_used]
    gas_coord = gas["RelativePosition_cm"][gas_used]

    tot_vel = np.average(gas_velocities, axis=0, weights=gas_masses)
    rel_gas_velocities = gas_velocities - tot_vel

    _, sfr_weighted_std = get_weighted_mean_std(
        rel_gas_velocities, weights=gas_sfr
    )
    _, mass_weighted_std = get_weighted_mean_std(
        rel_gas_velocities, weights=gas_masses
    )

    abs_sigma_sfr = np.linalg.norm(sfr_weighted_std) / np.sqrt(3)
    abs_sigma_mass = np.linalg.norm(mass_weighted_std) / np.sqrt(3)

    rot_dirs = np.cross(gas_coord, angular_momentum)
    rot_dirs_norm = (rot_dirs.T / np.linalg.norm(rot_dirs, axis=1)).T

    directional_vel = np.abs(
        np.multiply(rel_gas_velocities, rot_dirs_norm).sum(axis=1)
    )
    v_max = np.sort(directional_vel)[-2]
    return abs_sigma_sfr, abs_sigma_mass, v_max


def get_weighted_mean_std(values, weights):
    mean = np.average(values, axis=0, weights=weights)
    var = np.average((values - mean) ** 2, axis=0, weights=weights)
    return mean, np.sqrt(var)


def get_prop_dict(df, idx, snap_num):
    sim, sim_path = get_sim()
    z = get_redshift(sim, snap_num)
    gas = il.snapshot.loadSubhalo(sim_path, snap_num, idx, "gas")
    get_relative_coord(gas, df, idx)
    sphere_gas = select_sphere_gas(gas, df, idx, z, is_relative=True)
    get_pos_cm(sphere_gas, z)
    get_pec_vel(sphere_gas, z)
    mass_to_g(sphere_gas)

    prop_dict = {}
    prop_dict["idx"] = idx
    prop_dict["flow"] = gas_flow(sphere_gas)
    ang_mom = angular_momentum(sphere_gas)
    prop_dict["ang_momentum"] = np.linalg.norm(ang_mom)
    cent_mass = center_of_mass(sphere_gas)
    center_of_sfr_mass_value, center_of_sfr_sfr_value = center_of_sfr_mass(
        sphere_gas
    )
    sfr_mass_to_center_mass = np.linalg.norm(
        cent_mass - center_of_sfr_mass_value
    )
    sfr_sfr_to_center_mass = np.linalg.norm(
        cent_mass - center_of_sfr_sfr_value
    )
    prop_dict["sfr_mass_to_center_mass"] = sfr_mass_to_center_mass
    prop_dict["sfr_sfr_to_center_mass"] = sfr_sfr_to_center_mass
    abs_sigma_sfr, abs_sigma_mass, v_max = velocities(sphere_gas, ang_mom)
    prop_dict["abs_sigma_sfr"] = abs_sigma_sfr
    prop_dict["abs_sigma_mass"] = abs_sigma_mass
    prop_dict["v_max"] = v_max
    return prop_dict


def expand_prop_values(df_prefix, snap_num, base_path):
    snap = get_snap(snap_num)
    dir_path = os.path.join(base_path, snap)
    df_name = f"{df_prefix}_{str(snap_num)}.pickle"
    df_path = os.path.join(dir_path, df_name)
    df = pd.read_pickle(df_path)
    keys = [
        "idx",
        "flow",
        "ang_momentum",
        "sfr_mass_to_center_mass",
        "sfr_sfr_to_center_mass",
        "abs_sigma_sfr",
        "abs_sigma_mass",
        "v_max",
    ]
    full_dict = {}
    for key in keys:
        full_dict[key] = np.empty(len(df))
        full_dict[key][:] = np.nan

    for i, galaxy in enumerate(df.index):
        prop_dict = get_prop_dict(df, galaxy, snap_num)
        for key in keys:
            full_dict[key][i] = prop_dict[key]
    for key in keys:
        df[key] = full_dict[key]
    df.to_pickle(df_path)
    return


def update_prop_values_range(
    snap_min,
    snap_max,
    df_prefix=config["df_name"],
    base_path=config["base_path"],
):
    for i in range(snap_min, snap_max):
        print(f"Working on snapshot {i}")
        expand_prop_values(df_prefix, i, base_path)
    return


if __name__ == "__main__":
    update_prop_values_range(2, 3)
