import pandas as pd
import numpy as np
from astropy import constants
from astropy import units as u
import os
import h5py
from utils import get_snap

# Solar metallicit
Z_solar = 0.0134

np.seterr(divide="ignore", invalid="ignore")


def get_surface_dens(maps, grid_cell_size):
    grid_cell_area = grid_cell_size**2

    if "Sigma_SFR" in maps.keys():
        del maps["Sigma_SFR"]
    maps["Sigma_SFR"] = maps["SFR"] / grid_cell_area

    if "Sigma_gas" in maps.keys():
        del maps["Sigma_gas"]
    maps["Sigma_gas"] = maps["M_gas"] / grid_cell_area

    if "Sigma_star" in maps.keys():
        del maps["Sigma_star"]
    maps["Sigma_star"] = maps["M_star"] / grid_cell_area
    return


# Units of the bolometric flux are erg*cm-2*s-1
# Units of ionizing flux are cm-2
# Sigma_sfr needs to be converted to M_sum*yr-1*kpc-2
def get_lum_from_sfr(maps):
    sigma_sfr_to_ion_flux = 3e10
    mean_phot_e_ion_spec = 20.4
    bolometric_correction = 5
    cm_to_kpc = (1 * u.cm).to(u.kpc).value

    if "Ion_flux" in maps.keys():
        del maps["Ion_flux"]
    maps["Ion_flux"] = (
        np.array(maps["Sigma_SFR"]) * sigma_sfr_to_ion_flux / cm_to_kpc**2
    )

    if "Bol_flux" in maps.keys():
        del maps["Bol_flux"]
    maps["Bol_flux"] = (
        np.array(maps["Sigma_SFR"]) * bolometric_correction / cm_to_kpc**2
    )
    return


# This assumes a constant ratio between
# metallicity and dust
def normalized_dust(maps):
    if "Dust_norm" in maps.keys():
        del maps["Dust_norm"]
    maps["Dust_norm"] = np.array(maps["Metallicity"]) / Z_solar
    return


def gas_fraction(maps):
    if "f_g" in maps.keys():
        del maps["f_g"]
    maps["f_g"] = np.array(maps["M_gas"]) / (
        np.array(maps["M_gas"]) + np.array(maps["M_star"])
    )
    return


def N_crit(maps):
    one_over_dust_cross_converstion = 4.3e20

    if "N_d" in maps.keys():
        del maps["N_d"]
    maps["N_d"] = one_over_dust_cross_converstion / np.array(maps["Dust_norm"])
    return


def dust_crosssection_per_H(maps):
    conversion_factor = 4.8e-22

    if "sigma_d_H" in maps.keys():
        del maps["sigma_d_H"]
    maps["sigma_d_H"] = conversion_factor * np.array(maps["Dust_norm"])
    return


def gravitational_pressure(maps):
    m3_per_kg_to_cm3_per_g = 1e3

    if "p_g" in maps.keys():
        del maps["p_g"]
    maps["p_g"] = (
        m3_per_kg_to_cm3_per_g
        * np.pi
        / 2
        * constants.G.value
        * np.array(maps["Sigma_gas"]) ** 2
        / np.array(maps["f_g"])
    )
    return


def column_dens(maps, grid_cell_size):
    grid_cell_area = grid_cell_size**2
    mean_molecular_mass = 1
    kg_to_g = 1e3

    if "Column_dens" in maps.keys():
        del maps["Column_dens"]
    maps["Column_dens"] = (
        np.array(maps["M_gas"])
        / 2
        / grid_cell_area
        / (constants.m_p.value * kg_to_g)
        / mean_molecular_mass
    )
    grid_cell_size


def particle_dens(maps, scale_height):
    if "n_gas" in maps.keys():
        del maps["n_gas"]
    maps["n_gas"] = np.array(maps["Column_dens"]) / scale_height
    return


def photon_to_gas(maps):
    m_to_cm = 100

    if "U" in maps.keys():
        del maps["U"]
    maps["U"] = (
        np.array(maps["Ion_flux"])
        / np.array(maps["n_gas"])
        / constants.c.value
        / m_to_cm
    )
    return


def column_dens_stroemgren(maps):
    case_B_param = 2.6e-13
    meter_to_cm = 100

    if "Column_dens_stroemgren" in maps.keys():
        del maps["Column_dens_stroemgren"]
    maps["Column_dens_stroemgren"] = (
        np.array(maps["U"]) * constants.c.value * meter_to_cm / case_B_param
    )
    return


def U1(maps):
    if "U1" in maps.keys():
        del maps["U1"]
    maps["U1"] = np.array(maps["f_g"]) ** 3 * np.array(maps["U"])
    return


def tau_dust(maps):
    if "tau_d" in maps.keys():
        del maps["tau_d"]
    maps["tau_d"] = np.array(maps["sigma_d_H"]) * np.array(maps["Column_dens"])
    return


def radiation_pressure(maps):
    meter_to_cm = 100

    if "p_r" in maps.keys():
        del maps["p_r"]
    maps["p_r"] = (
        (1 - np.exp(-np.array(maps["tau_d"])))
        * np.array(maps["Bol_flux"])
        / (constants.c.value * meter_to_cm)
    )
    return


def outflow_vel(maps, scale_height):
    return (
        2
        * scale_height
        * (np.array(maps["p_r"]) - np.array(maps["p_g"]))
        / (np.array(maps["Sigma_gas"]))
    ) ** 0.5


def add_outflow_vel(maps, scale_height):
    if "v_inf" in maps.keys():
        del maps["v_inf"]
    maps["v_inf"] = np.where(
        np.array(maps["p_r"]) > np.array(maps["p_g"]),
        outflow_vel(maps, scale_height),
        0,
    )
    return


def column_dens_ratio(maps):
    if "N_ratio" in maps.keys():
        del maps["N_ratio"]
    maps["N_ratio"] = np.array(maps["Column_dens"]) / np.array(maps["N_d"])
    return


def critical_gas_fraction(maps):
    return 6 * (
        np.array(maps["Dust_norm"])
        * np.array(maps["U1"])
        * ((1 - np.array(maps["N_ratio"])) / np.array(maps["N_ratio"]))
    ) ** (1 / 3)


def add_critical_gas_fraction(maps):
    if "f_g_crit" in maps.keys():
        del maps["f_g_crit"]
    maps["f_g_crit"] = np.where(
        np.array(maps["N_ratio"]) < 1, critical_gas_fraction(maps), 0
    )
    return


def evac_gas_frac(maps, scale_height):
    sec_per_Myr = (1 * u.Myr).to(u.s).value
    t_OB = 2 * sec_per_Myr
    w_array = 0.5 * np.array(maps["v_inf"]) * t_OB / scale_height

    # Correction in case of full evaculation
    if "w" in maps.keys():
        del maps["w"]
    maps["w"] = np.where(np.array(w_array) > 1, 1, np.array(w_array))
    return


def reduced_column_den(maps):
    if "N_red" in maps.keys():
        del maps["N_red"]
    maps["N_red"] = (1 - np.array(maps["w"])) * np.array(maps["Column_dens"])
    return


def escape_fraction(maps):
    return np.exp(
        -np.array(maps["N_red"])
        * (
            1 / np.array(maps["Column_dens_stroemgren"])
            + 1 / np.array(maps["N_d"])
        )
    )


def f_esc(maps):
    if "f_esc" in maps.keys():
        del maps["f_esc"]
    maps["f_esc"] = np.where(
        np.array(maps["f_g"]) < np.array(maps["f_g_crit"]),
        escape_fraction(maps),
        0,
    )
    return


def update_to_fesc(maps, grid_cell_size, scale_height):
    get_surface_dens(maps, grid_cell_size)
    get_lum_from_sfr(maps)
    normalized_dust(maps)
    gas_fraction(maps)
    N_crit(maps)
    normalized_dust(maps)
    dust_crosssection_per_H(maps)
    gravitational_pressure(maps)
    column_dens(maps, grid_cell_size)
    particle_dens(maps, scale_height)
    photon_to_gas(maps)
    column_dens_stroemgren(maps)
    U1(maps)
    tau_dust(maps)
    radiation_pressure(maps)
    add_outflow_vel(maps, scale_height)
    column_dens_ratio(maps)
    add_critical_gas_fraction(maps)
    evac_gas_frac(maps, scale_height)
    reduced_column_den(maps)
    f_esc(maps)
    return


def update_maps(
    snap_num,
    grid_size,
    hdf_name,
    df_name,
    base="/ptmp/mpa/ivkos/semianalytic_fesc",
):
    snap = get_snap(snap_num)
    hdf_filename = hdf_name + ".hdf5"
    hdf_path = os.path.join(base, snap, hdf_filename)

    df_filename = df_name + ".pickle"
    origin_path = os.path.join(base, snap, df_filename)

    hdf_file = h5py.File(hdf_path, "a")
    group = hdf_file[str(grid_size)]
    df = pd.read_pickle(origin_path)

    for idx in df.index:
        galaxy_name = str(idx)
        galaxy_group = group[galaxy_name]
        # For testing only
        grid_size_column = "Grid_cell_size_" + str(grid_size)
        # grid_size_column = "Grid_cell_size"
        grid_cell_size = df.loc[idx, grid_size_column]
        scale_height = df.loc[idx, "Column_height"]
        update_to_fesc(
            maps=galaxy_group,
            grid_cell_size=grid_cell_size,
            scale_height=scale_height,
        )
    hdf_file.close()
    return
