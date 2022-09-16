import pandas as pd
import numpy as np
from astropy import constants
from astropy import units as u
from astropy.constants import m_p
import os

# Solar metallicit
Z_solar = 0.0134


def get_column_height_dens(maps, grid_cell_size):
    kg_to_g = 1000
    m_p_g = m_p.value * kg_to_g

    grid_cell_area = grid_cell_size**2

    maps["Sigma_SFR"] = maps["SFR"] / grid_cell_area
    maps["Sigma_gas"] = maps["M_gas"] / grid_cell_area
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
    maps["Ion_flux"] = (
        maps["Sigma_SFR"] * sigma_sfr_to_ion_flux / cm_to_kpc**2
    )
    maps["Bol_flux"] = (
        maps["Sigma_SFR"] * bolometric_correction / cm_to_kpc**2
    )
    return


# This assumes a constant ratio between
# metallicity and dust
def normalized_dust(maps):
    maps["Dust_norm"] = maps["Metallicity"] / Z_solar
    return


def gas_fraction(maps):
    maps["f_g"] = maps["M_gas"] / (maps["M_gas"] + maps["M_star"])
    return


def N_crit(maps):
    one_over_dust_cross_converstion = 4.3e20
    maps["N_d"] = one_over_dust_cross_converstion / maps["Dust_norm"]
    return


def dust_crosssection_per_H(maps):
    conversion_factor = 4.8e-22
    maps["sigma_d_H"] = conversion_factor * maps["Dust_norm"]
    return


def gravitational_pressure(maps):
    m3_per_kg_to_cm3_per_g = 1e3
    maps["p_g"] = (
        m3_per_kg_to_cm3_per_g
        * np.pi
        / 2
        * constants.G.value
        * maps["Sigma_gas"] ** 2
        / maps["f_g"]
    )
    return


def particle_dens(maps, grid_cell_size, scale_height):
    grid_cell_volume = grid_cell_size**2 * scale_height
    mean_molecular_mass = 1
    kg_to_g = 1e3
    maps["n_gas_r"] = (
        maps["M_gas_r"]
        / grid_cell_volume
        / (constants.m_p.value * kg_to_g)
        / mean_molecular_mass
    )
    return


def photon_to_gas(maps):
    m_to_cm = 100
    maps["U"] = maps["Ion_flux"] / maps["n_gas"] / constants.c.value / m_to_cm
    return


def column_dens_stroemgren(maps):
    case_B_param = 2.6e-13
    meter_to_cm = 100
    maps["Column_dens_stroemgren"] = (
        maps["U"] * constants.c.value * meter_to_cm / case_B_param
    )
    return


def U1(maps):
    maps["U1"] = maps["f_g"] ** 3 * maps["U"]
    return


def tau_dust(maps):
    maps["tau_d"] = maps["sigma_d_H"] * maps["Column_dens"]
    return


def radiation_pressure(maps):
    meter_to_cm = 100
    maps["p_r"] = (
        (1 - np.exp(-maps["tau_d"]))
        * maps["Bol_flux"]
        / (constants.c.value * meter_to_cm)
    )
    return


def outflow_vel(maps):
    return (
        2
        * maps["Column_height"]
        * (maps["p_r"] - maps["p_g"])
        / (maps["Sigma_gas"])
    ) ** 0.5


def add_outflow_vel(maps):
    maps["v_inf"] = np.where(maps["p_r"] > maps["p_g"], outflow_vel(maps), 0)
    return


def column_dens_ratio(maps):
    maps["N_ratio"] = maps["Column_dens"] / maps["N_d"]
    return


def critical_gas_fraction(maps):
    return 6 * (
        maps["Dust_norm"]
        * maps["U1"]
        * ((1 - maps["N_ratio"]) / maps["N_ratio"])
    ) ** (1 / 3)


def add_critical_gas_fraction(maps):
    maps["f_g_crit"] = np.where(
        maps["N_ratio"] < 1, critical_gas_fraction(maps), 0
    )
    return


def evac_gas_frac(maps):
    sec_per_Myr = (1 * u.Myr).to(u.s).value
    t_OB = 2 * sec_per_Myr
    maps["w"] = 0.5 * maps["v_inf"] * t_OB / maps["Column_height"]

    # Correction in case of full evaculation
    maps["w"] = np.where(maps["w"] > 1, 1, maps["w"])
    return


def reduced_column_den(maps):
    maps["N_red"] = (1 - maps["w"]) * maps["Column_dens"]
    return


def escape_fraction(maps):
    return np.exp(
        -maps["N_red"] * (1 / maps["Column_dens_stroemgren"] + 1 / maps["N_d"])
    )


def f_esc(maps):
    maps["f_esc"] = np.where(
        maps["f_g"] < maps["f_g_crit"], escape_fraction(maps), 0
    )
    return


def update_to_fesc(maps):
    get_column_height_dens(maps)
    get_lum_from_sfr(maps)
    normalized_dust(maps)
    gas_fraction(maps)
    N_crit(maps)
    normalized_dust(maps)
    dust_crosssection_per_H(maps)
    gravitational_pressure(maps)
    particle_dens(maps)
    photon_to_gas(maps)
    column_dens_stroemgren(maps)
    U1(maps)
    tau_dust(maps)
    radiation_pressure(maps)
    add_outflow_vel(maps)
    column_dens_ratio(maps)
    add_critical_gas_fraction(maps)
    evac_gas_frac(maps)
    reduced_column_den(maps)
    f_esc(maps)
    return


if __name__ == "__main__":
    level_2_name = "2_df.pickle"
    snap_num = 13
    level_3_name = "full_df.pickle"
    base = "/ptmp/mpa/ivkos/semianalytic_fesc/sn013"

    df_path = os.path.join(base, level_2_name)
    df = pd.read_pickle(df_path)

    update_to_fesc(df)

    save_path = os.path.join(base, level_3_name)
    df.to_pickle(save_path)
