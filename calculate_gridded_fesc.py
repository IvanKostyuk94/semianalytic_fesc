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


# Stopped here!
def add_outflow_vel(maps):
    maps["v_inf_r"] = maps.apply(
        lambda x: 0 if x["p_r_r"] < x["p_g_r"] else outflow_vel(x, mode="r"),
        axis=1,
    )
    return


def column_dens_ratio(df):
    df["N_ratio_r"] = df["Column_dens_r"] / df["N_d_r"]
    df["N_ratio_2r"] = df["Column_dens_2r"] / df["N_d_2r"]
    df["N_ratio_sf_r"] = df["Column_dens_r"] / df["N_d_sf_r"]
    df["N_ratio_sf_2r"] = df["Column_dens_2r"] / df["N_d_sf_2r"]
    return


def critical_gas_fraction(element, mode="r"):
    if mode == "r":
        return 6 * (
            element["Dust_norm_r"]
            * element["U1_r"]
            * ((1 - element["N_ratio_r"]) / element["N_ratio_r"])
        ) ** (1 / 3)
    elif mode == "2r":
        return 6 * (
            element["Dust_norm_2r"]
            * element["U1_2r"]
            * ((1 - element["N_ratio_2r"]) / element["N_ratio_2r"])
        ) ** (1 / 3)
    elif mode == "sf_r":
        return 6 * (
            element["Dust_norm_sf_r"]
            * element["U1_sf_r"]
            * ((1 - element["N_ratio_sf_r"]) / element["N_ratio_sf_r"])
        ) ** (1 / 3)
    elif mode == "sf_2r":
        return 6 * (
            element["Dust_norm_sf_2r"]
            * element["U1_sf_2r"]
            * ((1 - element["N_ratio_sf_2r"]) / element["N_ratio_sf_2r"])
        ) ** (1 / 3)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented yet")


def add_critical_gas_fraction(df):
    df["f_g_crit_r"] = df.apply(
        lambda x: 0
        if x["N_ratio_r"] > 1
        else critical_gas_fraction(x, mode="r"),
        axis=1,
    )
    df["f_g_crit_2r"] = df.apply(
        lambda x: 0
        if x["N_ratio_2r"] > 1
        else critical_gas_fraction(x, mode="2r"),
        axis=1,
    )
    df["f_g_crit_sf_r"] = df.apply(
        lambda x: 0
        if x["N_ratio_sf_r"] > 1
        else critical_gas_fraction(x, mode="sf_r"),
        axis=1,
    )
    df["f_g_crit_sf_2r"] = df.apply(
        lambda x: 0
        if x["N_ratio_sf_2r"] > 1
        else critical_gas_fraction(x, mode="sf_2r"),
        axis=1,
    )
    return


def evac_gas_frac(df):
    sec_per_Myr = (1 * u.Myr).to(u.s).value
    t_OB = 2 * sec_per_Myr
    df["w_r"] = 0.5 * df["v_inf_r"] * t_OB / df["Column_height_r"]
    df["w_2r"] = 0.5 * df["v_inf_2r"] * t_OB / df["Column_height_2r"]
    df["w_sf_r"] = 0.5 * df["v_inf_sf_r"] * t_OB / df["Column_height_r"]
    df["w_sf_2r"] = 0.5 * df["v_inf_sf_2r"] * t_OB / df["Column_height_2r"]

    # Correction in case of full evaculation
    df["w_r"] = df.apply(lambda x: x["w_r"] if x["w_r"] < 1 else 1, axis=1)
    df["w_2r"] = df.apply(lambda x: x["w_2r"] if x["w_2r"] < 1 else 1, axis=1)
    df["w_sf_r"] = df.apply(
        lambda x: x["w_sf_r"] if x["w_sf_r"] < 1 else 1, axis=1
    )
    df["w_sf_2r"] = df.apply(
        lambda x: x["w_sf_2r"] if x["w_sf_2r"] < 1 else 1, axis=1
    )
    return


def reduced_column_den(df):
    df["N_red_r"] = (1 - df["w_r"]) * df["Column_dens_r"]
    df["N_red_2r"] = (1 - df["w_2r"]) * df["Column_dens_2r"]
    df["N_red_sf_r"] = (1 - df["w_sf_r"]) * df["Column_dens_r"]
    df["N_red_sf_2r"] = (1 - df["w_sf_2r"]) * df["Column_dens_2r"]
    return


def escape_fraction(element, mode):
    if mode == "r":
        return np.exp(
            -element["N_red_r"]
            * (1 / element["Column_dens_stroemgren_r"] + 1 / element["N_d_r"])
        )
    elif mode == "2r":
        return np.exp(
            -element["N_red_2r"]
            * (
                1 / element["Column_dens_stroemgren_2r"]
                + 1 / element["N_d_2r"]
            )
        )
    elif mode == "sf_r":
        return np.exp(
            -element["N_red_sf_r"]
            * (
                1 / element["Column_dens_stroemgren_sf_r"]
                + 1 / element["N_d_sf_r"]
            )
        )
    elif mode == "sf_2r":
        return np.exp(
            -element["N_red_sf_2r"]
            * (
                1 / element["Column_dens_stroemgren_sf_2r"]
                + 1 / element["N_d_sf_2r"]
            )
        )
    else:
        raise NotImplementedError(f"The mode {mode} is not implemented yet")


def f_esc(df):
    df["f_esc_r"] = df.apply(
        lambda x: 0
        if x["f_g_r"] > x["f_g_crit_r"]
        else escape_fraction(x, mode="r"),
        axis=1,
    )
    df["f_esc_2r"] = df.apply(
        lambda x: 0
        if x["f_g_2r"] > x["f_g_crit_2r"]
        else escape_fraction(x, mode="2r"),
        axis=1,
    )
    df["f_esc_sf_r"] = df.apply(
        lambda x: 0
        if x["f_g_sf_r"] > x["f_g_crit_sf_r"]
        else escape_fraction(x, mode="sf_r"),
        axis=1,
    )
    df["f_esc_sf_2r"] = df.apply(
        lambda x: 0
        if x["f_g_sf_2r"] > x["f_g_crit_sf_2r"]
        else escape_fraction(x, mode="sf_2r"),
        axis=1,
    )

    df["f_esc_r"] = df.apply(
        lambda x: 1 if np.isnan(x["Column_dens_r"]) else x["f_esc_r"], axis=1
    )
    df["f_esc_2r"] = df.apply(
        lambda x: 1 if np.isnan(x["Column_dens_2r"]) else x["f_esc_2r"], axis=1
    )
    df["f_esc_sf_r"] = df.apply(
        lambda x: 1 if np.isnan(x["Column_dens_r"]) else x["f_esc_sf_r"],
        axis=1,
    )
    df["f_esc_sf_2r"] = df.apply(
        lambda x: 1 if np.isnan(x["Column_dens_2r"]) else x["f_esc_sf_2r"],
        axis=1,
    )
    return


def update_to_fesc(df):

    get_column_height_dens(df)
    get_lum_from_sfr(df)
    sf_ionizing_flux(df)
    sf_bol_flux(df)
    normalized_dust(df)
    gas_fraction(df)
    N_crit(df)
    normalized_dust(df)
    dust_crosssection_per_H(df)
    gravitational_pressure(df)
    particle_dens(df)
    photon_to_gas(df)
    column_dens_stroemgren(df)
    U1(df)
    tau_dust(df)
    radiation_pressure(df)
    add_outflow_vel(df)
    column_dens_ratio(df)
    add_critical_gas_fraction(df)
    evac_gas_frac(df)
    reduced_column_den(df)
    f_esc(df)
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
