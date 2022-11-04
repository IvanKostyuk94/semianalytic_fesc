import pandas as pd
import numpy as np
from astropy import constants
from astropy import units as u
from astropy.constants import m_p
import os

# Solar metallicit
Z_solar = 0.0134


def get_column_height_dens(df):
    area = (4 * df["r"]) ** 2

    kg_to_g = 1000
    m_p_g = m_p.value * kg_to_g
    df["Column_dens"] = df["M_gas"] / area / m_p_g / 2

    df["Sigma_SFR"] = df["SFR"] / area
    df["Sigma_gas"] = df["M_gas"] / area
    df["Sigma_star"] = df["M_star"] / area
    return


# Units of the bolometric flux are erg*cm-2*s-1
# Units of ionizing flux are cm-2
# Sigma_sfr needs to be converted to M_sum*yr-1*kpc-2
def get_lum_from_sfr(df):
    sigma_sfr_to_ion_flux = 3e10
    mean_phot_e_ion_spec = 20.4
    bolometric_correction = 5
    cm_to_kpc = (1 * u.cm).to(u.kpc).value
    df["Ion_flux"] = (
        df["Sigma_SFR"] * sigma_sfr_to_ion_flux / cm_to_kpc**2 / 2
    )
    df["Bol_flux"] = (
        df["Sigma_SFR"] * bolometric_correction / cm_to_kpc**2 / 2
    )
    return


def normalized_dust(df):
    df["Dust_norm"] = df["Z"] / Z_solar
    return


def gas_fraction(df):
    df["f_g"] = df["M_gas"] / (df["M_gas"] + df["M_star"])
    return


def N_crit(df):
    one_over_dust_cross_converstion = 4.3e20
    df["N_d"] = one_over_dust_cross_converstion / df["Dust_norm"]
    return


def dust_crosssection_per_H(df):
    conversion_factor = 4.8e-22
    df["sigma_d_H"] = conversion_factor * df["Dust_norm"]
    return


def gravitational_pressure(df):
    m3_by_kg_to_cm3_by_g = 1e3
    df["p_g"] = (
        m3_by_kg_to_cm3_by_g
        * np.pi
        / 2
        * constants.G.value
        * df["Sigma_gas"] ** 2
        / df["f_g"]
    )
    return


def particle_dens(df):
    mean_molecular_mass = 1
    kg_to_g = 1e3
    df["n_gas"] = (
        df["M_gas"]
        / (4 * df["r"]) ** 2
        / df["Column_height"]
        / (constants.m_p.value * kg_to_g)
        / mean_molecular_mass
        / 2
    )
    return


def photon_to_gas(df):
    m_to_cm = 100
    df["U"] = df["Ion_flux"] / df["n_gas"] / constants.c.value / m_to_cm
    return


def column_dens_stroemgren(df):
    case_B_param = 2.6e-13
    meter_to_cm = 100
    df["Column_dens_stroemgren"] = (
        df["U"] * constants.c.value * meter_to_cm / case_B_param
    )
    return


def U1(df):
    df["U1"] = df["f_g"] ** 3 * df["U"]
    return


def tau_dust(df):
    df["tau_d"] = df["sigma_d_H"] * df["Column_dens"]
    return


def radiation_pressure(df):
    meter_to_cm = 100
    df["p_r"] = (
        (1 - np.exp(-df["tau_d"]))
        * df["Bol_flux"]
        / (constants.c.value * meter_to_cm)
    )
    return


def outflow_vel(element):
    return (
        2
        * element["Column_height"]
        * (element["p_r"] - element["p_g"])
        / (element["Sigma_gas"])
    ) ** 0.5


def add_outflow_vel(df):
    df["v_inf"] = df.apply(
        lambda x: 0 if x["p_r"] < x["p_g"] else outflow_vel(x), axis=1
    )
    return


def column_dens_ratio(df):
    df["N_ratio"] = df["Column_dens"] / df["N_d"]
    return


def critical_gas_fraction(element):
    return 6 * (
        element["Dust_norm"]
        * element["U1"]
        * ((1 - element["N_ratio"]) / element["N_ratio"])
    ) ** (1 / 3)


def add_critical_gas_fraction(df):
    df["f_g_crit"] = df.apply(
        lambda x: 0 if x["N_ratio"] > 1 else critical_gas_fraction(x),
        axis=1,
    )
    return


def evac_gas_frac(df):
    sec_per_Myr = (1 * u.Myr).to(u.s).value
    t_OB = 2 * sec_per_Myr
    df["w"] = 0.5 * df["v_inf"] * t_OB / df["Column_height"]

    # Correction in case of full evaculation
    df["w"] = df.apply(lambda x: x["w"] if x["w"] < 1 else 1, axis=1)
    return


def reduced_column_den(df):
    df["N_red"] = (1 - df["w"]) * df["Column_dens"]
    return


def escape_fraction(element):
    return np.exp(
        -element["N_red"]
        * (1 / element["Column_dens_stroemgren"] + 1 / element["N_d"])
    )


def f_esc(df):
    df["f_esc"] = df.apply(
        lambda x: 0 if x["f_g"] > x["f_g_crit"] else escape_fraction(x),
        axis=1,
    )
    return


def update_to_fesc(df):
    get_column_height_dens(df)
    get_lum_from_sfr(df)
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
    level_2_name = "test_df_ad.pickle"
    snap_num = 13
    level_3_name = "test_df_ad.pickle"
    base = "/ptmp/mpa/ivkos/semianalytic_fesc/sn013/grid_tests"

    df_path = os.path.join(base, level_2_name)
    df = pd.read_pickle(df_path)
    df["M_gas"] = df["M_gas_0.1"]
    df["M_star"] = df["M_star_0.1"]
    df["SFR"] = df["SFR_0.1"]
    df["Z"] = df["Metallicity_0.1"]

    update_to_fesc(df)
    print("finished updating")
    save_path = os.path.join(base, level_3_name)
    df.to_pickle(save_path)
