import pandas as pd
import numpy as np
from astropy import constants
from astropy import units as u

# Solar metallicit
Z_solar = 0.0134


def normalized_dust(df):
    df['Dust_norm_r'] = df['Z_r']/Z_solar
    df['Dust_norm_2r'] = df['Z_2r']/Z_solar
    return


def gas_fraction(df):
    df['f_g_r'] = df['M_gas_r']/(df['M_gas_r']+df['M_star_r'])
    df['f_g_2r'] = df['M_gas_2r']/(df['M_gas_2r']+df['M_star_2r'])
    return


def N_crit(df):
    one_over_dust_cross_converstion = 4.3e20
    df['N_d_r'] = one_over_dust_cross_converstion/df['Dust_norm_r']
    df['N_d_2r'] = one_over_dust_cross_converstion/df['Dust_norm_2r']
    return


def dust_crosssection_per_H(df):
    conversion_factor = 4.8e-22
    df['sigma_d_H_r'] = conversion_factor*df['Dust_norm_r']
    df['sigma_d_H_2r'] = conversion_factor*df['Dust_norm_2r']
    return


def gravitational_pressure(df):
    m3_by_kg_to_cm3_by_g = 1e3
    df['p_g_r'] = m3_by_kg_to_cm3_by_g*np.pi/2*constants.G.value*df['Sigma_gas_r']**2/df['f_g_r']
    df['p_g_2r'] = m3_by_kg_to_cm3_by_g*np.pi/2*constants.G.value*df['Sigma_gas_2r']**2/df['f_g_2r']
    return


def particle_dens(df):
    mean_molecular_mass = 1
    kg_to_g = 1e3
    df['n_gas_r'] = df['M_gas_r']/df['V_gas_r']/(constants.m_p.value*kg_to_g)/mean_molecular_mass
    df['n_gas_2r'] = df['M_gas_2r']/df['V_gas_2r']/(constants.m_p.value*kg_to_g)/mean_molecular_mass
    return


def photon_to_gas(df):
    df['U_r'] = df['Ion_flux_r']/df['n_gas_r']/constants.c.value
    df['U_2r'] = df['Ion_flux_2r']/df['n_gas_2r']/constants.c.value
    return


def column_dens_stroemgren(df):
    case_B_param = 2.6e-13
    meter_to_cm = 100
    df['Column_dens_stroemgren_r'] = df['f_g_r']*constants.c.value*meter_to_cm/case_B_param
    df['Column_dens_stroemgren_2r'] = df['f_g_2r']*constants.c.value*meter_to_cm/case_B_param
    return


def U1(df):
    df['U1_r'] = df['f_g_r']**3*df['U_r']
    df['U1_2r'] = df['f_g_2r']**3*df['U_2r']
    return


def tau_dust(df):
    df['tau_d_r'] = df['sigma_d_H_r']*df['Column_dens_r']
    df['tau_d_2r'] = df['sigma_d_H_2r']*df['Column_dens_2r']
    return


def radiation_pressure(df):
    meter_to_cm = 100
    df['p_r_r'] = (1-np.exp(-df['tau_d_r']))*df['Bol_flux_r']/(constants.c.value*meter_to_cm)
    df['p_r_2r'] = (1-np.exp(-df['tau_d_2r']))*df['Bol_flux_2r']/(constants.c.value*meter_to_cm)
    return


def outflow_vel(element, mode='r'):
    if mode == 'r':
        return (2*element['Column_height_r']
                * (element['p_r_r']-element['p_g_r'])
                / (element['Sigma_gas_r']))**0.5
    elif mode == '2r':
        return (2*element['Column_height_2r']
                * (element['p_r_2r']-element['p_g_2r'])
                / (element['Sigma_gas_2r']))**0.5
    else:
        raise NotImplementedError(f'Mode {mode} is not implemented')


def add_outflow_vel(df):
    df['v_inf_r'] = df.apply(lambda x: 0 if x['p_r_r'] < x['p_g_r']
                             else outflow_vel(x, mode='r'), axis=1)
    df['v_inf_2r'] = df.apply(lambda x: 0 if x['p_r_2r'] < x['p_g_2r']
                              else outflow_vel(x, mode='2r'), axis=1)
    return


def column_dens_ratio(df):
    df['N_ratio_r'] = df['Column_dens_r']/df['N_d_r']
    df['N_ratio_2r'] = df['Column_dens_2r']/df['N_d_2r']
    return


def critical_gas_fraction(element, mode='r'):
    if mode == 'r':
        return 6*(element['Dust_norm_r'] * element['U1_r']
               * ((1-element['N_ratio_r']) / element['N_ratio_r']))**(1/3)
    elif mode == '2r':
        return 6*(element['Dust_norm_2r'] * element['U1_2r']
               * ((1-element['N_ratio_2r']) / element['N_ratio_2r']))**(1/3)
    else:
        raise NotImplementedError(f'Mode {mode} is not implemented yet')


def add_critical_gas_fraction(df):
    df['f_g_crit_r'] = df.apply(lambda x: 0 if x['N_ratio_r'] > 1
                                else critical_gas_fraction(x, mode='r'), axis=1)
    df['f_g_crit_2r'] = df.apply(lambda x: 0 if x['N_ratio_2r'] > 1
                                 else critical_gas_fraction(x, mode='2r'), axis=1)
    return


def evac_gas_frac(df):
    sec_per_Myr = (1*u.Myr).to(u.s).value
    t_OB = 2*sec_per_Myr
    df['w_r'] = 0.5*df['v_inf_r']*t_OB/df['Column_height_r']
    df['w_2r'] = 0.5*df['v_inf_2r']*t_OB/df['Column_height_2r']

    # Correction in case of full evaculation
    df['w_r'] = df.apply(lambda x: x['w_r'] if x['w_r'] < 1 else 1, axis=1)
    df['w_2r'] = df.apply(lambda x: x['w_2r'] if x['w_2r'] < 1 else 1, axis=1)
    return


def reduced_column_den(df):
    df['N_red_r'] = (1-df['w_r'])*df['Column_dens_r']
    df['N_red_2r'] = (1-df['w_2r'])*df['Column_dens_2r']
    return


def escape_fraction(element, mode):
    if mode == 'r':
        return np.exp(-element['N_red_r']
                      * (1/element['Column_dens_stroemgren_r']
                      + 1/element['N_d_r']))
    elif mode == '2r':
        return np.exp(-element['N_red_2r']
                      * (1/element['Column_dens_stroemgren_2r']
                      + 1/element['N_d_2r']))
    else:
        raise NotImplementedError(f'The mode {mode} is not implemented yet')


def f_esc(df):
    df['f_esc_r'] = df.apply(lambda x: 0 if x['f_g_r'] < x['f_g_crit_r']
                             else escape_fraction(x, mode='r'), axis=1)
    df['f_esc_2r'] = df.apply(lambda x: 0 if x['f_g_2r'] < x['f_g_crit_2r']
                              else escape_fraction(x, mode='r'), axis=1)
    df['f_esc_r'] = df.apply(lambda x: 1 if np.isnan(x['Column_dens_r'])
                             else x['f_esc_r'], axis=1)
    df['f_esc_2r'] = df.apply(lambda x: 1 if np.isnan(x['Column_dens_2r'])
                             else x['f_esc_2r'], axis=1)
    return


def update_to_fesc(df):
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
