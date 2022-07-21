import pandas as pd
import numpy as np
from astropy import constants

# Solar 
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
    df['p_g_r'] = np.pi/2*constants.G.value*df['Sigma_gas_r']**2/df['f_g_r']
    df['p_g_2r'] = np.pi/2*constants.G.value*df['Sigma_gas_2r']**2/df['f_g_2r']
    return


def particle_dens(df):
    mean_molecular_mass = 1
    df['n_gas_r'] = df['M_gas_r']/df['V_r']/constants.m_p.value/mean_molecular_mass
    df['n_gas_2r'] = df['M_gas_2r']/df['V_2r']/constants.m_p.value/mean_molecular_mass
    return


def photon_to_gas(df):
    df['U_r'] = df['Ion_flux_r']/df['n_gas_r']/constants.c.value
    df['U_2r'] = df['Ion_flux_2r']/df['n_gas_2r']/constants.c.value
    return


def column_dens_stroemgren(df):
    case_B_param = 2.6e-13
    df['Column_dens_stroemgren_r'] = df['f_g_r']*constants.c.value/case_B_param
    df['Column_dens_stroemgren_2r'] = df['f_g_2r']*constants.c.value/case_B_param
    return


def U_1(df):
    df['U_1_r'] = df['f_g_r']**3*df['U_r']
    df['U_1_2r'] = df['f_g_2r']**3*df['U_2r']
    return