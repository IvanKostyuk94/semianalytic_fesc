import numpy as np
import pandas as pd
from pyTNG.cosmology import TNGcosmo
import astropy.units as u

h = TNGcosmo.h


# Function which removes columns which are represented twice such as
# star and gas mass which we calculate in get_volume to ensure that
# everything is consistent and converts the stellar half mass rad to cm
def reformat_df(df, z):
    columns_to_drop = ['M_gas_r', 'M_gas_2r', 'M_star_r', 'M_star_2r']
    df.drop(labels=columns_to_drop, inplace=True)

    dist_to_cm = (1*u.kpc).to(u.cm)/h/(1+z)
    df['r'] = df['r']*dist_to_cm
    df['Redshift'] = z

    to_rename = {'Masses_r': 'M_gas_r', 'Masses_2r': 'M_gas_2r',
                 'Volumes_r': 'V_gas_r', 'Volumes_2r': 'V_gas_r',
                 'Star_masses_r': 'M_star_r', 'Star_masses_2r': 'M_star_2r'}
    df.rename(to_rename, inplace=True)
    return


def get_column_height_dens(df, z):
    areas_r = df['r']**2*np.pi
    areas_2r = (2*df['r'])**2*np.pi
    df['Column_height_r'] = df['Volumes_r']/areas_r
    df['Column_height_2r'] = df['Volumes_2r']/areas_2r
    df['Column_dens_r'] = df['Masses_r']/df['Column_height_r']
    df['Column_dens_2r'] = df['Masses_2r']/df['Column_height_2r']

    df['Sigma_SFR_r'] = df['SFR_r']/areas_r
    df['Sigma_SFR_2r'] = df['SFR_2r']/areas_2r
    df['Sigma_gas_r'] = df['M_gas_r']/areas_r
    df['Sigma_gas_2r'] = df['M_gas_2r']/areas_2r
    df['Sigma_star_r'] = df['M_star_r']/areas_r
    df['Sigma_star_2r'] = df['M_star_2r']/areas_2r
    return
