import pandas as pd
import numpy as np
import os
import illustris_python as il
import astropy.units as u
from pyTNG.cosmology import TNGcosmo
from utils import get_sim
import pyTNG.utils as utils
from utils import get_particle_dist
from utils import get_redshift

h = TNGcosmo.h


def keepWindParticles(starAndWindParts):
    try:
        idces = starAndWindParts['GFM_StellarFormationTime'] < 0
        utils._keepPartsByIdx(starAndWindParts, idces)
    except KeyError as e:
        if str(e) == 'GFM_StellarFormationTime' and starAndWindParts['count'] == 0:
            pass
        else:
            raise
    return


def separate_wind_stars(starAndWindParts):
    try:
        idces_wind = starAndWindParts['GFM_StellarFormationTime'] < 0
        idces_stars = starAndWindParts['GFM_StellarFormationTime'] > 0

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
                if 'not subscriptable' in str(e):
                    pass
                else:
                    raise
            # for numpy scalars
            except IndexError as e:
                if 'invalid index to scalar variable' in str(e):
                    pass
                else:
                    raise
        if 'count' in starAndWindParts:
            wind['count'] = newcount_wind
            stars['count'] = newcount_stars
    except KeyError as e:
        if str(e) == 'GFM_StellarFormationTime' and starAndWindParts['count'] == 0:
            pass
        else:
            raise

    return wind, stars


def update_df(df, sim_path, snap_num, z):
    counter = 0
    volumes = []
    masses = []
    tot_star_masses = []

    volumes_half_rad = []
    masses_half_rad = []
    tot_star_masses_half_rad = []

    mass_to_g = (1*u.Msun).to(u.g).value*1e10/h
    volume_to_cm3 = ((1*u.kpc).to(u.cm).value/h/(1+z))**3
    dist_to_cm = (1*u.kpc).to(u.cm).value/h/(1+z)
    
    # r is converted here to cm to be consistend with the use of get_particle_dist
    df['r'] = df['r']*dist_to_cm
    for idx in df.index:
        gas = il.snapshot.loadSubhalo(sim_path, snap_num, idx, 'gas')
        wind_stars = il.snapshot.loadSubhalo(sim_path, snap_num, idx, 'stars')
        wind, stars = separate_wind_stars(wind_stars)

        get_particle_dist(gas, df, idx)
        get_particle_dist(wind, df, idx)

        gas_in_rad = gas['rel_dist'] < 1
        wind_in_rad = wind['rel_dist'] < 1
        if len(wind_in_rad) > 0:
            wind_masses_rad = wind['Masses'][wind_in_rad]
        else:
            wind_masses_rad = 0
        gas_masses = gas['Masses'][gas_in_rad]
        gas_densities = gas['Density'][gas_in_rad]
        gas_volumes = gas_masses/gas_densities
        volumes.append(np.sum(gas_volumes))
        masses.append(np.sum(gas_masses)+np.sum(wind_masses_rad))

        gas_in_half_rad = gas['rel_dist'] < 0.5
        wind_in_half_rad = wind['rel_dist'] < 0.5
        gas_masses_half_rad = gas['Masses'][gas_in_half_rad]
        if len(wind_in_half_rad) > 0:
            wind_masses_half_rad = wind['Masses'][wind_in_half_rad]
        else:
            wind_masses_half_rad = 0
        gas_densities_half_rad = gas['Density'][gas_in_half_rad]
        gas_volumes_half_rad = gas_masses_half_rad/gas_densities_half_rad
        volumes_half_rad.append(np.sum(gas_volumes_half_rad))
        masses_half_rad.append(np.sum(gas_masses_half_rad) +
                               np.sum(wind_masses_half_rad))

        get_particle_dist(stars, df, idx)
        stars_in_rad = stars['rel_dist'] < 1
        star_masses = np.sum(stars['Masses'][stars_in_rad])
        tot_star_masses.append(star_masses)

        stars_in_half_rad = stars['rel_dist'] < 0.5
        star_masses_half_rad = np.sum(stars['Masses'][stars_in_half_rad])
        tot_star_masses_half_rad.append(star_masses_half_rad)

        if counter % 100 == 0:
            print(f'{100*counter/len(df):.2f}% of all halos processed')
        counter += 1
    df['Masses_2r'] = np.array(masses)*mass_to_g
    df['Volumes_2r'] = np.array(volumes)*volume_to_cm3
    df['Star_masses_2r'] = np.array(tot_star_masses)*mass_to_g

    df['Masses_r'] = np.array(masses_half_rad)*mass_to_g
    df['Volumes_r'] = np.array(volumes_half_rad)*volume_to_cm3
    df['Star_masses_r'] = np.array(tot_star_masses_half_rad)*mass_to_g
    return


def get_star_mass(df, sim_path, snap_num, z):
    counter = 0
    tot_star_masses = []
    mass_to_g = (1*u.Msun).to(u.kg).value*1e10/h
    for idx in df.index:
        stars = il.snapshot.loadSubhalo(sim_path, snap_num, idx, 'stars')
        get_particle_dist(stars, df, idx)
        stars_in_rad = stars['rel_dist'] < 1
        star_masses = stars['Masses'][stars_in_rad]
        tot_star_masses.append(np.sum(star_masses))
        if counter % 100 == 0:
            print(f'{100*counter/len(df):.2f}% of all halos processed')
        counter += 1
    df[('star_masses', 0)] = np.array(tot_star_masses)*mass_to_g
    return


# Function which removes columns which are represented twice such as
# star and gas mass which we calculate in get_volume to ensure that
# everything is consistent and converts the stellar half mass rad to cm
def reformat_df(df, z):
    columns_to_drop = ['M_gas_r', 'M_gas_2r', 'M_star_r', 'M_star_2r']
    df.drop(labels=columns_to_drop, inplace=True, axis=1)

    df['Redshift'] = z

    to_rename = {'Masses_r': 'M_gas_r', 'Masses_2r': 'M_gas_2r',
                 'Volumes_r': 'V_r', 'Volumes_2r': 'V_2r',
                 'Star_masses_r': 'M_star_r', 'Star_masses_2r': 'M_star_2r'}
    df.rename(columns=to_rename, inplace=True)
    return


if __name__ == '__main__':
    level_0_name = '1_test_df.pickle'
    snap_num = 13
    level_1_name = '1_test_df.pickle'
    base = '/ptmp/mpa/ivkos/semianalytic_fesc/sn013'

    df_path = os.path.join(base, level_0_name)
    df = pd.read_pickle(df_path)
    sim, sim_path = get_sim()
    z = get_redshift(sim, snap_num)
    update_df(df, sim_path, snap_num, z)
    reformat_df(df, z)
    save_path = os.path.join(base, level_1_name)
    df.to_pickle(save_path)
