import pandas as pd
import numpy as np
import os
import illustris_python as il
import astropy.units as u
from pyTNG.cosmology import TNGcosmo
from df_level0 import get_sim

h = TNGcosmo.h


def get_particle_dist(particles, df, index):
    gal_center = np.array([df.loc[index][('SubhaloPos', 0)], 
                           df.loc[index][('SubhaloPos', 1)], 
                           df.loc[index][('SubhaloPos', 2)]])
    rel_pos = particles['Coordinates']-gal_center
    radius = df.loc[index][('SubhaloHalfmassRadType', 4)]*2
    dist = np.sqrt(np.sum(np.square(rel_pos), axis=1))
    rel_dist = dist/radius
    particles['rel_dist'] = rel_dist
    return


def get_volume_mass(df, sim_path, snap_num, z):
    counter = 0
    volumes = []
    masses = []
    mass_to_g = (1*u.Msun).to(u.kg).value*1e10/h
    volume_to_cm3 = ((1*u.kpc).to(u.cm)/h/(1+z))**3
    for idx in df.index:
        gas = il.snapshot.loadSubhalo(sim_path, snap_num, idx, 'gas')
        get_particle_dist(gas, df, idx)
        gas_in_rad = gas['rel_dist'] < 1
        gas_masses = gas['Masses'][gas_in_rad]
        gas_densities = gas['Density'][gas_in_rad]
        gas_volumes = gas_masses/gas_densities
        volumes.append(np.sum(gas_volumes))
        masses.append(np.sum(gas_masses))
        if counter % 100 == 0:
            print(f'{100*counter/len(df):.2f}% of all halos processed')
        counter += 1
    df[('masses', 0)] = np.array(masses)*mass_to_g
    df[('volumes', 0)] = np.array(volumes)*volume_to_cm3
    return


def get_column_height_dens(df, z):
    rad = df[('SubhaloHalfmassRadType', 4)]*2
    dist_to_cm = (1*u.kpc).to(u.cm)/h/(1+z)
    rad_cm = rad*dist_to_cm
    areas = rad_cm**2*np.pi
    df[('column_height', 0)] = df[('volumes', 0)]/areas
    df[('column_dens', 0)] = df[('masses', 0)]/df[('column_height', 0)]
    return


if __name__ == '__main__':
    df_path = '/ptmp/mpa/ivkos/semianalytic_fesc/testing/sn013.pickle'
    df = pd.read_pickle(df_path)

    snap_num = 13
    df[('SubhaloMassInRadType', 0)]
    sim, sim_path = get_sim()
    z = sim.snap_cat[snap_num].header['Redshift']
    get_volume_mass(df, sim_path, snap_num, z)
    base = '/ptmp/mpa/ivkos/semianalytic_fesc'
    full_path = os.path.join(base, 'df_with_volumes.pickle')
    df.to_pickle(full_path)
