import os
import pandas as pd
import numpy as np
import illustris_python as il
from pyTNG.spectra import StarSpectrumFactory
from pyTNG import spectra
from pyTNG.cosmology import TNGcosmo
import scipy.integrate as integ
import pyTNG.utils as utils
from utils import get_stellar_dist_gas
from utils import get_particle_dist
from utils import get_redshift
from utils import get_sim
import astropy.units as u


h = TNGcosmo.h


def build_spec_fac():
    bpp = [
        '/ptmp/mpa/mglatzle/TNG_f_esc/BPASSv2.2.1_release-07-18-Tuatara/',
        '2.2.1',
        'chab300',
        'bin']
    specFac = StarSpectrumFactory(bpp)
    return specFac


def get_stellar_dist(stars, df, index):
    gal_center = np.array([df.loc[index]['Halo_pos_x'], 
                           df.loc[index]['Halo_pos_y'], 
                           df.loc[index]['Halo_pos_z']])
    rel_pos = stars['Coordinates']-gal_center
    radius = df.loc[index]['r']*2
    dist = np.sqrt(np.sum(np.square(rel_pos), axis=1))
    rel_dist = dist/radius
    stars['rel_dist'] = rel_dist
    return


def star_batches(stars, batchsize):
    stars_arr = []
    batches = stars['count']//batchsize+1
    start = 0
    end = batchsize
    for i in range(batches):
        star_batch = {}
        star_batch['Redshift'] = stars['Redshift']
        if i+1 < batches:
            star_batch['count'] = batchsize
            for key in stars.keys():
                if key not in ['count', 'Redshift']:
                    star_batch[key] = stars[key][start:end]
            stars_arr.append(star_batch)
            start = end
            end += batchsize
        else:
            for key in stars.keys():
                if key not in ['count', 'Redshift']:
                    star_batch[key] = stars[key][start:]
            star_batch['count'] = len(stars['ages'][start:])
            stars_arr.append(star_batch)
    return stars_arr


def calculate_batched_quants(stars_arr, specFac):
    ion_lum = 0
    luminosity = 0
    for batch in stars_arr:
        specFac.computeStellarSpectra(batch, Q_0=True)
        ion_lum += batch['Q_0'].sum()
        summed_spectrum = batch['spectra'].sum(axis=0)
        luminosity += integ.simps(summed_spectrum, batch['lambda'])
        del batch['spectra']
    return ion_lum, luminosity


def get_star_forming_gas(gas):
    star_forming_gas = {}
    star_forming = gas['StarFormationRate'] > 1e-9
    star_forming_gas['SFR'] = gas['StarFormationRate'][star_forming]
    star_forming_gas['Centers'] = gas['Coordinates'][star_forming]
    star_forming_gas['Densities'] = gas['Density'][star_forming]
    star_forming_gas['Coordinates'] = gas['Coordinates'][star_forming]
    star_forming_gas['Z'] = gas['GFM_Metallicity'][star_forming]
    star_forming_gas['Masses'] = gas['Masses'][star_forming]
    star_forming_gas['count'] = np.sum(star_forming)
    return star_forming_gas


def select_stars_in_gas(stars, gas):
    relevant_stars = {}
    for key in stars.keys():
        if key == 'count':
            relevant_stars[key] = 0
        elif key == 'Redshift':
            relevant_stars[key] = stars[key]
        else:
            relevant_stars[key] = []
    for i, center in enumerate(gas['Centers']):
        get_stellar_dist_gas(stars, center, gas['Radii'][i])
        new_stars = stars['rel_dist'] < 1
        for key in stars.keys():
            if key not in {'count', 'Redshift', 'rel_dist'}:
                relevant_stars[key].extend(stars[key][new_stars])
                stars[key] = np.delete(stars[key], new_stars, axis=0)
        relevant_stars['count'] += np.sum(new_stars)
    for key in stars.keys():
        if key not in {'count', 'Redshift', 'rel_dist'}:
            relevant_stars[key] = np.array(relevant_stars[key])
    return relevant_stars


def get_radius_vol(gas):
    gas['Radii'] = ((3/(4*np.pi)*gas['Masses']/gas['Densities']))**(1/3)
    gas['Volumes'] = gas['Masses']/gas['Densities']
    return gas


def get_subset(parts, idces):
    try:
        newcount = (idces == True).sum()

        subset = {}
        for key, value in parts.items():
            try:
                subset[key] = value[idces]
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
        if 'count' in parts:
            subset['count'] = newcount
    except KeyError as e:
        if str(e) == 'GFM_StellarFormationTime' and parts['count'] == 0:
            pass
        else:
            raise

    return subset


def compute_small_gal_em(stars, specFac, idx):
    try:
        specFac.computeStellarSpectra(stars, Q_0=True)
        ion_em = stars['Q_0'].sum()
        summed_spectrum = stars['spectra'].sum(axis=0)
        bol_lum = integ.simps(summed_spectrum, stars['lambda'])
    except:
        print(f'An error occured in halo {idx}')
        ion_em = np.nan
        bol_lum = np.nan
    return ion_em, bol_lum


def compute_large_gal_em(stars, specFac, idx):
    try:
        star_arr = star_batches(stars, batchsize=10000)
        ion_em, bol_lum = calculate_batched_quants(star_arr, specFac)
        del star_arr
    except:
        print(f'An error occured in halo {idx}')
        ion_em = np.nan
        bol_lum = np.nan
    return ion_em, bol_lum


# Function which adds additional subhalo properties (luminosities,
# ionizing luminosities, gas_masses, star_masses, sfrs, metallicities
# and volumes) corresponding to only star forming gas cells
def add_sf_quantities(df, sim_path, snap_num, z):

    mass_to_g = (1*u.Msun).to(u.g).value*1e10/h
    volume_to_cm3 = ((1*u.kpc).to(u.cm)/h/(1+z))**3
    L_sun_to_erg_s = (1*u.Lsun).to(u.erg/u.s)

    bol_lums_2r = []
    ion_ems_2r = []
    gas_masses_2r = []
    star_masses_2r = []
    sfrs_2r = []
    metallicities_2r = []
    volumes_2r = []
    
    bol_lums_r = []
    ion_ems_r = []
    gas_masses_r = []
    star_masses_r = []
    sfrs_r = []
    metallicities_r = []
    volumes_r = []

    specFac = build_spec_fac()
    for idx in df.index:
        print(f'Working on subhalo {idx}')
        stars = il.snapshot.loadSubhalo(sim_path, snap_num, idx, 'stars')
        utils.dropWindParticles(stars)
        stars['Redshift'] = z
        gas = il.snapshot.loadSubhalo(sim_path, snap_num, idx, 'gas')
        
        relevant_gas = get_star_forming_gas(gas)
        get_particle_dist(relevant_gas, df, idx)
        idces_gas_in_rad = relevant_gas['rel_dist'] < 1
        idces_gas_in_half_rad = relevant_gas['rel_dist'] < 0.5
        
        gas_in_rad = get_subset(relevant_gas, idces_gas_in_rad)
        gas_in_half_rad = get_subset(relevant_gas, idces_gas_in_half_rad)
        get_radius_vol(gas_in_rad)
        get_radius_vol(gas_in_half_rad)
        relevant_stars_rad = select_stars_in_gas(stars, gas_in_rad)
        relevant_stars_half_rad = select_stars_in_gas(stars, gas_in_half_rad)
        del stars

        # If no stars are left after filtering remove this halo from the df
        if relevant_stars_half_rad['count'] == 0:
            df.drop(index=idx, inplace=True)
            continue

        relevant_stars_rad['ages'] = spectra._computeAgeFromFormationTime(relevant_stars_rad['Redshift'], 
                                                                      relevant_stars_rad['GFM_StellarFormationTime'])

        relevant_stars_half_rad['ages'] = spectra._computeAgeFromFormationTime(relevant_stars_half_rad['Redshift'], 
                                                                      relevant_stars_half_rad['GFM_StellarFormationTime'])

        if relevant_stars_rad['count'] < 10000:
            ion_em_rad, bol_lum_rad = compute_small_gal_em(relevant_stars_rad, specFac, idx)
            ion_em_half_rad, bol_lum_half_rad = compute_small_gal_em(relevant_stars_half_rad, specFac, idx)
        else:
            ion_em_rad, bol_lum_rad = compute_large_gal_em(relevant_stars_rad, specFac, idx)
            ion_em_half_rad, bol_lum_half_rad = compute_large_gal_em(relevant_stars_half_rad, specFac, idx)
            
        bol_lums_2r.append(bol_lum_rad)
        ion_ems_2r.append(ion_em_rad)
        gas_masses_2r.append(np.sum(gas_in_rad['Masses']))
        star_masses_2r.append(np.sum(relevant_stars_rad['Masses']))
        sfrs_2r.append(np.sum(gas_in_rad['SFR']))
        metallicities_2r.append(np.sum(gas_in_rad['Z']*gas_in_rad['Masses'])/np.sum(gas_in_rad['Masses']))
        volumes_2r.append(np.sum(gas_in_rad['Volumes']))

        bol_lums_r.append(bol_lum_half_rad)
        ion_ems_r.append(ion_em_half_rad)
        gas_masses_r.append(np.sum(gas_in_half_rad['Masses']))
        star_masses_r.append(np.sum(relevant_stars_half_rad['Masses']))
        sfrs_r.append(np.sum(gas_in_half_rad['SFR']))
        metallicities_r.append(np.sum(gas_in_half_rad['Z']*gas_in_half_rad['Masses'])/np.sum(gas_in_half_rad['Masses']))
        volumes_r.append(np.sum(gas_in_half_rad['Volumes']))
    
    df['Bol_lum_sf_2r'] = np.array(bol_lums_2r)*L_sun_to_erg_s
    df['Ion_em_sf_2r'] = np.array(ion_ems_2r)
    df['M_gas_sf_2r'] = np.array(gas_masses_2r)*mass_to_g
    df['SFR_sf_2r'] = np.array(sfrs_2r)
    df['M_star_sf_2r'] = np.array(star_masses_2r)*mass_to_g
    df['V_sf_2r'] = np.array(volumes_2r)*volume_to_cm3
    df['Z_sf_2r'] = np.array(metallicities_2r)

    df['Bol_lum_sf_r'] = np.array(bol_lums_r)*L_sun_to_erg_s
    df['Ion_em_sf_r'] = np.array(ion_ems_r)
    df['M_gas_sf_r'] = np.array(gas_masses_r)*mass_to_g
    df['SFR_sf_r'] = np.array(sfrs_r)
    df['M_star_sf_r'] = np.array(star_masses_r)*mass_to_g
    df['V_sf_r'] = np.array(volumes_r)*volume_to_cm3
    df['Z_sf_r'] = np.array(metallicities_r)
    return


if __name__ == '__main__':
    level_1_name = '1_df.pickle'
    snap_num = 13
    level_2_name = '2_df.pickle'
    base = '/ptmp/mpa/ivkos/semianalytic_fesc/sn013'

    df_path = os.path.join(base, level_1_name)
    df = pd.read_pickle(df_path)

    snap_num = 13
    sim, sim_path = get_sim()
    z = get_redshift(sim, snap_num)
    add_sf_quantities(df, sim_path, snap_num, z)

    save_path = os.path.join(base, level_2_name)
    df.to_pickle(save_path)
