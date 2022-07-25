import os
import pandas as pd
import numpy as np
import illustris_python as il
from pyTNG.spectra import StarSpectrumFactory
from pyTNG import spectra
import scipy.integrate as integ
import pyTNG.utils as utils
from utils import get_stellar_dist_gas
from utils import get_particle_dist
from utils import get_redshift
from utils import get_sim


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


# Function which adds additional subhalo properties (luminosities,
# ionizing luminosities, gas_masses, star_masses, sfrs, metallicities
# and volumes) corresponding to only star forming gas cells
def add_sf_quantities(df, sim_path, snap_num, z):
    luminosities = []
    ion_lums = []
    gas_masses = []
    star_masses = []
    sfrs = []
    metallicities = []
    volumes = []
    specFac = build_spec_fac()
    for idx in df.index:
        print(f'Working on subhalo {idx}')
        stars = il.snapshot.loadSubhalo(sim_path, snap_num, idx, 'stars')
        utils.dropWindParticles(stars)
        stars['Redshift'] = z
        gas = il.snapshot.loadSubhalo(sim_path, snap_num, idx, 'gas')
        relevant_gas = get_star_forming_gas(gas)
        get_particle_dist(relevant_gas, df, idx)
        idces = relevant_gas['rel_dist'] < 1
        utils._keepPartsByIdx(relevant_gas, idces)
        get_radius_vol(relevant_gas)
        relevant_stars = select_stars_in_gas(stars, relevant_gas)
        del stars

        # If no stars are left after filtering remove this halo from the df
        if relevant_stars['count'] == 0:
            df.drop(index=idx, inplace=True)
            continue

        relevant_stars['ages'] = spectra._computeAgeFromFormationTime(relevant_stars['Redshift'], 
                                                                      relevant_stars['GFM_StellarFormationTime'])

        if relevant_stars['count'] < 10000:
            try:
                specFac.computeStellarSpectra(relevant_stars, Q_0=True)
                ion_lum = relevant_stars['Q_0'].sum()
                summed_spectrum = relevant_stars['spectra'].sum(axis=0)
                luminosity = integ.simps(summed_spectrum, relevant_stars['lambda'])
            except:
                print(f'An error occured in halo {idx}')
                ion_lum = np.nan
                luminosity = np.nan
        else:
            try:
                star_arr = star_batches(relevant_stars, batchsize=10000)
                ion_lum, luminosity = calculate_batched_quants(star_arr, specFac)
                del star_arr
            except:
                print(f'An error occured in halo {idx}')
                ion_lum = np.nan
                luminosity = np.nan

        luminosities.append(luminosity)
        ion_lums.append(ion_lum)
        gas_masses.append(np.sum(relevant_gas['Masses']))
        star_masses.append(np.sum(relevant_stars['Masses']))
        sfrs.append(np.sum(relevant_gas['SFR']))
        metallicities.append(np.sum(relevant_gas['Z']*relevant_gas['Masses'])/np.sum(relevant_gas['Masses']))
        volumes.append(np.sum(relevant_gas['Volumes']))
    df['Bol_lum_sf'] = luminosities
    df['Ion_em_sf'] = ion_lums
    df['M_gas_sf'] = gas_masses
    df['SFR_sf'] = sfrs
    df['M_star_sf'] = star_masses
    df['V_gas_sf'] = volumes
    df['Z_sf'] = metallicities
    return


if __name__ == '__main__':
    base = '/ptmp/mpa/ivkos/semianalytic_fesc/sn013'
    df_path = os.path.join(base, 'reduced_df_update1.pickle')
    df = pd.read_pickle(df_path)

    snap_num = 13
    sim, sim_path = get_sim()
    z = get_redshift(sim, snap_num)
    add_sf_quantities(df, sim_path, snap_num, z)

    save_path = os.path.join(base, 'reduced_df_update2.pickle')
    df.to_pickle(save_path)
