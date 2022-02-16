from pyTNG import data_interface as _data_interface
import os
import pandas as pd
import numpy as np
import illustris_python as il
from pyTNG.spectra import StarSpectrumFactory
from pyTNG import spectra
import scipy.integrate as integ
import pyTNG.utils as utils
from pyTNG.cosmology import TNGcosmo


def get_sim():
    basepath = "/virgo/simulations/IllustrisTNG/"
    sim_name = 'L35n2160TNG'
    sim = _data_interface.TNG50Simulation(os.path.join(basepath, sim_name))
    sim_path = os.path.join(basepath, sim_name, 'output')
    return sim, sim_path


def get_dataset_df(sim, snap_num):
    dataset = next(sim.group_cat[snap_num].chunk_generator('subhalo'))
    keys_needed = [
                    'SubhaloGasMetallicity', 'SubhaloGasMetallicityHalfRad',
                    'SubhaloHalfmassRadType', 'SubhaloMassInHalfRad',
                    'SubhaloMassInHalfRadType', 'SubhaloMassInRad',
                    'SubhaloMassInRadType', 'SubhaloSFRinHalfRad',
                    'SubhaloSFRinRad', 'SubhaloPos']
    sub_dict = {key: dataset[key] for key in keys_needed}
    dataset_df = utils.dfFromArrDict(sub_dict)
    return dataset_df


def reduce_df(df):
    filt = (df[('SubhaloMassInRadType', 4)] > 0) & (df[('SubhaloSFRinRad', 0)] > 0)
    reduced_df = df[filt]
    columns_of_interest = [
                        ('SubhaloGasMetallicity', 0),
                        ('SubhaloGasMetallicityHalfRad', 0),
                        ('SubhaloHalfmassRadType', 4),
                        ('SubhaloMassInHalfRad', 0),
                        ('SubhaloMassInHalfRadType', 4),
                        ('SubhaloMassInRad', 0),
                        ('SubhaloMassInRadType', 0),
                        ('SubhaloMassInRadType', 4),
                        ('SubhaloSFRinHalfRad', 0),
                        ('SubhaloSFRinRad', 0),
                        ('SubhaloPos', 0),
                        ('SubhaloPos', 1),
                        ('SubhaloPos', 2)]
    reduced_df = reduce_df[columns_of_interest]
    return reduced_df


def save_df(df, name):
    base = '/ptmp/mpa/ivkos/semianalytic_fesc'
    full_path = os.path.join(base, f'{name}.pickle')
    df.to_pickle(full_path)
    return


def build_spec_fac():
    bpp = [
        '/ptmp/mpa/mglatzle/TNG_f_esc/BPASSv2.2.1_release-07-18-Tuatara/',
        '2.2.1',
        'chab300',
        'bin']
    specFac = StarSpectrumFactory(bpp)
    return specFac


def get_stellar_dist(stars, df, index):
    gal_center = np.array([df.loc[index][('SubhaloPos', 0)], df.loc[index][('SubhaloPos', 1)], df.loc[index][('SubhaloPos', 2)]])
    rel_pos = stars['Coordinates']-gal_center
    radius = df.loc[index][('SubhaloHalfmassRadType', 4)]*2
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


def add_lum_ion(df, sim_path, snap_num, z, specFac):
    luminosities = []
    ion_lums = []
    df[('ion_lum', 0)] = np.nan
    for idx in df.index:
        stars = il.snapshot.loadSubhalo(sim_path, snap_num, idx, 'stars')
        utils.dropWindParticles(stars)
        stars['Redshift'] = z
        stars['ages'] = spectra._computeAgeFromFormationTime(stars['Redshift'], stars['GFM_StellarFormationTime'])
        get_stellar_dist(stars, df, idx)
        idces = stars['rel_dist'] < 1
        utils._keepPartsByIdx(stars, idces)

        if stars['count'] < 10000:
            specFac.computeStellarSpectra(stars, Q_0=True)
            ion_lum = stars['Q_0'].sum()
            summed_spectrum = stars['spectra'].sum(axis=0)
            luminosity = integ.simps(summed_spectrum, stars['lambda'])
        else:
            print(stars['count'])
            star_arr = star_batches(stars, batchsize=10000)
            ion_lum, luminosity = calculate_batched_quants(star_arr)
        del stars
        del star_arr
        luminosities.append(luminosity)
        ion_lums.append(ion_lum)
    df[('luminosity', 0)] = luminosities
    df[('ion_lum', 0)] = ion_lum
    return


def build_new_df(df, name, z, h):
    new_df = pd.DataFrame()
    new_df['HaloMass'] = df[('SubhaloMassInRad', 0)]
    new_df['GasMass'] = df[('SubhaloMassInRadType', 0)]
    new_df['StarMass'] = df[('SubhaloMassInRadType', 4)]
    new_df['SFR'] = df[('SubhaloSFRinRad', 0)]
    new_df['com_radius'] = 2*df[('SubhaloHalfmassRadType', 4)]
    new_df['Z'] = df[('SubhaloGasMetallicity', 0)]
    new_df['ion_lum'] = df[('ion_lum', 0)]
    new_df['luminosity'] = df[('luminosity', 0)]

    new_df['radius'] = (z+1)*new_df['com_radius']/h
    new_df['gas_surface_dens'] = 1e10*new_df['GasMass']/(h*np.pi*new_df['radius']**2)
    new_df['star_surface_dens'] = 1e10*new_df['StarMass']/(h*np.pi*new_df['radius']**2)
    new_df['sfr_surface_dens'] = new_df['SFR']/(np.pi*new_df['radius']**2)
    new_df['ionizing_flux'] = new_df['ion_lum']/(np.pi*new_df['radius']**2)

    save_df(df, name)
    return


def build_lv0(snap_num, output_name, from_tng=True, df_path=None, save_tng_df=False):
    sim, sim_path = get_sim()

    if from_tng:
        dataset_df = get_dataset_df(sim, snap_num)
        df = reduce_df(dataset_df)
        if save_tng_df:
            save_df(f'tng_df_snap0{snap_num}')
    else:
        df = pd.read_pickle(df_path)

    specFac = build_spec_fac()

    z = sim.snap_cat[snap_num].header['Redshift']
    h = TNGcosmo.h

    add_lum_ion(df, sim_path, snap_num, z, specFac)
    save_df(df, 'for_testing')

    build_new_df(df, output_name, z, h)
    return


snap_num = 13
df_path = f'/ptmp/mpa/ivkos/semianalytic_fesc/testing/sn0{snap_num}.pickle'
output_name = f'df_snap0{snap_num}_lv0'
build_lv0(snap_num, output_name, from_tng=False, df_path=df_path, save_tng_df=False)
