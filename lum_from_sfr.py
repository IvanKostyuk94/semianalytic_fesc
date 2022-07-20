import pandas as pd
import astropy.units as u


# Units of the bolometric flux are erg*cm-2*s-1
# Units of ionizing flux are cm-2
# Sigma_sfr needs to be converted to M_sum*yr-1*kpc-2
def get_lum_from_sfr(df):
    sigma_sfr_to_ion_flux = 3e10
    mean_phot_e_ion_spec = 20.4
    bolometric_correction = 5
    cm_to_kpc = (1*u.cm).to(u.kpc).value
    df['Ion_flux_r'] = df['Sigma_SFR_r']*sigma_sfr_to_ion_flux/cm_to_kpc**2
    df['Ion_flux_2r'] = df['Sigma_SFR_2r']*sigma_sfr_to_ion_flux/cm_to_kpc**2

    df['Bol_flux_r'] = df['Sigma_SFR_r']*bolometric_correction/cm_to_kpc**2
    df['Bol_flux_2r'] = df['Sigma_SFR_2r']*bolometric_correction/cm_to_kpc**2
    return 


if __name__ == '__main__':
    df_path = '/ptmp/mpa/ivkos/semianalytic_fesc/sn013/reduced_df_update1.pickle'
    df = pd.read_pickle(df_path)
    get_lum_from_sfr(df)
    df.to_pickle(df_path)
