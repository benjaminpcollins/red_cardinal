import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime
import fsps
import prospect.io.read_results as reader
from prospect.sources import FastStepBasis
from prospect.utils.plotting import posterior_samples
from .params import build_obs, build_model, get_MAP
from sedpy.observate import getSED
from astropy import constants as const
import pyphot

def predict_phot():

    lib = pyphot.get_library
    print(lib)
    
def get_model_photometry(spec, wave_spec, filters, zred):
    """
    Generate a model photometry prediction on a prospector model spectrum using sedpy
    
    Parameters:
    -----------
    spec : ndarray
        The model spectrum in units of maggies (standard units for Prospector).
    wave_spec : ndarray
        The wavelength grid corresponding to the model spectrum in Angstroms.
    filters : list of sedpy Filter objects
        The list of filters for which to compute the photometry.
    zred : float
        The redshift of the object.
    
    Returns:
    --------
    phot_new : ndarray
        The predicted photometry in the same units as the input spectrum.
    """ 
    
    # 1) maggies -> f_nu [erg/s/cm^2/Hz]
    fnu_cgs = spec * 3631e-23
    
    # 2) f_nu -> f_lambda_rest [erg/s/cm^2/Å]
    wave_cm = wave_spec * 1e-8  # Convert Å to cm
    flam_cgs = fnu_cgs * const.c.cgs.value / wave_cm**2 / 1e8  # /1e8 converts cm⁻¹ to Å⁻¹
    
    # 3) Move to observer frame
    lam_obs_A = wave_spec * (1.0 + zred)  # in AA
    flam_obs  = flam_cgs / (1.0 + zred)  # critical!
    
    # 4) Recreate photometry
    phot_new = getSED(lam_obs_A, flam_obs, filterlist=filters, linear_flux=True)   
    
    return phot_new

def compute_residuals(pkl_file):
    with open(pkl_file, 'rb') as f:
        fit_data = pkl.load(f)

    obs = fit_data['obs']
    filters = obs['filters']
    filters_all = obs['filters_all']
    wave_phot_all = obs['phot_wave_all']
    maggies_all = obs['maggies_all']
    
    # dir(f) to check attributes
    
    for f in filters_all:
        print(f"Filter {f.name}: {f.wave_effective} Å")
    
    model = fit_data['model']
    wave_spec = model['wave_spec']
    spec = model['spec_bestfit']
    spec_16th = model['spec_16th']
    spec_84th = model['spec_84th']
    
    phot = model['phot']
    wave_phot = model['wave_phot']
    
    corr_factor = fit_data['calib']['correction']
    factor = 3631e6 # convert maggies to µJy
    scale_factor = corr_factor * factor
    calib_unc = fit_data['calib']['uncertainty']
    zred = fit_data['redshift']
    
    spec *= corr_factor
    spec_16th *= corr_factor
    spec_84th *= corr_factor
    
    # Use sedpy to compute the model photometry in the MIRI bands
    phot_new = get_model_photometry(spec, wave_spec, filters, zred)
    phot_new_all = get_model_photometry(spec, wave_spec, filters_all, zred)
    
    phot_16th = get_model_photometry(spec_16th, wave_spec, filters_all, zred)
    phot_84th = get_model_photometry(spec_84th, wave_spec, filters_all, zred)
    
    # Compare to stored obs
    obs_flux = obs['maggies']
    obs_err  = obs['maggies_unc']
    names    = [f.name for f in filters]
    
    # Calculate the ratio between my model photometry and the Prospector model photometry
    ratio = phot_new / phot
    
    print("Ratio between model photometries: ", ratio)
    print("median ratio:", np.median(ratio))
    print("mean ratio  :", np.mean(ratio))
    print("std ratio   :", np.std(ratio))
    
    phot_new_all /= ratio.mean()  # normalise to match
    phot_84th /= ratio.mean()
    phot_16th /= ratio.mean()
    
    plt.plot(wave_spec*(1+zred), spec, alpha=0.8, label="Model spectrum")
    plt.scatter(wave_phot_all, maggies_all, color='black', label='Observed photometry') 
    plt.scatter(wave_phot, phot, marker='x', color='green', alpha=1, label='Prospector model photometry')
    plt.scatter(wave_phot_all, phot_new_all, marker='x', color='red', alpha=0.8, label='Recreated model photmetry')
    plt.loglog()
    plt.xlim(4000, 350000)  
    plt.ylim(1e-12, 1e-6)  
    plt.legend()
    plt.show()
    
    # Extract model predictions at MIRI bands
    miri_mask = (wave_phot_all > 75000) & (wave_phot_all < 210000)  # AA
    model_flux = phot_new_all[miri_mask]
    model_err  = 0.5 * (phot_84th[miri_mask] - phot_16th[miri_mask]) # approximate error for the prospector fit
    
    # Extract obs at MIRI bands
    obs_wave = obs['phot_wave_all']
    obs_flux = obs['maggies_all']
    obs_err  = obs['maggies_unc_all']

    obs_flux = obs_flux[miri_mask]
    obs_err  = obs_err[miri_mask]

    # Properly account for the calibration uncertainty (individual for each band -> array)
    fcal = (calib_unc/corr_factor)
    cal_err = fcal * model_flux
    
    # Compute N_sigma
    delta = np.abs(obs_flux - model_flux)
    # Compute it also in percentage of observed MIRI flux
    # ...
    
    tot_err = np.sqrt(model_err**2 + obs_err**2)# + cal_err**2)
    N_sigma = delta / tot_err
    
    print("N_sigma values for MIRI bands:", N_sigma)

    return N_sigma