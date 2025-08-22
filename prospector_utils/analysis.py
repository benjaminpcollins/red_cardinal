import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime
import pandas as pd
import fsps
import prospect.io.read_results as reader
from prospect.sources import FastStepBasis
from prospect.utils.plotting import posterior_samples
from .params import build_obs, build_model, get_MAP
from .plotting import load_and_display
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

def compute_residuals(objid, show_plot=True):
    """Calculate the residuals between the Prospector model photometry and the observed photometry for a given object ID.

    Args:
        objid (int): The galaxy ID for which to compute the residuals.
        show_plot (bool): Whether to display the plot of the model and observed photometry. Defaults to True.

    Returns:
        rows (dict): Dictionary containing the computed residuals and other relevant data.
    """
    
    try:
        pkl_file = f'/Users/benjamincollins/University/Master/Red_Cardinal/prospector/pickle_files/{objid}.pkl'
        
        with open(pkl_file, 'rb') as f:
            fit_data = pkl.load(f)
            
    except FileNotFoundError:
        print(f"File {pkl_file} not found. Please check the object ID or file path.")
        return None

    print('======================================')
    print(f"Analysing fit data for galaxy {objid}...")

    obs = fit_data['obs']
    
    filters = obs['filters']
    filters_all = obs['filters_all']
    
    if len(filters) == len(filters_all):
        return None  # No MIRI bands, nothing to do
    
    wave_phot_all = obs['phot_wave_all']
    maggies_all = obs['maggies_all']
    model = fit_data['model']
    
    wave_spec = model['wave_spec']
    spec = model['spec_bestfit']
    spec_16th = model['spec_16th']
    spec_84th = model['spec_84th']
    
    phot = model['phot']
    wave_phot = model['wave_phot']
    
    zred = fit_data['redshift']
    
    # Compare to stored obs
    obs_flux = obs['maggies']
    obs_err  = obs['maggies_unc']
    
    # Use sedpy to compute the model photometry in the MIRI bands
    phot_new = get_model_photometry(spec, wave_spec, filters, zred)
    phot_new_all = get_model_photometry(spec, wave_spec, filters_all, zred)

    phot_16th = get_model_photometry(spec_16th, wave_spec, filters_all, zred)
    phot_84th = get_model_photometry(spec_84th, wave_spec, filters_all, zred)
    
    # Calculate the ratio between my model photometry and the Prospector model photometry
    ratio = phot_new / phot
    
    print("Ratio between model photometries: ", ratio)
    print("median ratio:", np.median(ratio))
    print("mean ratio  :", np.mean(ratio))
    print("std ratio   :", np.std(ratio))
    
    phot_new_all /= ratio.mean()  # normalise to match
    phot_84th /= ratio.mean()
    phot_16th /= ratio.mean()
    
    # Extract model predictions at MIRI bands
    miri_mask = (wave_phot_all > 75000) & (wave_phot_all < 210000)  # AA
    model_flux = phot_new_all[miri_mask]
    model_err  = 0.5 * (phot_84th[miri_mask] - phot_16th[miri_mask]) # approximate error for the prospector fit
    
    if show_plot:
        load_and_display(objid, mod=model_flux, mod_err=model_err)
    
    # Extract obs at MIRI bands
    obs_wave = obs['phot_wave_all']
    obs_flux = obs['maggies_all']
    obs_err  = obs['maggies_unc_all']
    
    obs_wave = obs_wave[miri_mask]
    obs_flux = obs_flux[miri_mask]
    obs_err  = obs_err[miri_mask]

    # Compute N_sigma
    delta = np.abs(obs_flux - model_flux)
    
    # Compute it also in percentage of observed MIRI flux
    perc = delta / obs_flux
    
    tot_err = np.sqrt(model_err**2 + obs_err**2)
    N_sigma = delta / tot_err
    
    # Filters for MIRI bands
    miri_bands = [f for f, keep in zip(filters_all, miri_mask) if keep]    
    
    rows = []
    for f, lam, nsig, obs, obs_err, mod, mod_err, p in zip(
        miri_bands, obs_wave, N_sigma, obs_flux, obs_err, model_flux, model_err, perc
    ):
    
        rows.append({
            "galaxy_id": objid,
            "zred": zred,
            "filter_name": f.name,
            "obs_wave": lam,
            "obs_flux": obs,
            "obs_err": obs_err,
            "model_flux": mod,
            "model_err": mod_err,
            "N_sigma": nsig,
            "perc_diff": p,
            "mean_ratio": ratio.mean(),
            "std_ratio": ratio.std()
        })
    
    print("Sigmas: ", N_sigma)
    print("Percentage difference:", perc)
    
    return rows
