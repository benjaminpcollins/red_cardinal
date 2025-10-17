import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime
import pandas as pd
import fsps
import prospect.io.read_results as reader
from .params import build_obs, build_model, get_MAP
from .plotting import load_and_display
from sedpy.observate import getSED
from astropy import constants as const
from astropy.io import fits
import pyphot
from astropy.cosmology import WMAP9 as cosmo
from prospect.models.transforms import logsfr_ratios_to_sfrs


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
        print("⚠️It seems like there are no MIRI data available...Skipping")
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
    
    phot_new_all /= ratio.mean()  # normalise to match
    phot_84th /= ratio.mean()
    phot_16th /= ratio.mean()
    
    # Extract model predictions at MIRI bands
    miri_mask = (wave_phot_all > 75000) & (wave_phot_all < 300000)  # AA
    model_flux = phot_new_all[miri_mask]
    model_err  = 0.5 * (phot_84th[miri_mask] - phot_16th[miri_mask]) # approximate error for the prospector fit
    
    # Extract obs at MIRI bands
    obs_wave = obs['phot_wave_all']
    obs_flux = obs['maggies_all']
    obs_err  = obs['maggies_unc_all']
    
    obs_wave = obs_wave[miri_mask]
    obs_flux = obs_flux[miri_mask]
    obs_err  = obs_err[miri_mask]

    if show_plot: load_and_display(objid, mod=model_flux, mod_err=model_err, outfile=f"/Users/benjamincollins/University/Master/Red_Cardinal/prospector/fits_v3/{objid}.png")

    # Compute N_sigma
    delta = obs_flux - model_flux
    
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
    
    if objid == '8465': 
        print("⚠️ Found galaxy with ID 8465, printing stats")
    
    return rows

def get_galaxy_properties(gid, phot_miri, non_detections=None):
    """Obtain the star formation rates (SFRs) from the Prospector fit for a given galaxy ID.

    Args:
        gid (int): The galaxy ID for which to obtain the SFRs.
    
    Returns:
        sfrs (ndarray): Array of star formation rates in solar masses per year.
    """
    
    # ============================
    # Part related to PROSPECTOR
    # ============================
    prospect_dir = "/Users/benjamincollins/University/master/Red_Cardinal/prospector/outputs/"
    
    # Load the h5 file for the given objid
    h5_path = os.path.join(prospect_dir, f"output_{gid}*.h5")
    h5_file = glob.glob(h5_path)
    
    try:
        h5_file = h5_file[0]
        print(f"Found PROSPECTOR results for objid {gid}: {h5_file}")
    except IndexError:
        print(f"No PROSPECTOR results found for objid {gid}.")
        return None
    
    # Load PROSPECTOR results
    full_path = os.path.join(prospect_dir, h5_file)
    results, _, _ = reader.results_from(full_path)
    
    # Get the MAP parameters
    map_parameters = get_MAP(results)
    map_parameters = map_parameters[:-3]
    
    # Build the MAP dictionary
    MAP = {}
    for a,b in zip(results['theta_labels'], map_parameters):
        MAP[a] = b

    zred = MAP['zred']
    logmass = MAP['logmass']
    dust2 = MAP['dust2']    # extract the diffuse dust V-band optical depth
    
    """
    dust_tesc – (default: 7.0) 
        Stars younger than dust_tesc are attenuated by both dust1 and dust2, 
            while stars older are attenuated by dust2 only. Units are log(yrs).
    dust1 – (default: 0.0) 
        Dust parameter describing the attenuation of young stellar light, 
            i.e. where t <= dust_tesc (for details, see Conroy et al. 2009a).
    dust2 – (default: 0.0) 
        Dust parameter describing the attenuation of old stellar light, 
        i.e. where t > dust_tesc (for details, see Conroy et al. 2009a).

    Summary taken from https://dfm.io/python-fsps/current/stellarpop_api/#fsps.StellarPopulation.dust_mass
    """ 

    # Reconstruct agebins used in the fits
    tuniv = cosmo.age(zred).value
    agelims_Myr = np.append( np.logspace( np.log10(30.0), np.log10(0.8*tuniv*1000), 12), [0.9*tuniv*1000, tuniv*1000])
    agelims = np.concatenate( ( [0.0], np.log10(agelims_Myr*1e6) ))
    agebins = np.array([agelims[:-1], agelims[1:]]).T
    nbins = len(agelims) - 1
    
    # Collect logsfr_ratios
    logsfr_ratios = np.array([MAP[f"logsfr_ratios_{i}"] for i in range(1, len([k for k in MAP if k.startswith("logsfr_ratios_")])+1)])        
    
    # Convert to SFRs
    sfrs = logsfr_ratios_to_sfrs(logmass, logsfr_ratios, agebins)
    
    # Convert log age bins to linear time (yr)
    bin_edges = 10**agebins  # shape (nbins, 2)
    
    # Select bins younger than 100 Myr
    timescale = 1e8  # 100 Myr in years
    tcut = timescale
    
    # Compute overlap of each bin with interval [0, tcut]
    overlap = np.maximum(0.0, np.minimum(bin_edges[:,1], tcut) - np.minimum(bin_edges[:,0], tcut))
    
    # For bins that are fully within [0,tcut] overlap == dt, partial bins get partial dt
    mass_in_window = np.sum(sfrs * overlap)
    sfr_last100 = mass_in_window / timescale
    
    # ============================
    # Part related to the photometry
    # ============================
    
    # List of all MIRI bands
    all_bands = ['F770W', 'F1000W', 'F1800W', 'F2100W']

    ph_miri = phot_miri[phot_miri['ID'] == gid]

    if len(ph_miri) == 0:
        print(f"No MIRI entry for galaxy {gid}")
        return None

    # Filters actually observed for this galaxy
    filters_available = ph_miri['Filters'][0].split(',')  # e.g., ['F770W', 'F1800W']
    flux_array = np.ma.filled(ph_miri['Flux'][0], fill_value=np.nan)
    err_array  = np.ma.filled(ph_miri['Flux_Err'][0], fill_value=np.nan)

    print(f"Flux array: {flux_array}")

    # Initialize dictionaries
    flux = {}       # Only contains valid fluxes
    err  = {}
    detections = {band: False for band in all_bands}  # Default False

    # Fill in values
    for band, fval, ferr in zip(all_bands, flux_array, err_array):
        # Check for non-detections
        is_detected = True
        if non_detections is not None and gid in non_detections.get(band, []):
            is_detected = False
        elif np.isnan(fval) or fval < 0:
            is_detected = False

        detections[band] = is_detected

        if is_detected:
            flux[band] = fval
            err[band]  = ferr

    # Example output
    print(f"Galaxy {gid} fluxes (valid only): {flux}")
    print(f"Galaxy {gid} detections (all bands): {detections}")


    filters = ph_miri['Filters'][0].split(',') # e.g. ['F770W', 'F1800W']
    
    print(filters)
    
    # Dictionary to track which bands are valid 
    detected = {band: True for band in filters} 
    
    for band in filters: 
        # If this galaxy is marked as a non-detection in that band 
        if non_detections is not None and gid in non_detections.get(band, []): 
            detected[band] = False
            
    # ============================
    # Part related to the fit quality
    # ============================
    
    csv_path = '/Users/benjamincollins/University/Master/Red_Cardinal/prospector/analysis/residuals_abs.csv'
    df = pd.read_csv(csv_path)
    subset = df[df['galaxy_id'] == gid]
    nsig = subset['N_sigma']
    
    # Compute reduced chi^2 per galaxy
    n_filters = len(subset)
    
    # for undetected galaxies there are 0 valid MIRI bands
    if n_filters == 0:
        chi2_red = np.nan
    else:
        chi2_red = np.sum(nsig**2) / n_filters           

    perc_diff = subset['perc_diff']
    
    
    galaxy_data = {
        "gid": gid,
        "zred": zred,
        "logmass": logmass,
        "sfrs": sfrs,                     # SFR in each bin
        "dust": dust2,
        "sfr_last100": sfr_last100,       # averaged over last 100 Myr
        "fluxes": flux,
        "errors": err,
        "detections": detected,
        "nsig": dict(zip(filters, nsig)),
        "chi2_red": chi2_red,
        "frac_diff": dict(zip(filters, perc_diff))
    }
    
    return galaxy_data


def get_extremes(values, gids, n=2, abs=False, dropna=True):
    """
    Return the lowest and highest n values (with IDs).
    
    Parameters
    ----------
    values : array-like
        Array of values (e.g. dust, nsig).
    gids : array-like
        IDs corresponding to the values.
    n : int
        Number of extremes per side.
    dropna : bool
        If True, filter out NaN values first.
        
    Returns
    -------
    dict with keys "lowest" and "highest", 
    each containing list of (id, value) tuples.
    """
    vals = np.array(values)
    ids  = np.array(gids)

    if dropna:
        mask = ~np.isnan(vals)
        vals, ids = vals[mask], ids[mask]

    if abs == True:
        vals2 = np.abs(vals)
    else:
        vals2 = np.copy(vals)
            
    order = np.argsort(vals2)
    lowest  = [(ids[i], vals[i]) for i in order[:n]]
    highest = [(ids[i], vals[i]) for i in order[-n:]]

    return {"lowest": lowest, "highest": highest}
