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
from astropy import constants as const
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo
from prospect.models.transforms import logsfr_ratios_to_sfrs



def compute_residuals(filename):
    """Calculate the residuals between the Prospector model photometry and the observed photometry for a given object ID.

    Args:
        objid (int): The galaxy ID for which to compute the residuals.
        show_plot (bool): Whether to display the plot of the model and observed photometry. Defaults to True.

    Returns:
        rows (dict): Dictionary containing the computed residuals and other relevant data.
    """
    
    try:    # try to open
        with open(filename, 'rb') as f:
            fit_data = pkl.load(f)
    except FileNotFoundError:
        print(f"⚠️ Attention: File {filename} not found. Skipping...")
        return 
    
    gid = fit_data['id']
    zred = fit_data['zred']
    
    model = fit_data['model']
    spec_best = model['spec_best']
    spec_16th = model['spec_16th']
    spec_median = model['spec_median']
    spec_84th = model['spec_84th']
    wave_spec = model['wave_spec']
    sample_specs = model['sample_specs']
    phot = model['phot']
    phot_miri = model['phot_miri']
    phot_miri_err = model['phot_miri_err']
    wave_phot = model['wave_phot']
    wave_phot_miri = model['wave_phot_miri']
    
    obs = fit_data['obs']
    obs_miri = fit_data['obs_miri']
    maggies_to_muJy = fit_data['maggies_to_muJy']
    
    # Convert to µJy
    lower_scaled = spec_16th * maggies_to_muJy    
    median_scaled = spec_median * maggies_to_muJy
    upper_scaled = spec_84th * maggies_to_muJy
    spec_scaled = spec_best * maggies_to_muJy
    
    wave_phot_microns = wave_phot * 1e-4  # convert to µm
    wave_phot_miri_microns = wave_phot_miri * 1e-4  # convert to µm
    
    phot_scaled = phot * maggies_to_muJy
    phot_miri_scaled = phot_miri * maggies_to_muJy
    phot_miri_err_scaled = phot_miri_err * maggies_to_muJy
    
    wave_spec_rs = wave_spec * 1e-4 * (1+zred)
    
    filters = obs['filters']
    filters_all = obs_miri['filters_all']
    
    if len(filters) == len(filters_all):
        print("⚠️It seems like there are no MIRI data available...Skipping")
        return None  # No MIRI bands, nothing to do
    
    phot_wave_all = obs_miri['phot_wave_all']
    
    # Extract model predictions at MIRI bands
    miri_mask = (phot_wave_all > 75000) & (phot_wave_all < 300000)  # AA
    
    # Extract obs at MIRI bands
    obs_wave = phot_wave_all[miri_mask]
    obs_flux = obs_miri['maggies_all'][miri_mask]
    obs_miri_err  = obs_miri['maggies_unc_all'][miri_mask]
    
    # Compute N_sigma
    delta = obs_flux - phot_miri
    tot_err = np.sqrt(phot_miri_err**2 + obs_miri_err**2)
    N_sigma = delta / tot_err
    
    # Compute it also in percentage of observed MIRI flux
    perc = delta / obs_flux
    
    # Filters for MIRI bands
    miri_bands = [f for f, keep in zip(filters_all, miri_mask) if keep]    
    
    rows = []
    for f, lam, nsig, obs, obs_err, mod, mod_err, p in zip(
        miri_bands, obs_wave, N_sigma, obs_flux, obs_err, phot_miri, phot_miri_err, perc
    ):   
    
        rows.append({
            "galaxy_id": gid,
            "zred": zred,
            "filter_name": f.name,
            "obs_wave": lam,
            "obs_flux": obs,
            "obs_err": obs_err,
            "model_flux": mod,
            "model_err": mod_err,
            "N_sigma": nsig,
            "perc_diff": p
        })
    
    

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

    #print(f"Flux array: {flux_array}")

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
    #print(f"Galaxy {gid} fluxes (valid only): {flux}")
    #print(f"Galaxy {gid} detections (all bands): {detections}")

    for band in all_bands:
        if band not in filters_available:
            # Remove keys for bands not observed
            detections.pop(band, None)

    filters = ph_miri['Filters'][0].split(',') # e.g. ['F770W', 'F1800W']
    
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
        "detections": detections,
        "nsig": dict(zip(filters_available, nsig)),
        "chi2_red": chi2_red,
        "frac_diff": dict(zip(filters_available, perc_diff))
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
