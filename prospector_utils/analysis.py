import os
import glob
import numpy as np
import pickle as pkl
import pandas as pd
import prospect.io.read_results as reader

from astropy.io import fits
#from astropy.cosmology import WMAP9 as cosmo
from prospect.models.transforms import logsfr_ratios_to_sfrs
from prospect.sources import FastStepBasis
from prospect.utils.plotting import posterior_samples
from astropy.cosmology import Planck18 as cosmo
from prospect.models.sedmodel import PolySpecModel, SpecModel


from astropy import units as u
from astropy import constants as const
from scipy.integrate import trapezoid

from .params import *
from .plotting import *



def analyse_fits(galaxy_ids, phot_table, data_dir, plot_dir=None, stats_dir=None, 
                 add_duste=True, n_samples=100, run='standard', overwrite=False):
    """Main function to reconstruct and plot PROSPECTOR results with MIRI data
    
    Parameters:
    -----------
    galaxy_ids : list
        List of galaxy IDs to analyze
    phot_table : str
        Path to the MIRI photometry table
    data_dir : str
        Directory storing the PROSPECTOR h5 output files
    plot_dir : str, optional
        Directory to store the plots in
    stats_dir : str, optional
        Directory to write the fit statistics to
    add_duste : bool, optional
        Add dust emission to the fits
    n_samples : int, optional
        Number of posterior samples to draw for uncertainty estimation
    run : str, optional
        Accepted run arguments are 'standard' or 'nirspec'
    overwrite : bool, optional
        Whether to overwrite existing pickle files in stats_dir
    """

    print(f"Analysing fits of {len(galaxy_ids)} galaxies...\n")
    
    os.makedirs(stats_dir, exist_ok=True)
    
    for objid in galaxy_ids:
        
        objid = int(objid)
        
        filename = os.path.join(stats_dir, f"{objid}.pkl")
        
        if os.path.exists(filename) and not overwrite:
            print(f"Skipping galaxy {objid} - output file already exists!")
            continue
        
        elif os.path.exists(filename) and overwrite:
            print(f"Overwriting existing output for galaxy {objid}...")
        
        print(f"============ Processing galaxy {objid} ================")
        
        # ------------------------------------------------------------------
        # Section 1: Getting the best-fit model
        # ------------------------------------------------------------------
        
        # Load the h5 file for the given objid
        h5_file = glob.glob(os.path.join(data_dir, f"output_{objid}*.h5"))
        
        try:
            h5_file = h5_file[0]
        except IndexError:
            print(f"No PROSPECTOR results found for objid {objid}.")
            continue

        # Load PROSPECTOR results
        full_path = os.path.join(data_dir, h5_file)
        results, obs, model = reader.results_from(full_path)
        
        # Now we have to exclude the last 3 parameters from the fit
        map_parameters = get_MAP(results)
        
        # Build the MAP dictionary
        MAP = {}
        for a,b in zip(results['theta_labels'], map_parameters):
            MAP[a] = b

        zred = MAP['zred']
        logmass = MAP['logmass']
        dust2 = MAP['dust2']    # extract the diffuse dust V-band optical depth
        
        ##########################################
        #
        # Section 1: Rebuilding the PROSPECTOR fit
        #
        ##########################################
        
        # Calculate the spectrum based on the Maximum A Posteriori (MAP) parameters
        sps = FastStepBasis(zcontinuous=1)
        
        if run == 'nirspec':
            model = rebuild_model(zred, add_neb=True)
            
            # Add model to the results variable
            results['model'] = model
            
            # Keys that relate to the 1D spectrum
            spec_keys = ['wavelength', 'spectrum', 'unc', 'mask', 'sigma_v']

            for key in spec_keys:
                if key in obs:
                    obs[key] = None

            # Optional: ensure logify_spectrum is False if you aren't using a spectrum
            obs['logify_spectrum'] = False

        elif model == None:
            print("⚠️ Warning: No model found in the results. Rebuilding the model from parameters...")
            model = dict(results['model_params'][0])
            model = PolySpecModel(model)  
            
        # Decide on whether to include dust emission or not
        if add_duste:
            # Obtain best fit model spectrum and model photometry    
            spec, phot, _ = model.predict(map_parameters, obs=obs, sps=sps)
        else:
            print("⚠️ Dust emission is turned off.")
        
            # 1. Create a copy of your MAP parameters
            no_dust_params = map_parameters.copy()

            # 2. Toggle the model setting to False
            # This prevents the code from adding the IR 'glow'
            model.params['add_dust_emission'] = np.array([False])

            # 3. Predict the spectrum
            # The resulting spec/phot will show the attenuated UV but NO IR emission
            spec, phot, _ = model.predict(no_dust_params, obs=obs, sps=sps)    
        
        # Convert maggies to µJy
        maggies_to_muJy = 3631e6
        
        # Wavelengths of the model spectrum
        wave_spec = sps.wavelengths
        
        # Convert to arrays
        phot = np.array(phot)
        
        # Make sure everything is scaled properly for NIRSpec if needed
        if run == 'nirspec':
            # Make sure to scale the spectra and photometry accordingly
            cat_path = '/Users/benjamincollins/Data/Bluejay/combined_catalog_v2.0.4.fits'
            cat_table = Table.read(cat_path)
    
            mask = cat_table['ID'] == objid
            result = cat_table['slit_flux_fraction_F444W'][mask]
            slit_flux_fraction = result.value[0]
        
            # Scale photometry
            phot /= slit_flux_fraction
            obs['maggies'] /= slit_flux_fraction
            obs['maggies_unc'] /= slit_flux_fraction
            
            # Scale spectroscopy
            spec /= slit_flux_fraction
        
        # Compute best-fit IR luminosity (scaled if NIRSpec)
        lir_best = _integrate_ir_lum(spec, wave_spec, zred)
        log_lir_best = np.log10(lir_best) if lir_best > 0 else np.nan
                
        sample_specs = []
        phot_miri_samples = []
        lir_samples = []
        
        # Extend the obs dictionary with MIRI photometry for plotting
        obs_miri = update_obs_with_miri(objid, obs, phot_table)
        filters = obs['filters']
        miri_filters = obs_miri['filters_miri']
        miri_band_names = obs_miri['filter_code_miri']
        
        # Draw weighted posterior samples
        samples = posterior_samples(results, n_samples)
        theta_names = list(results['theta_labels'])
        
        for params_i in samples:
            spec_i, _, _ = model.predict(params_i, obs=obs, sps=sps)
            if run == 'nirspec':
                spec_i /= slit_flux_fraction
            sample_specs.append(spec_i)
            
            # Integrate L_IR for each individual sample spectra
            lir_i = _integrate_ir_lum(spec_i, wave_spec, zred)
            lir_samples.append(lir_i)
            
            # Compute predicted photometry for MIRI bands for each sample
            phot_i = get_model_photometry(spec_i, wave_spec, miri_filters, zred)
            phot_miri_samples.append(phot_i)
        
        sample_specs = np.array(sample_specs)  # shape: (nsample, nwave)
        lir_samples = np.array(lir_samples)  # shape: (nsample,)
                
        # Takes the per-pixel percentiles such that the final spectra are not actual spectra of Prospectors parameter space
        lower = np.percentile(sample_specs, 16, axis=0)
        median = np.percentile(sample_specs, 50, axis=0)
        upper = np.percentile(sample_specs, 84, axis=0)    
        
        # Identify valid MIRI bands from the attached table
        miri_obs_flux = []
        miri_obs_err  = []
        
        miri_flux_all = obs_miri['maggies_miri'] 
        miri_err_all  = obs_miri['maggies_unc_miri']
        
        for f_o, e_o in zip(miri_flux_all, miri_err_all):
            if np.isfinite(f_o) and np.isfinite(e_o) and (f_o > 0) and (e_o > 0):
                miri_obs_flux.append(f_o)
                miri_obs_err.append(e_o)
        
        miri_obs_flux = np.array(miri_obs_flux)
        miri_obs_err  = np.array(miri_obs_err)
        
        # Compute filter wavelength in microns
        phot_wave      = np.array([filt.wave_effective for filt in filters])  # in Angstroms
        phot_wave_miri = np.array([filt.wave_effective for filt in miri_filters])  # in Angstroms            
        
        # ------------------------------------------------------------------
        # Section 2: Pulling weighted samples and storing the parameters
        # ------------------------------------------------------------------
        
        # Compute log10(L_IR) for valid samples (L_IR > 0)
        valid_lir = lir_samples > 0
        log_lir_samples = np.full(n_samples, np.nan)
        log_lir_samples[valid_lir] = np.log10(lir_samples[valid_lir])
        
        # Get predicted photometry for MIRI bands + errors from the posterior samples
        phot_miri_best   = get_model_photometry(spec, wave_spec, miri_filters, zred)
        phot_miri_med = np.nanpercentile(phot_miri_samples, 50, axis=0)
        phot_miri_16 = np.nanpercentile(phot_miri_samples, 16, axis=0)
        phot_miri_84 = np.nanpercentile(phot_miri_samples, 84, axis=0)
        phot_miri_err   = 0.5*(phot_miri_84 - phot_miri_16)          
                    
        print("Successfully reconstructed fit...")
        
        # ------------------------------------------------------------------
        # Section 3: Reconstruct SFH
        # ------------------------------------------------------------------
        
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
        
        # Select bins younger than a certain timescale
        t100 = 1e8  # 100 Myr in years
        t30 = 3e7   #  30 Myr in years
        
        # Compute overlap of each bin with interval [0, tcut]
        overlap100 = np.maximum(0.0, np.minimum(bin_edges[:,1], t100) - np.minimum(bin_edges[:,0], t100))
        overlap30 = np.maximum(0.0, np.minimum(bin_edges[:,1], t30) - np.minimum(bin_edges[:,0], t30))
        
        # For bins that are fully within [0,tcut] overlap == dt, partial bins get partial dt
        sfr_100_best = np.sum(sfrs * overlap100) / t100
        sfr_30_best = np.sum(sfrs * overlap30) / t30
        
        # Convert to log10(SFR)
        log_sfr_100_best = np.log10(sfr_100_best) if sfr_100_best > 0 else np.nan
        log_sfr_30_best  = np.log10(sfr_30_best)  if sfr_30_best  > 0 else np.nan

        log_ssfr_100_best = log_sfr_100_best - logmass if np.isfinite(log_sfr_100_best) else np.nan
        log_ssfr_30_best  = log_sfr_30_best  - logmass if np.isfinite(log_sfr_30_best)  else np.nan
        
        # ----------------------------------------------------------------------
        # Compute SFR distribution across posterior samples
        # ----------------------------------------------------------------------
        
        # Map theta indices for fast lookup
        theta_names = results['theta_labels']
        fit_params_stats = {}
        param_chains = {}
        
        for idx, param_name in enumerate(theta_names):
            # Exclude SFR ratio parameters
            if param_name.startswith('logsfr_ratios'):
                continue
            
            chain = samples[:, idx]
            map_val = MAP.get(param_name, np.nan)
            
            if np.isnan(map_val):
                print(f"⚠️ Warning: MAP value for parameter '{param_name}' is NaN. Check the results.")
            
            # Save the 1D sample vector (shape: n_samples,)
            param_chains[param_name] = chain
            
            # Save quantiles, uncertainties, and MAP estimate
            fit_params_stats[param_name] = summarise_chain(chain, map_value=map_val)
        
        logmass_idx = theta_names.index('logmass')
        dust2_idx = theta_names.index('dust2')
        ratio_indices = [
            i for i, name in enumerate(theta_names) 
            if name.startswith('logsfr_ratios_')
        ]

        log_sfr_100_samples = []
        log_sfr_30_samples = []
        
        log_ssfr_100_samples = []
        log_ssfr_30_samples = []
        
        dust2_samples = []

        for s_params in samples:
            s_logm = s_params[logmass_idx]
            s_r = s_params[ratio_indices]
                        
            # Get SFRs for each sample using the same age bins
            s_bins = logsfr_ratios_to_sfrs(s_logm, s_r, agebins)
            s_sfr_100 = np.sum(s_bins * overlap100) / t100
            s_sfr_30  = np.sum(s_bins * overlap30) / t30
            
            s_log_sfr_100 = np.log10(s_sfr_100) if s_sfr_100 > 0 else np.nan
            s_log_sfr_30  = np.log10(s_sfr_30)  if s_sfr_30  > 0 else np.nan
            
            log_sfr_100_samples.append(s_log_sfr_100)
            log_sfr_30_samples.append(s_log_sfr_30)

            # Specific starformation rate (sSFR) = SFR / stellar mass
            s_log_ssfr_100 = s_log_sfr_100 - s_logm if np.isfinite(s_log_sfr_100) else np.nan
            s_log_ssfr_30 = s_log_sfr_30 - s_logm if np.isfinite(s_log_sfr_30) else np.nan
            
            log_ssfr_100_samples.append(s_log_ssfr_100)
            log_ssfr_30_samples.append(s_log_ssfr_30)
            
        log_sfr_100_samples = np.array(log_sfr_100_samples)
        log_sfr_30_samples = np.array(log_sfr_30_samples)
        
        log_ssfr_100_samples = np.array(log_ssfr_100_samples)
        log_ssfr_30_samples = np.array(log_ssfr_30_samples)
        
        # Convert dust2 to A_V
        dust2_samples = 1.086 * samples[:, dust2_idx]

        # Add derived chains
        derived_chains = {
            'log_sfr_100myr':  log_sfr_100_samples,
            'log_sfr_30myr':   log_sfr_30_samples,
            'log_ssfr_100myr': log_ssfr_100_samples,
            'log_ssfr_30myr':  log_ssfr_30_samples,
            'A_V':             dust2_samples,
            'log_L_ir':        log_lir_samples  # From the sample-by-sample integration loop
        }
        
        # Add derived stats including percentiles
        derived_stats = {
            'log_sfr_100myr':  summarise_chain(log_sfr_100_samples, map_value=log_sfr_100_best),
            'log_sfr_30myr':   summarise_chain(log_sfr_30_samples, map_value=log_sfr_30_best),
            'log_ssfr_100myr': summarise_chain(log_ssfr_100_samples, map_value=log_ssfr_100_best),
            'log_ssfr_30myr':  summarise_chain(log_ssfr_30_samples, map_value=log_ssfr_30_best),
            'A_V':             summarise_chain(dust2_samples, map_value=1.086 * dust2),
            'log_L_ir':        summarise_chain(log_lir_samples, map_value=log_lir_best)
        }
    
        print("Extracted galaxy properties...")
        
        # ------------------------------------------------------------------
        # Section 4: Get fit quality statistics for each band
        # ------------------------------------------------------------------
        fit_quality = {}
        
        # Compute N_sigma
        delta = phot_miri_med - miri_obs_flux # change sign here
        tot_err = np.sqrt(phot_miri_err**2 + miri_obs_err**2)
        
        # Compute N_sigma only for valid errors to avoid division by zero
        valid_err = tot_err > 0
        N_sigma = np.full_like(delta, np.nan)
        N_sigma[valid_err] = delta[valid_err] / tot_err[valid_err]
        
        # Compute reduced chi-squared for MIRI bands
        chi2_red_miri = np.nanmean(N_sigma**2) if np.any(np.isfinite(N_sigma)) else np.nan        
        fit_quality['chi2_red'] = chi2_red_miri
        
        for i, band in enumerate(miri_band_names):
                        
            f_o = miri_obs_flux[i]
            e_o = miri_obs_err[i]
            f_m = phot_miri_med[i]
            e_m = phot_miri_err[i]
            
            snr = (f_o / e_o) if e_o > 0 else 0
            
            # Compute (log) ratios and propagate error
            if (f_o > 0) and (f_m > 0) and (e_o > 0) and (e_m > 0):
            
                # Factional variance
                frac_var = (e_m / f_m)**2 + (e_o / f_o)**2
                rel_err = np.sqrt(frac_var)
                
                # Linear flux ratio + error
                ratio = f_m / f_o
                ratio_err = ratio * rel_err
                
                # Logarithmic flux ratio + error
                log_ratio = np.log10(f_m / f_o)    
                log_ratio_err = (1.0 / np.log(10)) * np.sqrt(frac_var)
            else:
                ratio = np.nan
                ratio_err = np.nan
                log_ratio = np.nan
                log_ratio_err = np.nan
            
            fit_quality[band] = {
                'galaxy_id': objid,
                'zred': zred,
                'obs_flux': f_o * maggies_to_muJy,
                'obs_err': e_o * maggies_to_muJy,
                'mod_flux': f_m * maggies_to_muJy,
                'mod_err': e_m * maggies_to_muJy,
                'n_sigma': N_sigma[i],
                'ratio': ratio,
                'ratio_err': ratio_err,
                'log_ratio': log_ratio,
                'log_ratio_err': log_ratio_err,
                'snr': snr
            }            
        
        print("Computed fit quality statistics")
        
        data = {
            # Important metadata
            'id': objid,
            'zred': zred,
            'maggies_to_muJy': maggies_to_muJy,
            
            'model': {
                'spec_best': spec,
                'spec_16th': lower,
                'spec_median': median,
                'spec_84th': upper,
                'wave_spec': wave_spec,
                'sample_specs': sample_specs[:10],
                'phot': phot,
                'phot_wave': phot_wave,
                'phot_miri_best': phot_miri_best,
                'phot_miri_med': phot_miri_med,
                'phot_miri_16': phot_miri_16,
                'phot_miri_84': phot_miri_84,
                'phot_miri_err': phot_miri_err,
                'phot_wave_miri': phot_wave_miri
            },
            
            # One entry for the observation dictionary
            'obs': obs_miri,
            
            # Fit quality statistics for each MIRI band
            'fit_quality': fit_quality,
            
            # Every fitted parameter from theta_labels (dust1_fraction, duste_gamma, etc.)
            'fit_params': fit_params_stats,

            # Derived integrated physical properties
            'derived_props': derived_stats,

            # 1D posterior chains for all parameters & derived quantities
            # ** operator makes sure that all entries are stored at the same level
            'chains': {
                **param_chains,
                **derived_chains
            },
            
            # Moved it outside so it's easier accessible
            'map_theta': MAP, # Keep the full raw dictionary just in case    
            
            # Age bins for SFH
            'agebins': agebins
        }
        
        # Write output to a pickle file
        with open(filename, 'wb') as f:
            pkl.dump(data, f)
        print(f"💾 Saved data to {filename}")
            
        if plot_dir:
            plot_reconstructed_fit(filename, plot_dir)
    
    return 


# --- Helper function to extract and scale spectra ---
def _load_and_extract_spectra(file_path):
    """Loads a Prospector pickle file and extracts scaled best-fit,

    percentile spectra (in uJy), rest-frame wavelengths (in um), and zred.
    """
    with open(file_path, 'rb') as f:
        fit_data = pkl.load(f)

    zred = fit_data['zred']
    maggies_to_muJy = fit_data['maggies_to_muJy']
    model = fit_data['model']
    props = fit_data['galaxy_properties']
    
    wave_spec_um = (model['wave_spec'] * 1e-4) / (1 + zred)  # Rest-frame um

    # Extract & scale spectra in a dictionary
    spectra = {
        'best': model['spec_best'] * maggies_to_muJy,
        '16th': model['spec_16th'] * maggies_to_muJy,
        'median': model['spec_median'] * maggies_to_muJy,
        '84th': model['spec_84th'] * maggies_to_muJy
    }

    return spectra, wave_spec_um, zred, props


# --- Helper function for integration ---
def _integrate_ir_lum(spec_uJy, wave_rest_um, zred):
    """Integrates f_nu over rest-frame 8-1000 um and converts to L_sun."""
    # Mask rest-frame 8 to 1000 um
    mask = (wave_rest_um >= 8.0) & (wave_rest_um <= 1000.0)

    if not np.any(mask):
        return np.nan * u.L_sun

    w_rest = wave_rest_um[mask] * u.um
    f_nu = spec_uJy[mask] * u.uJy

    # Comnvert to rest-frame frequency in Hz
    fq_rest = (const.c / w_rest).to(u.Hz).value

    # Integrate over frequency (reverse for positive dx)
    f_nu_cgs = f_nu.to(u.erg / (u.s * u.cm**2 * u.Hz)).value
    F_ir_rest = trapezoid(f_nu_cgs[::-1], fq_rest[::-1]) * (u.erg / (u.s * u.cm**2))

    # Total integrated IR-flux in the observed frame
    F_ir_obs = F_ir_rest / (1 + zred)  

    # Convert flux to luminosity using luminosity distance
    dL = cosmo.luminosity_distance(zred)
    L_ir = (4.0 * np.pi * dL**2 * F_ir_obs).to(u.L_sun)

    return L_ir.value  # Return as a float in L_sun
    
def summarise_chain(chain_array, map_value=np.nan):
    """Computes 16th, 50th (median), 84th percentiles and asymmetric uncertainties."""
    p16, p50, p84 = np.nanpercentile(chain_array, [16, 50, 84])
    return {
        'map': float(map_value) if np.isfinite(map_value) else np.nan,
        'median': float(p50),
        'p16': float(p16),
        'p84': float(p84),
        'err_low': float(p50 - p16),
        'err_high': float(p84 - p50)
    }
            
            
            
            
    """
    
    
    elif run == 'miri':
                
                # 1. Initialise containers
                miri_filters    = []
                miri_band_names = []
                miri_obs_flux   = []
                miri_obs_err    = []
                phot_miri       = []
                phot_wave_miri  = []
        
                # 2. Iterate and Filter
                for i, filt in enumerate(obs['filters']):
                    # Check for NaN first to clean the entire analysis
                    if np.isnan(obs['maggies'][i]):
                        continue
                    
                    fname = filt.name.lower().strip()
                    if fname in filter_dict_miri:
                        f_val = obs['maggies'][i]
                        e_val = obs['maggies_unc'][i]
                        
                        # Strictly require valid, positive detections
                        if np.isfinite(f_val) and np.isfinite(e_val) and (f_val > 0) and (e_val > 0):
                            miri_filters.append(filt)
                            miri_band_names.append(filter_dict_miri[fname])
                            miri_obs_flux.append(f_val)
                            miri_obs_err.append(e_val)
                            
                            # Extract the exact corresponding predicted model flux from MAP predict
                            phot_miri.append(phot[i])
                            phot_wave_miri.append(filt.wave_effective)
                        
                miri_obs_flux  = np.array(miri_obs_flux)
                miri_obs_err   = np.array(miri_obs_err)
                phot_miri      = np.array(phot_miri)
                phot_wave_miri = np.array(phot_wave_miri)
                        
                # Compute filter wavelength in microns
                phot_wave = np.array([filt.wave_effective for filt in obs['filters']])  # in Angstroms
                
                phot_miri_upper = get_model_photometry(upper, wave_spec, miri_filters, zred)
                phot_miri_lower = get_model_photometry(lower, wave_spec, miri_filters, zred)
                phot_miri_err   = 0.5*(phot_miri_upper - phot_miri_lower)
            
            else:
                print(f"⚠️ Warning: Unknown run type '{run}'. Defaulting to 'standard' behaviour.")  
    
    
    """