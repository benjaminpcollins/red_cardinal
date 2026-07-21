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



def analyse_fits(phot_table, data_dir, plot_dir=None, stats_dir=None, add_duste=True):
    """Main function to reconstruct and plot PROSPECTOR results with MIRI data
    
    Parameters:
    -----------
    phot_table : str
        Path to the MIRI photometry table
    data_dir : str
        Directroy storing the PROSPECTOR h5 output files
    plot_dir : str, optional
        Directory to store the plots in
    stats_dir : str, optional
        Directory to write the fit statistics to
    add_duste : bool, optional
        Add dust emission to the fits
    """
    
    with fits.open(phot_table) as hdul:
        galaxy_ids = hdul[1].data['ID']
    
    print(f"Analysing fits of {len(galaxy_ids)} galaxies...\n")
    
    os.makedirs(stats_dir, exist_ok=True)
    
    for objid in galaxy_ids:
        
        objid = int(objid)
        
        if objid in [7696, 11247]:
            print("Skipping high-z filler target")
            continue    # Skip high-z filler targets
        
        if objid in [12020, 18977]:
            print("Skipping broad-line AGN")
            continue    # Skip broad-line AGN
        
        filename = os.path.join(stats_dir, f"{objid}.pkl")
        
        if os.path.exists(filename):
            print(f"Skipping galaxy {objid} - output file already exists!")
            continue
        
        print(f"============ Processing galaxy {objid} ================")
        
        
        
        ######################################
        #
        # Section 1: Getting galaxy properties
        #
        ######################################
        
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
        
        if model == None:
            model = dict(results['model_params'][0])
            model = PolySpecModel(model)
            
        #print(obs['filters'])
        #return
        
        # Now we have to exclude the last 3 parameters from the fit
        map_parameters = get_MAP(results)
        
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
        
        # Select bins younger than a certain timescale
        t100 = 1e8  # 100 Myr in years
        t30 = 3e7   #  30 Myr in years
        
        # Compute overlap of each bin with interval [0, tcut]
        overlap100 = np.maximum(0.0, np.minimum(bin_edges[:,1], t100) - np.minimum(bin_edges[:,0], t100))
        overlap30 = np.maximum(0.0, np.minimum(bin_edges[:,1], t30) - np.minimum(bin_edges[:,0], t30))
        
        # For bins that are fully within [0,tcut] overlap == dt, partial bins get partial dt
        mass_in_last_100 = np.sum(sfrs * overlap100)
        sfr_last100 = mass_in_last_100 / t100

        mass_in_last_30 = np.sum(sfrs * overlap30)
        sfr_last30 = mass_in_last_30 / t30
        
        print("Extracted galaxy properties...")
        
        ##########################################
        #
        # Section 2: Rebuilding the PROSPECTOR fit
        #
        ##########################################
        
        # Calculate the spectrum based on the Maximum A Posteriori (MAP) parameters
        sps = FastStepBasis(zcontinuous=1)

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
        
        # Draw 100 posterior samples
        samples = posterior_samples(results, 100)
        
        sample_specs = []
        for params_i in samples:
            spec_i, _, _ = model.predict(params_i, obs=obs, sps=sps)
            sample_specs.append(spec_i)
        sample_specs = np.array(sample_specs)  # shape: (nsample, nwave)
        
        # Takes the per-pixel percentiles such that the final spectra are not actual spectra of Prospectors parameter space
        lower = np.percentile(sample_specs, 16, axis=0)
        median = np.percentile(sample_specs, 50, axis=0)
        upper = np.percentile(sample_specs, 84, axis=0)
        
        # Extend the obs dictionary with MIRI photometry for plotting
        obs_miri = update_obs_with_miri(objid, obs, phot_table)
        miri_filters = obs_miri['filters_miri']
        
        # Get predicted photometry for MIRI bands (+ Errors)
        phot_miri = get_model_photometry(spec, wave_spec, miri_filters, zred)
        phot_miri_upper = get_model_photometry(upper, wave_spec, miri_filters, zred)
        phot_miri_lower = get_model_photometry(lower, wave_spec, miri_filters, zred)
        phot_miri_err = 0.5*(phot_miri_upper - phot_miri_lower)
        
        # Compute filter wavelength in microns
        phot_wave = np.array([filt.wave_effective for filt in obs['filters']])  # in Angstroms
        phot_wave_all = np.array([filt.wave_effective for filt in obs_miri['filters_all']])  # in Angstroms
        phot_wave_miri = np.array([filt.wave_effective for filt in obs_miri['filters_miri']])  # in Angstroms
        
        print("Successfully reconstructed fit...")
        
        ##########################################
        #
        # Section 3: Calculating fit quality stats
        #
        ##########################################
        
        # Safer way to ensure you align with phot_miri
        n_orig = len(obs['filters'])
        miri_flux = obs_miri['maggies_all'][n_orig:] 
        miri_err  = obs_miri['maggies_unc_all'][n_orig:]
        
        # Compute N_sigma
        delta = miri_flux - phot_miri
        tot_err = np.sqrt(phot_miri_err**2 + miri_err**2)
        N_sigma = delta / tot_err
        
        # Compute it also in percentage of observed MIRI flux
        perc = delta / miri_flux
        
        chi2_red = np.sum(N_sigma**2) / len(N_sigma)
        
        fit_quality = {}
        fit_quality['chi2_red'] = chi2_red
        
        for i, filt in enumerate(obs_miri['filters_miri']):
            
            # Extracts 'F770W' from 'jwst_f770w'
            name = filt.name.split('_')[-1].upper()
            
            snr = (miri_flux[i] / miri_err[i]) if miri_err[i] > 0 else 0
            
            fit_quality[name] = {
                'galaxy_id': objid,
                'zred': zred,
                'obs_flux': miri_flux[i] * maggies_to_muJy,
                'obs_err': miri_err[i] * maggies_to_muJy,
                'mod_flux': phot_miri[i] * maggies_to_muJy,
                'mod_err': phot_miri_err[i] * maggies_to_muJy,
                'n_sigma': N_sigma[i],
                'frac_diff': perc[i],
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
                'phot_miri': phot_miri,
                'phot_miri_err': phot_miri_err,
                'phot_wave_miri': phot_wave_miri
            },
            
            # One entry for the observation dictionary
            'obs': obs_miri,
            
            'fit_quality': fit_quality,
            
            'galaxy_properties': {
                'logmass': logmass,
                'dust2': dust2,
                'sfr_100myr': sfr_last100,
                'sfr_30myr': sfr_last30,
                'sfr_bins': sfrs,
                'agebins': agebins
            },
            
            # Moved it outside so it's easier accessible
            'map_theta': MAP # Keep the full raw dictionary just in case    
        }
        
        # Write output to a pickle file
        with open(filename, 'wb') as f:
            pkl.dump(data, f)
        print(f"💾 Saved data to {filename}")
            
        if plot_dir:
            plot_reconstructed_fit(filename, plot_dir)
    
    return 




def analyse_miri_fits(phot_table, data_dir, plot_dir=None, stats_dir=None, add_dust=True):
    """Main function to reconstruct and plot PROSPECTOR results with MIRI data
    
    Parameters:
    -----------
    phot_table : str
        Path to the MIRI photometry table
    data_dir : str
        Directroy storing the PROSPECTOR h5 output files
    plot_dir : str, optional
        Directory to store the plots in
    stats_dir : str, optional
        Directory to write the fit statistics to
    """
    
    with fits.open(phot_table) as hdul:
        galaxy_ids = hdul[1].data['ID']
    
    #galaxy_ids = [17517]
    
    print(f"Analysing fits of {len(galaxy_ids)} galaxies...\n")
    
    os.makedirs(stats_dir, exist_ok=True)
    
    for objid in galaxy_ids:
        
        objid = int(objid)
        
        if objid in [7696, 11247]:
            continue    # Skip high-z filler targets
        
        if objid in [12020, 18977]:
            continue    # Skip broad-line AGN
        
        filename = os.path.join(stats_dir, f"{objid}.pkl")
        
        #if os.path.exists(filename):
        #    print(f"Skipping galaxy {objid} - output file already exists!")
        #    continue
        
        #if os.path.exists(filename):
        #    print(f"Skipping galaxy {objid} - output file already exists!")
        #    continue
        
        print(f"============ Processing galaxy {objid} ================")
        
        
        
        ######################################
        #
        # Section 1: Getting galaxy properties
        #
        ######################################
        
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
        
        # Select bins younger than a certain timescale
        t100 = 1e8  # 100 Myr in years
        t30 = 3e7   #  30 Myr in years
        
        # Compute overlap of each bin with interval [0, tcut]
        overlap100 = np.maximum(0.0, np.minimum(bin_edges[:,1], t100) - np.minimum(bin_edges[:,0], t100))
        overlap30 = np.maximum(0.0, np.minimum(bin_edges[:,1], t30) - np.minimum(bin_edges[:,0], t30))
        
        # For bins that are fully within [0,tcut] overlap == dt, partial bins get partial dt
        mass_in_last_100 = np.sum(sfrs * overlap100)
        sfr_last100 = mass_in_last_100 / t100

        mass_in_last_30 = np.sum(sfrs * overlap30)
        sfr_last30 = mass_in_last_30 / t30
        
        print("Extracted galaxy properties...")
        
        ##########################################
        #
        # Section 2: Rebuilding the PROSPECTOR fit
        #
        ##########################################
        
        # Calculate the spectrum based on the Maximum A Posteriori (MAP) parameters
        sps = FastStepBasis(zcontinuous=1)
        
        if add_dust == False:
            # Toggle the model setting to False
            # This prevents the code from adding the IR 'glow'
            model.params['add_dust_emission'] = np.array([False])

        # Obtain best fit model spectrum and model photometry    
        spec, phot, _ = model.predict(map_parameters, obs=obs, sps=sps)
        
        # Convert maggies to µJy
        maggies_to_muJy = 3631e6
        
        # Wavelengths of the model spectrum
        wave_spec = sps.wavelengths
        
        # Convert to arrays
        phot = np.array(phot)
        
        # Draw 100 posterior samples
        samples = posterior_samples(results, 100)
        
        sample_specs = []
        for params_i in samples:    
            spec_i, _, _ = model.predict(params_i, obs=obs, sps=sps)
            sample_specs.append(spec_i)
        sample_specs = np.array(sample_specs)  # shape: (nsample, nwave)
        
        # Takes the per-pixel percentiles such that the final spectra are not actual spectra of Prospectors parameter space
        lower = np.percentile(sample_specs, 16, axis=0)
        median = np.percentile(sample_specs, 50, axis=0)
        upper = np.percentile(sample_specs, 84, axis=0)
        
        # Compute filter wavelength in microns
        phot_wave = np.array([filt.wave_effective for filt in obs['filters']])  # in Angstroms
        
        filter_dict_miri = {
            'F770W':  'jwst_f770w',
            'F1000W': 'jwst_f1000w',
            'F1800W': 'jwst_f1800w',
            'F2100W': 'jwst_f2100w'
        }
        
        # 1. Initialize containers
        miri_mask = []        # Indices in the 'obs' arrays that are MIRI and not NaN
        miri_band_names = []  # To keep track of which specific band is at which index
        miri_filters = []

        # 2. Iterate and Filter
        for i, filt in enumerate(obs['filters']):
            # Check for NaN first to clean the entire analysis
            if np.isnan(obs['maggies'][i]):
                continue
                
            # Check if this filter is in our MIRI dictionary
            if filt.name in filter_dict_miri.values():
                miri_mask.append(i)
                
                # Map back to the short name (F770W, etc.) for your stats dict
                for code, sedpy_name in filter_dict_miri.items():
                    if sedpy_name == filt.name:
                        miri_band_names.append(code)
                        miri_filters.append(filt)
                        break

        # 3. Compute Separate Statistics
        # Extract only the valid MIRI data
        phot_miri = phot[miri_mask] # phot is the model photometry from model.predict
        phot_wave_miri = phot_wave[miri_mask]
        
        phot_miri_upper = get_model_photometry(upper, wave_spec, miri_filters, zred)
        phot_miri_lower = get_model_photometry(lower, wave_spec, miri_filters, zred)
        phot_miri_err = 0.5*(phot_miri_upper - phot_miri_lower)
        
        print("Successfully reconstructed fit...")
        
        ##########################################
        #
        # Section 3: Calculating fit quality stats
        #
        ##########################################
        
        # Safer way to ensure you align with phot_miri
        miri_flux = obs['maggies'][miri_mask]
        miri_err  = obs['maggies_unc'][miri_mask]
        
        # Compute N_sigma
        delta = miri_flux - phot_miri
        tot_err = np.sqrt(phot_miri_err**2 + miri_err**2)
        N_sigma = delta / tot_err
        
        # Compute it also in percentage of observed MIRI flux
        perc = delta / miri_flux
        
        chi2_red = np.sum(N_sigma**2) / len(N_sigma)
        
        fit_quality = {}
        fit_quality['chi2_red'] = chi2_red
        
        for i, filt in enumerate(miri_filters):
            
            # Extracts 'F770W' from 'jwst_f770w'
            name = filt.name.split('_')[-1].upper()
            
            snr = (miri_flux[i] / miri_err[i]) if miri_err[i] > 0 else 0
            
            fit_quality[name] = {
                'galaxy_id': objid,
                'zred': zred,
                'obs_flux': miri_flux[i] * maggies_to_muJy,
                'obs_err': miri_err[i] * maggies_to_muJy,
                'mod_flux': phot_miri[i] * maggies_to_muJy,
                'mod_err': phot_miri_err[i] * maggies_to_muJy,
                'n_sigma': N_sigma[i],
                'frac_diff': perc[i],
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
                'phot_miri': phot_miri,
                'phot_miri_err': phot_miri_err,
                'phot_wave_miri': phot_wave_miri
            },
            
            # One entry for the observation dictionary
            'obs': obs,
            
            'fit_quality': fit_quality,
            
            'galaxy_properties': {
                'logmass': logmass,
                'dust2': dust2,
                'sfr_100myr': sfr_last100,
                'sfr_30myr': sfr_last30,
                'sfr_bins': sfrs,
                'agebins': agebins
            },
            
            # Moved it outside so it's easier accessible
            'map_theta': MAP # Keep the full raw dictionary just in case    
        }
        
        # Write output to a pickle file
        with open(filename, 'wb') as f:
            pkl.dump(data, f)
        print(f"💾 Saved data to {filename}")
            
        if plot_dir:
            plot_miri_fit(filename, plot_dir)
 



def analyse_photspec_fits(phot_table, data_dir, plot_dir=None, stats_dir=None):
    """Main function to reconstruct and plot PROSPECTOR results with MIRI data
    
    Parameters:
    -----------
    phot_table : str
        Path to the MIRI photometry table
    data_dir : str
        Directroy storing the PROSPECTOR h5 output files
    plot_dir : str, optional
        Directory to store the plots in
    stats_dir : str, optional
        Directory to write the fit statistics to
    """
    
    with fits.open(phot_table) as hdul:
        galaxy_ids = hdul[1].data['ID']
    
    print(f"Analysing fits of {len(galaxy_ids)} galaxies...\n")
    
    os.makedirs(stats_dir, exist_ok=True)
    
    #galaxy_ids = [17517]
    
    for objid in galaxy_ids:
        
        objid = int(objid)
        
        if objid in [7696, 11247]:
            continue    # Skip high-z filler targets
        
        if objid in [12020, 18977]:
            continue    # Skip broad-line AGN
        
        filename = os.path.join(stats_dir, f"{objid}.pkl")
        
        if os.path.exists(filename):
            print(f"Skipping galaxy {objid} - output file already exists!")
            continue
        
        print(f"============ Processing galaxy {objid} ================")
        
        ######################################
        #
        # Section 1: Getting galaxy properties
        #
        ######################################
        
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
        mass_in_last_100 = np.sum(sfrs * overlap100)
        sfr_last100 = mass_in_last_100 / t100

        mass_in_last_30 = np.sum(sfrs * overlap30)
        sfr_last30 = mass_in_last_30 / t30
        
        print("Extracted galaxy properties...")
        
        ##########################################
        #
        # Section 2: Rebuilding the PROSPECTOR fit
        #
        ##########################################
        
        # Calculate the spectrum based on the Maximum A Posteriori (MAP) parameters
        sps = FastStepBasis(zcontinuous=1)

        model = rebuild_model(zred, add_neb=True)
        
        # Add model to the results variable
        results['model'] = model
        
        #print(model)
        #print(MAP)

        #map_theta = results['theta_labels']

        #print(MAP.keys())

        #print(f"Theta length: {len(map_theta)}")
        #print(f"Model ndim: {model.ndim}")
        #print(f"Model free parameters: {model.free_params}")    

        # Keys that relate to the 1D spectrum
        spec_keys = ['wavelength', 'spectrum', 'unc', 'mask', 'sigma_v']

        for key in spec_keys:
            if key in obs:
                obs[key] = None

        # Optional: ensure logify_spectrum is False if you aren't using a spectrum
        obs['logify_spectrum'] = False
        
        # Obtain best fit model spectrum and model photometry    
        spec, phot, _ = model.predict(map_parameters, obs=obs, sps=sps)
        
        # Convert maggies to µJy
        maggies_to_muJy = 3631e6
        
        # Wavelengths of the model spectrum
        wave_spec = sps.wavelengths
        
        # Convert to arrays
        phot = np.array(phot)
                
        # Draw 100 posterior samples
        samples = posterior_samples(results, 100)
        
        sample_specs = []
        for params_i in samples:
            spec_i, _, _ = model.predict(params_i, obs=obs, sps=sps)
            sample_specs.append(spec_i)
        sample_specs = np.array(sample_specs)  # shape: (nsample, nwave)
        
        # Takes the per-pixel percentiles such that the final spectra are not actual spectra of Prospectors parameter space
        lower = np.percentile(sample_specs, 16, axis=0)
        median = np.percentile(sample_specs, 50, axis=0)
        upper = np.percentile(sample_specs, 84, axis=0)
        
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
        sample_specs /= slit_flux_fraction
        
        lower /= slit_flux_fraction
        median /= slit_flux_fraction
        upper /= slit_flux_fraction
        
        # Extend the obs dictionary with MIRI photometry for plotting
        obs_miri = update_obs_with_miri(objid, obs, phot_table)
        miri_filters = obs_miri['filters_miri']
        
        # Get predicted photometry for MIRI bands (+ Errors)
        phot_miri = get_model_photometry(spec, wave_spec, miri_filters, zred)
        phot_miri_upper = get_model_photometry(upper, wave_spec, miri_filters, zred)
        phot_miri_lower = get_model_photometry(lower, wave_spec, miri_filters, zred)
        phot_miri_err = 0.5*(phot_miri_upper - phot_miri_lower)
        
        # Compute filter wavelength in microns
        phot_wave = np.array([filt.wave_effective for filt in obs['filters']])  # in Angstroms
        phot_wave_all = np.array([filt.wave_effective for filt in obs_miri['filters_all']])  # in Angstroms
        phot_wave_miri = np.array([filt.wave_effective for filt in obs_miri['filters_miri']])  # in Angstroms
        
        print("Successfully reconstructed fit...")
        
        ##########################################
        #
        # Section 3: Calculating fit quality stats
        #
        ##########################################
        
        # Safer way to ensure you align with phot_miri
        n_orig = len(obs['filters'])
        miri_flux = obs_miri['maggies_all'][n_orig:] 
        miri_err  = obs_miri['maggies_unc_all'][n_orig:]
        
        # Compute N_sigma
        delta = miri_flux - phot_miri
        tot_err = np.sqrt(phot_miri_err**2 + miri_err**2)
        N_sigma = delta / tot_err
        
        # Compute it also in percentage of observed MIRI flux
        perc = delta / miri_flux
        
        chi2_red = np.sum(N_sigma**2) / len(N_sigma)
        
        fit_quality = {}
        fit_quality['chi2_red'] = chi2_red
        
        for i, filt in enumerate(obs_miri['filters_miri']):
            
            # Extracts 'F770W' from 'jwst_f770w'
            name = filt.name.split('_')[-1].upper()
            
            snr = (miri_flux[i] / miri_err[i]) if miri_err[i] > 0 else 0
            
            fit_quality[name] = {
                'galaxy_id': objid,
                'zred': zred,
                'obs_flux': miri_flux[i] * maggies_to_muJy,
                'obs_err': miri_err[i] * maggies_to_muJy,
                'mod_flux': phot_miri[i] * maggies_to_muJy,
                'mod_err': phot_miri_err[i] * maggies_to_muJy,
                'n_sigma': N_sigma[i],
                'frac_diff': perc[i],
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
                'phot_miri': phot_miri,
                'phot_miri_err': phot_miri_err,
                'phot_wave_miri': phot_wave_miri
            },
            
            # One entry for the observation dictionary
            'obs': obs_miri,
            
            'fit_quality': fit_quality,
            
            'galaxy_properties': {
                'logmass': logmass,
                'dust2': dust2,
                'sfr_100myr': sfr_last100,
                'sfr_30myr': sfr_last30,
                'sfr_bins': sfrs,
                'agebins': agebins
            },
            
            # Moved it outside so it's easier accessible
            'map_theta': MAP # Keep the full raw dictionary just in case    
        }
        
        # Write output to a pickle file
        with open(filename, 'wb') as f:
            pkl.dump(data, f)
        print(f"💾 Saved data to {filename}")
            
        if plot_dir:
            plot_reconstructed_fit(filename, plot_dir)
    
    return 

def get_dust_luminosity(objid, data_dir, plot_dir=None):
    
    dust = os.path.join(data_dir, "pickle_files", f"{objid}.pkl")
    nodust = os.path.join(data_dir, "pickle_nodust", f"{objid}.pkl")
    
    
    with open(dust, 'rb') as f:
        fit_data = pkl.load(f)

    zred = fit_data['zred']
    maggies_to_muJy = fit_data['maggies_to_muJy']
    
    model_dust = fit_data['model']
    spec_best = model['spec_best']
    spec_16th = model['spec_16th']
    spec_median = model['spec_median']
    spec_84th = model['spec_84th']
    
    spec_dust = model_dust['spec_best']
    wave_spec = model_dust['wave_spec']
    
    # Convert to µJy
    spec_dust_scaled = spec_dust * maggies_to_muJy
    
    wave_spec_um = wave_spec * 1e-4
    
    # No dust case
    with open(nodust, 'rb') as f:
        fit_data = pkl.load(f)

    zred = fit_data['zred']
    
    model_nodust = fit_data['model']
    spec_nodust = model_nodust['spec_best']
    
    # Convert to µJy
    spec_nodust_scaled = spec_nodust * maggies_to_muJy
    
    
    spec_ir = spec_dust_scaled - spec_nodust_scaled
    
    # Specify luminosity between 8 and 1000µm
    wave_mask = (wave_spec_um >= 8.0) & (wave_spec_um <= 1000.0)
    
    # Apply mask to spectrum(s)
    spec_within = spec_ir[wave_mask]  # works for 1D or 2D (e.g. percentiles)
    
    w_rest_slice = wave_spec_um[wave_mask] * u.um
    f_nu_slice = spec_ir[wave_mask] * u.uJy

    # 3. Convert wavelengths to frequency (Hz)
    # Nu = c / lambda
    freq_slice = (const.c / w_rest_slice).to(u.Hz)

    # 4. Integrate f_nu over frequency (d_nu)
    # Note: since frequency decreases as wavelength increases, reverse arrays for positive integral
    F_ir = trapezoid(f_nu_slice.to(u.erg / (u.s * u.cm**2 * u.Hz)).value[::-1], 
                    freq_slice.value[::-1]) * (u.erg / (u.s * u.cm**2))

    # 5. Convert Flux to Luminosity using Luminosity Distance D_L
    dL = cosmo.luminosity_distance(zred)

    # Bolometric Luminosity formula
    L_ir = (4 * np.pi * dL**2 * F_ir * (1 + zred)).to(u.L_sun)

    print(f"Log(L_IR / L_sun) = {np.log10(L_ir.value):.2f}")
    
    fig, ax = plt.subplots(figsize=(8, 5))


    #w_rest_slice = wave_rest[mask] * u.um
    #f_nu_slice = spec_ir[mask] * u.uJy

    # Compute y-axis limits
    ymin = np.nanmin(spec_within)
    ymax = np.nanmax(spec_within)
    
    # Add margin proportionally, protecting against log-scale issues
    ymin_plot = ymin * 0.2  # reduce, but stay > 0
    ymax_plot = ymax * 5   # increase

    # Set limits
    ax.set_ylim(ymin_plot, ymax_plot)

    # Plot formatting
    ax.set_xlabel('Restframe Wavelength [µm]', fontsize=13)
    ax.set_ylabel('Flux [µJy]', fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(wave_spec_um, spec_ir)
    plt.show()
    
    
    return 