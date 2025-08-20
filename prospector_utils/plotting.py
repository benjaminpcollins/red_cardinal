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

# This script is designed to work with PROSPECTOR results and MIRI photometry data.

dirout = "/Users/benjamincollins/University/master/Red_Cardinal/prospector/outputs/"


def reconstruct(objid, plot_dir=None, stats_dir=None, add_duste=True):
    """Main function to reconstruct and plot PROSPECTOR results with MIRI data"""
    
    # Load the h5 file for the given objid
    h5_file = glob.glob(os.path.join(dirout, f"output_{objid}*.h5"))
    
    try:
        h5_file = h5_file[0]
    except IndexError:
        print(f"No PROSPECTOR results found for objid {objid}.")
        return None

    # Load PROSPECTOR results
    full_path = os.path.join(dirout, h5_file)
    results, loaded_obs, loaded_model = reader.results_from(full_path)
    
    # Build new observations including MIRI
    obs = build_obs(objid)
    
    # Now we have to exclude the last 3 parameters from the fit
    map_parameters = get_MAP(results)
    
    original_theta_labels = results['theta_labels']
    original_map_parameters = map_parameters.copy()  # Keep original for reference
    
    # Build the MAP dictionary
    MAP = {}
    for a,b in zip(results['theta_labels'], map_parameters):
        MAP[a] = b
    
    # Section to decide whether to include dust emission or not
    if add_duste == True:
        map_parameters = map_parameters[:-3]
        add_duste = results['run_params']['add_duste']
        add_agn = results['run_params']['add_agn']  # Maybe check if this is True?
        suffix = None
    else:
        map_parameters = map_parameters[:-6]
        add_agn = False
        suffix = "nodust"
        
    # Get accurate redshifts from the MAP (obtained by MJ)    
    zred = map_parameters[0]
    print(f"Processing galaxy {objid} at redshift z = {zred}")

    # Build model matching original setup    
    # Somehow this works if zred=objid and waverange=zred, but not if I pass the arguments correctly
    model = build_model(zred=zred,
                        waverange=None,
                        add_duste=add_duste,
                        add_neb=False,  # Set add_neb to False
                        add_agn=add_agn,
                        fit_afe=results['run_params']['fit_afe']
                        )
    
    model.params['polyorder'] = 10

    print("New model.ndim:", model.ndim)
    print("New model.theta_index:", model.theta_index)
    print("New model theta_labels:", [model.theta_labels()[i] for i in range(model.ndim)])

    # Map parameters from original fit to new model structure
    new_theta_labels = [model.theta_labels()[i] for i in range(model.ndim)]
    new_map_parameters = []
    
    # Create mapping between old and new parameter structures
    for new_label in new_theta_labels:
        if new_label in original_theta_labels:
            old_idx = original_theta_labels.index(new_label)
            new_map_parameters.append(original_map_parameters[old_idx])
            print(f"Mapped {new_label}: {original_map_parameters[old_idx]}")
        else:
            # This shouldn't happen if we're just removing dust parameters
            print(f"Warning: {new_label} not found in original parameters")
            new_map_parameters.append(0.0)  # Default value

    new_map_parameters = np.array(new_map_parameters)

    print(f"Original parameters length: {len(original_map_parameters)}")
    print(f"New parameters length: {len(new_map_parameters)}")
    print(f"New model expects: {model.ndim} parameters")
    
    #for a,b in zip(results['theta_labels'],map_parameters):
    #    print(a, b)
    
    #print("Adjusted length of map_parameters: ", len(map_parameters), "\n")

    # Calculate the spectrum based on the Maximum A Posteriori (MAP) parameters
    sps = FastStepBasis(zcontinuous=1)

    # Obtain best fit model spectrum and model photometry    
    spec, phot, _ = model.predict(map_parameters, obs=obs, sps=sps)
    
    #get the spectral jitter  for the errors
    #spec_jitter = map_parameters[5]
    #errs = obs['unc']*spec_jitter/calib_vector * 3631 # observational errors
    
    maggies_to_muJy = 3631e6 # maggies to µJy conversion factor
    
    # wavelengths of the model spectrum
    wave_spec = sps.wavelengths
    wave_spec_rs = sps.wavelengths * 1e-4 * (1 + zred)   # convert to µm, redshifted    
    
    # Convert to arrays
    phot = np.array(phot)
    maggies = np.array(obs['maggies'])
    phot_err = np.array(obs['maggies_unc'])

    # Avoid divide-by-zero or NaN issues
    valid = (
        (phot > 0) & 
        (maggies > 0) & 
        np.isfinite(phot) & 
        np.isfinite(maggies) & 
        np.isfinite(phot_err) & 
        (phot_err > 0)
    )

    # Masked, valid values only
    phot = phot[valid]
    maggies = maggies[valid]
    phot_err = phot_err[valid]

    # Compute ratio and weights
    ratio = maggies / phot
    weights = 1.0 / phot_err**2

    # Weighted average correction factor
    corr_factor = np.sum(ratio * weights) / np.sum(weights)

    # Compute uncertainty on correction factor
    residuals = ratio - corr_factor
    weighted_var = np.sum(weights * residuals**2) / np.sum(weights)
    corr_uncertainty = np.sqrt(weighted_var / len(ratio))
    
    print("Mean ratio between photometries: ", corr_factor)
    print("Standard deviation: ", corr_uncertainty)
    
    # Initialise the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    #########   PLOT POSTERIOR SAMPLES     #########
    
    nsample = 100
    samples = posterior_samples(results, nsample)
    
    all_specs = []
    for params_i in samples:
        params_i = params_i[:-3]
        spec_i, _, _ = model.predict(params_i, obs=obs, sps=sps)
        all_specs.append(spec_i)
        
    all_specs = np.array(all_specs)  # shape: (nsample, nwave)
    
    # Takes the per-pixel percentiles such that the final spectra are not actual spectra of Prospectors parameter space
    lower = np.percentile(all_specs, 16, axis=0)
    median = np.percentile(all_specs, 50, axis=0)
    upper = np.percentile(all_specs, 84, axis=0)
    
    # Convert to µJy
    lower_scaled = lower * maggies_to_muJy    
    median_scaled = median * maggies_to_muJy
    upper_scaled = upper * maggies_to_muJy
    
    # Plot shaded region for 1σ uncertainty
    ax.fill_between(wave_spec_rs, lower_scaled, upper_scaled, color='crimson', alpha=0.2, label='1σ uncertainty')
    
    for i in range(10):
        ax.plot(wave_spec_rs, all_specs[i]*maggies_to_muJy, color='crimson', alpha=0.15, lw=0.8)
    
    #ax.plot(wave_spec_rs, lower_scaled, color='blue', lw=0.8, label='16th percentile')
    #ax.plot(wave_spec_rs, upper_scaled, color='blue', lw=0.8, label='84th percentile')
    #########       PLOT THE BEST FIT      #########
    
    spec_scaled = spec * maggies_to_muJy    # convert to µJy
    ax.plot(wave_spec_rs, spec_scaled, '-', color='crimson', alpha=0.8, lw=1.5, label='Best-fit model')
    
    #########   PLOT OBSERVED SPECTRUM     #########
    
    #spectrum  = loaded_obs['spectrum'][mask] / calib_vector[mask] * factor
    
    obs['mask'] = loaded_obs['mask']
    obs['spectrum'] = loaded_obs['spectrum']
    obs['wavelength'] = loaded_obs['wavelength']
    obs['unc'] = loaded_obs['unc']
    
    print("Successfully added observed spectrum to obs file.")
    
    #ax.plot(wavelengths, spectrum, '-', color='blue', alpha=0.8, lw=1.5, label='Observed spectrum')
    
    ######### PLOT THE CALIBRATED SPECTRUM #########
    
    #param_file = '/Users/benjamincollins/University/master/Red_Cardinal/prospector/spec_calib/spec_calibrated_12717.pkl'
    #data = pkl.load(open(param_file, 'rb'))
    #cal_spec = data['emi_off']['MAP']['wave_obs'] * factor
    #lambdas = data['wave_obs'] * 1e-4
    #ax.plot(lambdas, cal_spec, linestyle="-", alpha=0.8, lw=1.5, label="Calibrated spectrum")
    
    #########    PLOT MODEL PHOTOMETRY     #########
    
    wave_phot = np.array([filt.wave_effective for filt in obs['filters']])  # in Angstroms
    wave_phot_microns = wave_phot * 1e-4  # convert to µm
    wave_phot_microns = wave_phot_microns[valid]
    phot_scaled = phot * maggies_to_muJy  # convert maggies to µJy
    ax.plot(wave_phot_microns, phot_scaled, 'd', markersize=6, color='black', label='Model photometry')
    
    #########  PLOT MEASURED PHOTOMETRY    #########

    plot_photometry(ax, obs)
    # Thanks to the function this is literally a one-liner now
    
    # Compute bounds
    wave_mask = (wave_spec_rs >= 0.4) & (wave_spec_rs <= 35)
    
    # Apply mask to spectrum(s)
    spec_within = spec_scaled[wave_mask]  # works for 1D or 2D (e.g. percentiles)
    spec_within = [ele for ele in spec_within if ele > 0]

    # Compute y-axis limits
    ymin = np.nanmin(spec_within)
    ymax = np.nanmax(spec_within)
    
    # Add margin proportionally, protecting against log-scale issues
    ymin_plot = ymin * 0.2  # reduce, but stay > 0
    ymax_plot = ymax * 5   # increase

    # Set limits
    ax.set_ylim(ymin_plot, ymax_plot)

    # Plot formatting
    ax.set_xlabel('Observed Wavelength (µm)')
    ax.set_ylabel('Flux (µJy)')
    ax.set_xlim(0.4, 35)#200)    # Change x range    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    
    zred_rounded = np.round(zred,2)
    plt.title(f"Galaxy {objid} at z={zred_rounded}")
    plt.tight_layout()
    
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        fname = os.path.join(plot_dir, f'{objid}_{suffix}.png')
        plt.savefig(fname)
        print(f"✅ Plot saved to {fname}")
    plt.show()
    plt.close()
    
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)
    
        filename = os.path.join(stats_dir, f"{objid}.pkl")
        
        fit_data = {
            # One entry for the model spectrum + photometry
            'model': {
                'spec_bestfit': spec,
                'spec_16th': lower,
                'spec_50th': median,
                'spec_84th': upper,
                'wave_spec': wave_spec,
                'phot': phot,
                'wave_phot': wave_phot
            },
            
            # One entry for the observation dictionary
            'obs': obs,
            'galaxy_id': objid,
            'redshift': zred,
            'saved_at': datetime.now().isoformat()
        }
        
        # Write output to a pickle file
        with open(filename, 'wb') as f:
            pkl.dump(fit_data, f)

        print(f"✅ Fit results saved to: {filename}")
    

def plot_transmission_curves(ax, filters):
    """Plot the transmission curves of the given filters on the provided axis."""
    # overlay filters
    for f in filters:
        lam = f.wavelength        # wavelength grid (Å)
        trans = f.transmission    # dimensionless throughput (0–1)
        ax.plot(lam, trans, label=f.name)
    
def plot_photometry(ax, obs, factor=3631e6):
    """Plot the photometry data on the provided axis."""
    
    # Define the style per instrument
    instrument_styles = {
        'acs':     {'color': 'royalblue',   'marker': 'o', 'edgecolor': 'black', 'label': 'HST ACS', 'ms': 10},
        'wfc3':    {'color': 'limegreen',  'marker': 'o', 'edgecolor': 'black', 'label': 'HST WFC3', 'ms': 10},
        'nircam':  {'color': 'orange', 'marker': 'p', 'edgecolor': 'black',    'alpha': 0.7, 'label': 'JWST NIRCam', 'ms': 10},
        'miri':    {'color': 'firebrick',    'marker': 'p', 'edgecolor': 'black',    'alpha': 0.7, 'label': 'JWST MIRI (overlaid)', 'ms': 10}
    }
    
    for i, filt in enumerate(obs['filters_all']):
        name = filt.name.lower()

        if 'acs' in name:
            style = instrument_styles['acs']
        elif 'wfc3' in name:
            style = instrument_styles['wfc3']
        elif 'miri' in name or any(m in name for m in ['f770w', 'f1000w', 'f1800w', 'f2100w']):
            style = instrument_styles['miri']
        elif 'nircam' in name or ('jwst' in name and 'f' in name and 'w' in name):
            style = instrument_styles['nircam']
        else:
            continue  # skip unknown filters

        wave = obs['phot_wave_all'][i] * 1e-4  # convert to µm
        flux = obs['maggies_all'][i] * factor  # µJy
        err  = obs['maggies_unc_all'][i] * factor  # µJy
        
        ax.errorbar(
            wave, flux, yerr=err,
            fmt=style['marker'],
            color=style['color'],
            markeredgecolor=style.get('edgecolor', 'none'),
            alpha=style.get('alpha', 1.0),
            markersize=10,
            label=style['label'] if style['label'] not in ax.get_legend_handles_labels()[1] else None
        )
        
        

def load_and_display(objid):
    path_to_pkl = f'/Users/benjamincollins/University/Master/Red_Cardinal/prospector/pickle_files/{objid}.pkl'
    with open(path_to_pkl, 'rb') as f:
        fit_data = pkl.load(f)
    
    date_and_time = fit_data['saved_at']
    zred = fit_data['redshift']
    print(f"Loading fit data of galaxy {objid} at redshift {zred}")
    print(f"Fit was reconstructed on {date_and_time}")
    
    # Load model spectra
    spec = fit_data['model']['spec_bestfit']
    lower = fit_data['model']['spec_16th']
    median = fit_data['model']['spec_50th']
    upper = fit_data['model']['spec_84th']
    wave_spec = fit_data['model']['wave_spec']
    
    # Load model photometry
    phot = fit_data['model']['phot']
    wave_phot = fit_data['model']['wave_phot']
    
    # Load observation dictionary
    obs = fit_data['obs']
    
    corr_factor = fit_data['calib']['correction']
    corr_unc = fit_data['calib']['uncertainty']
    corr_method = fit_data['calib']['method']
    
    factor = 3631e6  # maggies to µJy conversion factor
    scale_factor = corr_factor * factor
    
    wave_spec = wave_spec * 1e-4 * (1 + zred)
    spec = spec * scale_factor # convert to µJy
    median = median * scale_factor
    lower = lower * scale_factor
    upper = upper * scale_factor
    
    wave_phot = wave_phot * 1e-4 # convert to microns
    phot = phot * scale_factor  # convert to µJy
    
    print(f"Correction factor used to calibrate the spectrum: {corr_factor}")
    print(f"Uncertainty on the calibration: {corr_unc}")
    print(f"The calibration was obtained using the {corr_method}.")

    # Initialise the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot 1 sigma confidence interval
    ax.fill_between(wave_spec, lower, upper, color='crimson', alpha=0.2, label='1σ uncertainty')

    # Optional: Plot median spectrum
    ax.plot(wave_spec, median, color='crimson', alpha=0.5, lw=0.8, label="Median model")
    
    # Plot bestfit spectrum
    ax.plot(wave_spec, spec, '-', color='crimson', alpha=0.8, lw=1.5, label='Best-fit model')

    # Plot model photometry
    ax.plot(wave_phot, phot, 'd', markersize=6, color='black', label='Model photometry')

    # Plot measured photometry
    plot_photometry(ax, obs)

    # Compute bounds
    wave_mask = (wave_spec >= 0.4) & (wave_spec <= 35)
    # Apply mask to spectrum(s)
    spec_within = spec[wave_mask]  # works for 1D or 2D (e.g. percentiles)
    spec_within = [ele for ele in spec_within if ele > 0]
    
    # Compute y-axis limits
    ymin = np.nanmin(spec_within)
    ymax = np.nanmax(spec_within)
    
    # Add margin proportionally, protecting against log-scale issues
    ymin_plot = ymin * 0.2  # reduce, but stay > 0
    ymax_plot = ymax * 5   # increase

    # Set limits
    ax.set_ylim(ymin_plot, ymax_plot)

    # Plot formatting
    ax.set_xlabel('Observed Wavelength (µm)')
    ax.set_ylabel('Flux (µJy)')
    ax.set_xlim(0.4, 35)    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    
