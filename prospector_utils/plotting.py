import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import traceback
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor
import h5py

from astropy.table import Table
import prospect.io.read_results as reader
from prospect.sources import FastStepBasis
from prospect.utils.plotting import posterior_samples


# This script is designed to work with PROSPECTOR results and MIRI photometry data.

# Set paths (adjust these to your system)
cat_dir = '/Users/benjamincollins/University/master/Red_Cardinal/photometry/catalogues/'

dirout = "/Users/benjamincollins/University/master/Red_Cardinal/prospector/outputs/"


def plot_reconstruction_with_miri(objid, output_dir=None):
    """Main function to reconstruct and plot PROSPECTOR results with MIRI data"""
    
    # Load the h5 file for the given objid
    h5_file = glob.glob(os.path.join(dirout, f"output_{objid}*.h5"))
    
    try:
        h5_file = h5_file[0]
    except IndexError:
        print(f"No PROSPECTOR results found for objid {objid}.")
        return None, None, None

    # Load PROSPECTOR results
    full_path = os.path.join(dirout, h5_file)
    results, loaded_obs, model = reader.results_from(full_path)

    # Calculate the spectrum based on the Maximum A Posteriori (MAP) parameters
    sps = FastStepBasis(zcontinuous=1)
    wave_full = sps.wavelengths
    
    # Build new observations including MIRI
    obs = build_obs(objid)
    
    # Now we have to exclude the last 3 parameters from the fit
    map_parameters = get_MAP(results)
    map_parameters = map_parameters[:-3]

    # Get accurate redshifts from the MAP (obtained by MJ)    
    zred = map_parameters[0]
    print(f"Processing galaxy {objid} at redshift z = {zred}")

    # Build model matching original setup
    model = build_model(objid, zred, add_duste=results['run_params']['add_duste'],
                    add_neb=False,  # Set add_neb to False
                    add_agn=results['run_params']['add_agn'],
                    fit_afe=results['run_params']['fit_afe'])
    
    model.params['polyorder'] = 10

    # Obtain calibration vector
    spec_new, _, _ = model.predict(map_parameters, obs=loaded_obs, sps=sps)
    
    mask = loaded_obs['mask']
    calib_vector = model._speccal
    calib_vector = calib_vector[mask]
    calib_mean = np.mean(calib_vector)/1e10
    calib_std = np.std(calib_vector)/1e10
    
    # Obtain best fit model spectrum and model photometry    
    spec, phot, _ = model.predict(map_parameters, obs=obs, sps=sps)
    
    #get the spectral jitter  for the errors
    #spec_jitter = map_parameters[5]
    #errs = obs['unc']*spec_jitter/calib_vector * 3631 # observational errors
    
    factor = 3631e6 # maggies to µJy conversion factor
    
    # Redshifted model spectrum
    wave_spec = sps.wavelengths * 1e-4 * (1 + zred)   # convert to µm, redshifted    
    
    # Define the style per instrument
    instrument_styles = {
        'acs':     {'color': 'royalblue',   'marker': 'o', 'edgecolor': 'black', 'label': 'HST ACS', 'ms': 10},
        'wfc3':    {'color': 'limegreen',  'marker': 'o', 'edgecolor': 'black', 'label': 'HST WFC3', 'ms': 10},
        'nircam':  {'color': 'orange', 'marker': 'p', 'edgecolor': 'black',    'alpha': 0.7, 'label': 'JWST NIRCam', 'ms': 10},
        'miri':    {'color': 'firebrick',    'marker': 'p', 'edgecolor': 'black',    'alpha': 0.7, 'label': 'JWST MIRI (overlaid)', 'ms': 10}
    }
    
    ratio = []
    for p,o in zip(phot, obs['maggies']):
        ratio.append(o/p)
    
    phot_err = obs['maggies_unc']
    weights = 1.0 / phot_err**2
    corr_factor = np.sum(ratio * weights) / np.sum(weights)
    
    residuals = np.array(ratio) - corr_factor
    weighted_var = np.sum(weights * residuals**2) / np.sum(weights)
    corr_uncertainty = np.sqrt(weighted_var / len(ratio))

    
    print("Mean ratio between photometries: ", corr_factor)
    print("Standard deviation: ", corr_uncertainty)
    
    scale_factor = corr_factor * factor
    
    # Initialise the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ################################################
    #########   PLOT POSTERIOR SAMPLES     #########
    ################################################
    
    nsample = 100
    samples = posterior_samples(results, nsample)
    
    all_specs = []
    for params_i in samples:
        params_i = params_i[:-3]
        spec_i, _, _ = model.predict(params_i, obs=obs, sps=sps)
        spec_i = spec_i * scale_factor    # convert to µJy
        all_specs.append(spec_i)
        
    all_specs = np.array(all_specs)  # shape: (nsample, nwave)
    
    lower = np.percentile(all_specs, 16, axis=0)
    upper = np.percentile(all_specs, 84, axis=0)
    
    ax.fill_between(wave_spec, lower, upper, color='crimson', alpha=0.2, label='1σ uncertainty')
    
    for i in range(10):
        ax.plot(wave_spec, all_specs[i], color='crimson', alpha=0.15, lw=0.8)
    
    ################################################
    #########       PLOT THE BEST FIT      #########
    ################################################
    
    spec = spec * scale_factor    # convert to µJy
    ax.plot(wave_spec, spec, '-', color='crimson', alpha=0.8, lw=1.5, label='Best-fit model')
    
    ################################################
    #########   PLOT OBSERVED SPECTRUM     #########
    ################################################
        
    
    #spectrum  = loaded_obs['spectrum'][mask] / calib_vector[mask] * factor
    
    #ax.plot(wavelengths, spectrum, '-', color='blue', alpha=0.8, lw=1.5, label='Observed spectrum')
    
    ################################################
    ######### PLOT THE CALIBRATED SPECTRUM #########
    ################################################
    
    #param_file = '/Users/benjamincollins/University/master/Red_Cardinal/prospector/spec_calib/spec_calibrated_12717.pkl'
    #data = pkl.load(open(param_file, 'rb'))
    #cal_spec = data['emi_off']['MAP']['wave_obs'] * factor
    #lambdas = data['wave_obs'] * 1e-4
    #ax.plot(lambdas, cal_spec, linestyle="-", alpha=0.8, lw=1.5, label="Calibrated spectrum")
    
    ################################################
    #########    PLOT MODEL PHOTOMETRY     #########
    ################################################
    
    wave_model = np.array([filt.wave_effective for filt in obs['filters']]) * 1e-4  # µm
    phot = phot * scale_factor  # µJy
    ax.plot(wave_model, phot, 'd', markersize=6, color='black', label='Model photometry')
    
    ################################################
    #########  PLOT MEASURED PHOTOMETRY    #########
    ################################################

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
    
    # Compute bounds
    ymin = np.min(spec)
    ymax = np.max(spec)

    # Add 10% margin
    yrange = ymax - ymin
    ymin_plot = max(ymin - 0.1 * yrange, 1e-2)  # prevent negative or log(0)
    ymax_plot = ymax + 0.1 * yrange

    # Set limits
    ax.set_ylim(ymin_plot, ymax_plot)

    # Plot formatting
    ax.set_xlabel('Observed Wavelength (µm)')
    ax.set_ylabel('Flux (µJy)')
    ax.set_xlim(0.4, 35)    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    zred = np.round(zred,2)
    plt.title(f"Galaxy {objid} at z={zred}")
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f'{objid}_miri_overlay_custom_calib.png')
        plt.savefig(fname)
        print(f"Saved plot to {fname}")
    plt.show()
    plt.close()
    
    return obs, model, results

table_path = '/Users/benjamincollins/University/Master/Red_Cardinal/photometry/phot_tables/Photometry_Table_MIRI_v6.fits'

table = Table.read(table_path, format='fits')
galaxy_ids = [str(gid) for gid in table['ID']]    


#if __name__ == "__main__":
#    max_workers = min(6, os.cpu_count())
#    with ProcessPoolExecutor(max_workers=max_workers) as executor:
#        executor.map(plot_reconstruction_with_miri, galaxy_ids)

# Todo
# Try overplotting calibrated spectrum from Dropbox and see if it aligns either with the photometry or with the fit