import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, ListedColormap
import pickle as pkl
from datetime import datetime
from scipy.stats import norm
import prospect.io.read_results as reader
from prospect.sources import FastStepBasis
from prospect.utils.plotting import posterior_samples
from .params import build_obs, build_model, get_MAP
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch

from matplotlib.image import imread

from miri_utils.photometry_tools import load_vis

# This script is designed to work with PROSPECTOR results and MIRI photometry data.

dirout = "/Users/benjamincollins/University/master/Red_Cardinal/prospector/outputs/"


def reconstruct(objid, plot_dir=None, stats_dir=None, add_duste=True):
    """Main function to reconstruct and plot PROSPECTOR results with MIRI data
    
    Parameters:
    -----------
    objid : int
        Galaxy ID of the object of interest
    plot_dir : str, optional
        Directory to store the plots in
    stats_dir : str, optional
        Directory to write the fit statistics to
    add_duste : bool, optional
        Specify whether dust emission is active or not
        Defaults to True
    """
    
    print(f"Processing galaxy {objid} =============================")
    
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
    else:
        map_parameters = map_parameters[:-6]
        add_agn = False
        
    # Get accurate redshifts from the MAP (obtained by MJ)    
    zred = map_parameters[0]

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

    #print("New model.ndim:", model.ndim)
    #print("New model.theta_index:", model.theta_index)
    #print("New model theta_labels:", [model.theta_labels()[i] for i in range(model.ndim)])

    # Map parameters from original fit to new model structure
    new_theta_labels = [model.theta_labels()[i] for i in range(model.ndim)]
    new_map_parameters = []
    
    # Create mapping between old and new parameter structures
    for new_label in new_theta_labels:
        if new_label in original_theta_labels:
            old_idx = original_theta_labels.index(new_label)
            new_map_parameters.append(original_map_parameters[old_idx])
            #print(f"Mapped {new_label}: {original_map_parameters[old_idx]}")
        else:
            # This shouldn't happen if we're just removing dust parameters
            print(f"Warning: {new_label} not found in original parameters")
            new_map_parameters.append(0.0)  # Default value

    new_map_parameters = np.array(new_map_parameters)

    #print(f"Original parameters length: {len(original_map_parameters)}")
    #print(f"New parameters length: {len(new_map_parameters)}")
    #print(f"New model expects: {model.ndim} parameters")
    
    for a,b in zip(results['theta_labels'],map_parameters):
        print(a, b)
    
    #print("Adjusted length of map_parameters: ", len(map_parameters), "\n")

    # Calculate the spectrum based on the Maximum A Posteriori (MAP) parameters
    sps = FastStepBasis(zcontinuous=1)

    # Obtain best fit model spectrum and model photometry    
    spec, phot, _ = model.predict(map_parameters, obs=obs, sps=sps)
    
    maggies_to_muJy = 3631e6 # maggies to µJy conversion factor
    
    # wavelengths of the model spectrum
    wave_spec = sps.wavelengths
    wave_spec_rs = sps.wavelengths * 1e-4 * (1 + zred)   # convert to µm, redshifted    
    
    # Convert to arrays
    phot = np.array(phot)
    maggies = np.array(obs['maggies'])
    maggies_unc = np.array(obs['maggies_unc'])

    assert len(phot) == len(maggies), 'Model photometry does not match observed photometry length'
    
    # Compute ratio and weights
    ratio = maggies / phot
    weights = 1.0 / maggies_unc**2

    # Weighted average correction factor
    corr_factor = np.sum(ratio * weights) / np.sum(weights)

    # Compute uncertainty on correction factor
    residuals = ratio - corr_factor
    weighted_var = np.sum(weights * residuals**2) / np.sum(weights)
    corr_uncertainty = np.sqrt(weighted_var / len(ratio))
        
    # Initialise the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
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
    
    #########    PLOT MODEL PHOTOMETRY     #########
    
    wave_phot = np.array([filt.wave_effective for filt in obs['filters']])  # in Angstroms
    wave_phot_microns = wave_phot * 1e-4  # convert to µm
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
    ax.set_xlabel('Observed Wavelength (µm)', fontsize=13)
    ax.set_ylabel('Flux (µJy)', fontsize=13)
    ax.set_xlim(0.4, 35)#200)    # Change x range    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=13)

    
    zred_rounded = np.round(zred,2)
    #plt.title(f"Galaxy {objid} at z={zred_rounded}")
    plt.tight_layout()
    
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        fname = os.path.join(plot_dir, f'{objid}.png')
        plt.savefig(fname)
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
            
            # Alignment between the model and observed photometry
            'alignment': {
                'ratio': ratio,
                'corr_factor': corr_factor,
                'corr_uncertainty': corr_uncertainty,
                'weights': weights,
                'maggies_to_muJy': maggies_to_muJy
            },

            # Remaining useful data
            'galaxy_id': objid,
            'redshift': zred,
            'saved_at': datetime.now().isoformat()
        }
        
        # Write output to a pickle file
        with open(filename, 'wb') as f:
            pkl.dump(fit_data, f)

    if plot_dir:
        print(f"✅ Plot saved to {fname}")
    if stats_dir:    
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
        
        

def load_and_display(objid, duste=False, mod=None, mod_err=None, outfile=None):
    """Code to load the fit data from the reconstruct function and plot 
       quickly without using predict() 

    Args:
        objid (int): Object ID of the galaxy of interest
        duste (bool, optional): Specify whether to load files with or without 
                                dust emission. Defaults to False.
        mod (ndarray, optional): Model photometry. Defaults to None.
        mod_err (ndarray, optional): Model uncertainty. Defaults to None.
        outfile (str, optional): Path to output figure. Defaults to None.
        
    Note:  
        mod and mod_err are only relevant if you want to include your custom model 
            photometry, e.g. using sedpy to create phototmetry for bands that were
            not used to fit and hence prospector doesn't return a model for them.
        
       
    """
    
    if duste == False:
        path_to_pkl = f'/Users/benjamincollins/University/Master/Red_Cardinal/prospector/pickle_files/{objid}.pkl'
    else:
        path_to_pkl = f'/Users/benjamincollins/University/Master/Red_Cardinal/prospector/pickle_nodust/{objid}.pkl'

    try:    # try to open
        with open(path_to_pkl, 'rb') as f:
            fit_data = pkl.load(f)
    except FileNotFoundError:
        print(f"⚠️ Attention: File {path_to_pkl} not found. Skipping...")
        return
    
    zred = fit_data['redshift']
    print(f"Loading fit data of galaxy {objid} at redshift {zred}")
    
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
    
    factor = 3631e6  # maggies to µJy conversion factor
    
    wave_spec = wave_spec * 1e-4 * (1 + zred)
    spec *= factor # convert to µJy
    median *= factor
    lower *= factor
    upper *= factor
    
    wave_phot = wave_phot * 1e-4 # convert to microns
    phot *= factor  # convert to µJy
    
    # Initialise the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot 1 sigma confidence interval
    ax.fill_between(wave_spec, lower, upper, color='crimson', alpha=0.2, label='1σ uncertainty')

    # Optional: Plot median spectrum
    ax.plot(wave_spec, median, color='crimson', alpha=0.5, lw=0.8, label="Median model")
    
    # Plot bestfit spectrum
    ax.plot(wave_spec, spec, '-', color='crimson', alpha=0.8, lw=1.5, label='Best-fit model')

    # Plot model photometry
    ax.plot(wave_phot, phot, 'd', markersize=6, color='black', label='Model photometry (Prospector)')

    # Plot model photometry created with sedpy
    if mod is not None and mod_err is not None:
        # Plot provided photometry with error bars
        wave_phot_all = obs['phot_wave_all'] * 1e-4  # convert to microns
        
        miri_mask = (wave_phot_all > 7) & (wave_phot_all < 25)  # AA
        wave_phot_all = wave_phot_all[miri_mask]
        mod_err = np.full_like(mod, mod_err)  # Ensure phot_err is the same length as phot
                
        ax.errorbar(wave_phot_all, mod*factor, yerr=mod_err*factor, fmt='d', color='blue', label='Model photometry (Sedpy)', markersize=6)
        plot_dir = "/Users/benjamincollins/University/Master/Red_Cardinal/prospector/fits/"
        os.makedirs(plot_dir, exist_ok=True)
        filename = os.path.join(plot_dir, f"{objid}.png")
    
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
    ax.set_xlabel('Observed Wavelength (µm)', fontsize=13)
    ax.set_ylabel('Flux (µJy)', fontsize=13)
    ax.set_xlim(0.4, 35)    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.legend()
    
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
        print(f"✅ Plot saved to {outfile}")
    plt.show()
    
    
def create_hist(csv_path, out_dir, bins=25):
    """
    Create a histogram of the N_sigma values for all galaxies and for all bands as stored in the csv file.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File {csv_path} not found. Please check the file path.")
        return
    
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    
    # Create output folder
    os.makedirs(out_dir, exist_ok=True)

    # Compute global x-axis limits (clip outliers if needed)
    #x_min, x_max = np.percentile(df['N_sigma'], [1, 99])  
    x_min = -8.5
    x_max = 8.5
    bins = np.linspace(x_min, x_max, 25)

    filters = df['filter_name'].unique()

    # Example: sort filters by central wavelength
    # (replace this mapping with your actual filters & λ)
    filter_wavelengths = {
        'jwst_f770w': 7.7,
        'jwst_f1000w': 10.0,
        'jwst_f1800w': 18.0,
        'jwst_f2100w': 21.0,
    }

    bands = ['F770W', 'F1000W', 'F1800W', 'F2100W']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # Distinct colors per band

    # Sort filter names by wavelength
    filters_sorted = sorted(filters,
                            key=lambda f: filter_wavelengths.get(f, np.inf))

    # Make a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()  # easier to index

    for i, ax, f in zip((0,1,2,3), axes, filters_sorted):
        subset = df[df['filter_name'] == f]
        
        ax.set_title(f'{bands[i]}')
        #ax.set_xlim(x_min, x_max)
        ax.set_xlabel(r'$N_\sigma$')
        ax.set_ylabel('Number of galaxies')
        
        #if i in [0,1]: ax.set_ylim(0, 24)
        #elif i in [2,3]: ax.set_ylim(0,12)

        nsigmas = subset['N_sigma']
        # Add compact statistics
        mean_ratio = np.mean(nsigmas)
        std_ratio = np.std(nsigmas)
        N = len(subset['galaxy_id'].unique())
        num = f'N = {N}'
        
        counts, bin_edges, _ = ax.hist(nsigmas, bins=bins, color=colors[i], alpha=0.7, edgecolor='black')

        x = np.linspace(x_min, x_max, 500)
        gaussian_norm = norm.pdf(x, loc=0, scale=1)
        gaussian_obs = norm.pdf(x, loc=mean_ratio, scale=std_ratio)

        # Scale Gaussians to match histogram counts
        gaussian_norm_scaled = gaussian_norm * len(nsigmas) * (bin_edges[1] - bin_edges[0])
        gaussian_obs_scaled = gaussian_obs * len(nsigmas) * (bin_edges[1] - bin_edges[0])
        
        ax.plot(x, gaussian_norm_scaled, 'gray', lw=2, alpha=1, label=r'$\mathcal{N}(0,1)$')
        ax.plot(x, gaussian_obs_scaled, colors[i], lw=2, alpha=1, label=r'$\mathcal{N}'+f'({mean_ratio:.2f},{std_ratio:.2f})$')
        
        median_ratio = np.median(nsigmas)
        
        stats_text = f'μ={mean_ratio:.2f}\nσ={std_ratio:.2f}\nMed={median_ratio:.2f}\n\n{num}'
        ax.legend()
        ax.text(0.8, 0.71, stats_text, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Annotate in the top-right corner (adjust x,y if needed)
        #ax.text(0.95, 0.95, f'N = {n_galaxies}', 
        #        transform=ax.transAxes, ha='right', va='top',
        #        fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        
    #plt.suptitle(r'$N_\sigma$ distribution for each MIRI filter', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Save single combined figure
    filename = os.path.join(out_dir, 'Nsigma_all_filters_gauss_v2.png')
    plt.savefig(filename, dpi=300)
    plt.show()
    
    # Plot histograms for galaxies    
    galaxies = df['galaxy_id'].unique()

    #x_min, x_max = df['N_sigma'].min(), df['N_sigma'].max()
    
    x_min = -10
    x_max = 10
    
    for gal in galaxies:
        subset = df[df['galaxy_id'] == gal]
        
        plt.figure(figsize=(5, 4))
        plt.hist(subset['N_sigma'], bins=25, color='skyblue', alpha=0.7, range=(x_min, x_max), edgecolor='black')
        plt.xlim(0, x_max)        
        #plt.title(r'$N_\sigma$ Distribution - ' + f'{gal}')
        plt.xlabel(r'$N_\sigma$')    
        plt.ylabel('Number of bands')
        plt.tight_layout()
        
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f'{gal}_Nsigma_abs.png')
        #plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✅ Saved histogram for galaxy {gal} to {filename}")


    # Compute reduced chi^2 per galaxy
    reduced_chi2 = []

    for gal in df['galaxy_id'].unique():
        subset = df[df['galaxy_id'] == gal]
        n_filters = len(subset)
        if n_filters > 0:
            chi2 = np.sum(subset['N_sigma']**2) / n_filters           
            reduced_chi2.append({'galaxy_id': gal, 'reduced_chi2': chi2})

    chi2_df = pd.DataFrame(reduced_chi2)

    # Plot histogram of reduced chi^2
    plt.figure(figsize=(6,4))
    plt.hist(chi2_df['reduced_chi2'], bins=25, color='salmon', alpha=0.7, edgecolor='black', range=(0, chi2_df['reduced_chi2'].quantile(0.95)))
    plt.xlabel(r'Reduced $\chi^2$')
    plt.ylabel('Number of galaxies')
    #plt.title(r'Reduced $\chi^2$ distribution')
    
    # Count how many chi2 values are in the histogram
    chi2_values = len(chi2_df)
    # Annotate in the top-right corner (adjust x,y if needed)
    plt.text(0.95, 0.95, f'N = {chi2_values}',
        transform=plt.gca().transAxes,  # coordinates relative to the axes (0–1)
        ha='right', va='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    plt.tight_layout()
    filename = os.path.join(out_dir, 'reduced_chi2_hist_notitle.png')
    plt.savefig(filename, dpi=300)
    plt.show()
    
    chi2_df['n_filters'] = chi2_df['galaxy_id'].apply(
    lambda g: len(df[df['galaxy_id'] == g])
    )

    # Compute 95th percentile of reduced chi^2
    q95 = chi2_df['reduced_chi2'].quantile(0.95)

    # Filter out galaxies above this threshold
    filtered = chi2_df[chi2_df['reduced_chi2'] <= q95]

    # Scatter plot with filtered data
    plt.scatter(filtered['n_filters'], filtered['reduced_chi2'], alpha=0.7)    
    plt.xlabel('Number of photometric data points')
    plt.ylabel(r'Reduced $\chi^2$')
    #plt.title(r'Reduced $\chi^2$ vs. number of MIRI bands')
    plt.axhline(1, color='red', linestyle='--')
    filename = os.path.join(out_dir, 'reduced_chi2_vs_npoints_notitle.png')
    plt.savefig(filename, dpi=300)
    plt.show()

    # Compute average fractional discrepancy per galaxy
    frac_disc = (
        df.groupby('galaxy_id')['perc_diff']
        .mean()
        .reset_index()
        .rename(columns={'perc_diff': 'mean_frac_diff'})
    )

    # Merge with chi2_df
    chi2_df = chi2_df.merge(frac_disc, on='galaxy_id', how='left')

    # Plot histogram of the mean fractional difference
    plt.figure(figsize=(6,4))
    plt.hist(chi2_df['mean_frac_diff'], bins=25, color='palegreen', alpha=0.7, edgecolor='black', range=(0, chi2_df['mean_frac_diff'].quantile(0.9)))
    plt.xlabel('Mean fractional difference per galaxy')
    plt.ylabel('Number of galaxies')
    #plt.title('Mean fractional difference per galaxy')
    
    # Annotate in the top-right corner (adjust x,y if needed)
    plt.text(0.95, 0.95, f'N = {chi2_values}',
        transform=plt.gca().transAxes,  # coordinates relative to the axes (0–1)
        ha='right', va='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    plt.tight_layout()
    filename = os.path.join(out_dir, 'mean_frac_diff_hist_notitle.png')
    plt.savefig(filename, dpi=300)
    plt.show()

    # Move n_filters to the last column explicitly
    cols = [c for c in chi2_df.columns if c != 'n_filters'] + ['n_filters']
    chi2_df = chi2_df[cols]
    
    # Sort by reduced_chi2 (ascending = best fit first)
    chi2_df_sorted = chi2_df.sort_values('reduced_chi2', ascending=True).reset_index(drop=True)

    # Save to CSV
    analysis_dir = "/Users/benjamincollins/University/Master/Red_Cardinal/prospector/analysis/"
    filename = os.path.join(analysis_dir, 'fit_quality.csv')
    chi2_df_sorted.to_csv(filename, index=False)

    print(f"Saved ranked fit quality table to {filename}")
    print(chi2_df_sorted.head(10))  # quick preview



"""
def setup_publication_style():
    # Set up matplotlib for publication-quality plots
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
"""


def get_color_scheme(scheme_name='viridis'):
    """Get color schemes for detection plotting"""
    schemes = {
        'viridis': ['#440154', '#31688e', '#35b779', '#fde725'],
        'plasma': ['#0d0887', '#7e03a8', '#cc4678', '#f89441', '#f0f921'],
        'cool': ['#3182bd', '#6baed6', '#9ecae1', '#c6dbef'],
        'warm': ['#d73027', '#f46d43', '#fdae61', '#fee08b'],
        'scientific': ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    }
    return schemes.get(scheme_name, schemes['viridis'])

def plot_main_sequence(masses, sfr100, zred_ms, detections, data=None, color_scheme='viridis', gradient='absolute', save_path='main_sequence.png'):
    """
    Plot the star-forming main sequence
    
    Parameters:
    -----------
    masses : array-like
        Stellar masses in solar masses
    sfr100 : array-like
        Star formation rates (100 Myr)
    zred_ms : float
        Median redshift of the sample
    detections : dict
        Dict of filters and detections per galaxy
    data : dict, optional
        Dict of filters and data to colourise by
    color_scheme : str, optional
        Color scheme to use
    gradient : str, optional
        Choose what to colour by
    save_path : str
        Path to save the plot
    """
    
    # Example galaxy data arrays
    logM = np.log10(masses)          # masses in solar masses
    logSFR_sample = np.log10(sfr100) # SFR in solar masses per year

    # Median redshift for MS line
    t = cosmo.age(zred_ms).to(u.Gyr).value  # cosmic time in Gyr

    # Speagle+14 coefficients
    slope = 0.84 - 0.026 * t
    intercept = -(6.51 - 0.11 * t)

    # 1-sigma errors
    slope_err = 0.02 + 0.003 * t
    intercept_err = 0.24 + 0.03 * t

    # Mass grid in log10(M)
    logM_grid = np.linspace(np.min(logM)-0.1, np.max(logM)+0.1, 200)

    # Main sequence
    logSFR_MS = slope * logM_grid + intercept
    logSFR_high = (slope + slope_err) * logM_grid + (intercept + intercept_err)
    logSFR_low  = (slope - slope_err) * logM_grid + (intercept - intercept_err)
    
    # Now let's introduce the plot   
    
    if gradient == 'absolute':
        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = plt.get_cmap(color_scheme, 5)  # 5 discrete colors: 0,1,2,3,4
        bounds = np.arange(-0.5, 5.5, 1)
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=5)
        N_detected = []
        for det in detections: 
            N_detected.append(sum(det.values()))
        N_detected = np.array(N_detected)
        
        # Scatter plot
        sc = plt.scatter(logM, logSFR_sample, c=N_detected, cmap=cmap, norm=norm, s=60, alpha=0.8, edgecolor='black')
        
        # Colorbar
        cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(0, 5))   # ticks at 0,1,2,3,4
        cbar.set_label('Number of MIRI detections')
        cbar.ax.set_yticklabels([str(i) for i in range(5)])    # ensure labels 0..4
        save_path += f'sfms_{gradient}.png'
        
    elif gradient == 'relative':  
        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = plt.get_cmap(color_scheme)
        N_detected = []
        N_available = []
        for det in detections: 
            N_detected.append(sum(det.values()))
            N_available.append(len(det.values()))
        N_detected = np.array(N_detected)
        N_available = np.array(N_available)
        f_det = N_detected / N_available  # fraction 0-1  
        
        # Scatter plot
        sc = ax.scatter(logM, logSFR_sample, c=f_det, cmap=cmap, s=60, edgecolor='black', norm=Normalize(vmin=0, vmax=1))
        
        # Colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
        cbar.set_label('Relative number of MIRI detections')
        save_path += f'sfms_{gradient}.png'
    
    elif gradient in ['f770w', 'f1000w', 'f1800w', 'f2100w']:
        fig, ax = plt.subplots(figsize=(6, 4))
        flux_array = []
        mask = []
        band = gradient.upper()
        
        for det_dict, flux_dict in zip(detections, data):
            if det_dict.get(band, False) and band in flux_dict:
                flux_array.append(flux_dict[band])
                mask.append(True)
            else:
                flux_array.append(np.nan)
                mask.append(False)
        
        mask = np.array(mask)
        
        # Apply mask to all quantities
        flux_array = np.array(flux_array)[mask]*1e6
        logM = np.array(logM)[mask]
        logSFR_sample = np.array(logSFR_sample)[mask]
        
        # Convert flux to log scale
        log_flux_array = np.log10(flux_array)

        if gradient == 'f770w': color_scheme = "Blues"
        elif gradient == 'f1000w': color_scheme = "Greens"
        elif gradient == 'f1800w': color_scheme = "Oranges"
        elif gradient == 'f2100w': color_scheme = "Reds"

        # Create scatter plot coloured by log flux
        sc = ax.scatter(logM, logSFR_sample, c=log_flux_array, cmap=color_scheme, s=60, alpha=0.8, edgecolor='black')
        
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(rf'$\log_{{10}}(F_{{\mathrm{{{gradient.upper()}}}}})\ [µJy]$')        #cbar.set_label("log$_{10}$(F770W flux) [μJy]")
        #ax.set_xlim(8, 13)
        ax.set_ylim(-1, 4)
        save_path += f'sfms_{gradient}.png'
        
    elif gradient in ['nsig_f770w', 'nsig_f1000w', 'nsig_f1800w', 'nsig_f2100w']:
        fig, ax = plt.subplots(figsize=(6, 4))
        nsig_array = []
        mask = []
        band = gradient.split('_')[1].upper()
        
        color_scheme = "gnuplot2"
        
        for det_dict, nsig_dict in zip(detections, data):
            if det_dict.get(band, False) and band in nsig_dict:
                nsig_array.append(nsig_dict[band])
                mask.append(True)
            else:
                nsig_array.append(np.nan)
                mask.append(False)
        
        mask = np.array(mask)

        # Apply mask to all quantities
        nsig_array = np.array(nsig_array)[mask]
        logM = np.array(logM)[mask]
        logSFR_sample = np.array(logSFR_sample)[mask]
        
        # Create scatter plot coloured by log flux
        sc = ax.scatter(logM, logSFR_sample, c=nsig_array, cmap=color_scheme, s=60, alpha=0.8, vmin=-7, vmax=7, edgecolor='black')
        
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(rf'$N_\sigma$ ({band})')
        ax.set_ylim(-1, 4)
        save_path += f'sfms_{gradient}.png'
        
    else:
        print("⚠️Gradient has to be set to either absolute or relative.")
        return None

    # MS line and shaded 1-sigma region
    ax.plot(logM_grid, logSFR_MS, 'k--', alpha=0.5, label=f'Speagle+14 MS (z={zred_ms:.2f})')
    ax.fill_between(logM_grid, logSFR_low, logSFR_high, color='gray', alpha=0.15, label='1σ uncertainty')

    # Labels and legend
    ax.set_xlabel('log$_{10}$(M$_*$/M$_\\odot$)', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\mathrm{SFR} / $M$_\odot\,\mathrm{yr}^{-1})$', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as {save_path}")

def plot_mass_vs_redshift(zreds, logmasses, detections, data=None, color_scheme='plasma', gradient='absolute', save_path='z_mass_parameter_space.png'):
    """
    Plot z-M parameter space
    
    Parameters:
    -----------
    zreds : array-like
        Redshift values
    logmasses : array-like
        Log stellar masses
    detections : dict-like
        Available filters with True/False
    color_scheme : str
        Color scheme to use
    gradient : str
        Specify whether to use absolute or relative detections for the colorbar
    save_path : str
        Path to save the plot
    """
    
    if gradient == 'absolute':
        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = plt.get_cmap(color_scheme, 5)  # 5 discrete colors: 0,1,2,3,4
        bounds = np.arange(-0.5, 5.5, 1)
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=5)
        N_detected = []
        for det in detections: 
            N_detected.append(sum(det.values()))
        N_detected = np.array(N_detected)
        sc = plt.scatter(zreds, logmasses, c=N_detected, cmap=cmap, norm=norm, s=60, alpha=0.8, edgecolor='black')
        
        cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(0, 5))   # ticks at 0,1,2,3,4
        cbar.set_label('Number of MIRI detections')
        cbar.ax.set_yticklabels([str(i) for i in range(5)])    # ensure labels 0..4
        
        detected = N_detected > 0
        # Add sample statistics as text
        ax.text(0.78, 0.98, f'Total: {len(zreds)} galaxies\nDetected: {np.sum(detected)} ({100*np.sum(detected)/len(zreds):.1f}%)', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        save_path = save_path + f'zM_{gradient}.png'
        
    elif gradient == 'relative': 
        fig, ax = plt.subplots(figsize=(10, 5)) 
        N_detected = []
        N_available = []
        for det in detections: 
            N_detected.append(sum(det.values()))
            N_available.append(len(det.values()))
        N_detected = np.array(N_detected)
        N_available = np.array(N_available)
        f_det = N_detected / N_available  # fraction 0-1  
        
        cmap = plt.get_cmap(color_scheme)

        sc = ax.scatter(zreds, logmasses, c=f_det, cmap=cmap, s=60, edgecolor='black', norm=Normalize(vmin=0, vmax=1))

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
        cbar.set_label('Relative number of MIRI detections')
        
        detected = N_detected > 0
        # Add sample statistics as text
        ax.text(0.78, 0.98, f'Total: {len(zreds)} galaxies\nDetected: {np.sum(detected)} ({100*np.sum(detected)/len(zreds):.1f}%)', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        save_path = save_path + f'zM_{gradient}.png'
        
    elif gradient in ['f770w', 'f1000w', 'f1800w', 'f2100w']:
        fig, ax = plt.subplots(figsize=(6, 4))
        flux_array = []
        mask = []
        band = gradient.upper()
        
        for det_dict, flux_dict in zip(detections, data):
            if det_dict.get(band, False) and band in flux_dict:
                flux_array.append(flux_dict[band])
                mask.append(True)
            else:
                flux_array.append(np.nan)
                mask.append(False)
        
        mask = np.array(mask)
        
        # Apply mask to all quantities
        flux_array = np.array(flux_array)[mask]*1e6
        zreds = np.array(zreds)[mask]
        logmasses = np.array(logmasses)[mask]

        # Convert flux to log scale
        log_flux_array = np.log10(flux_array)

        if gradient == 'f770w': color_scheme = "Blues"
        elif gradient == 'f1000w': color_scheme = "Greens"
        elif gradient == 'f1800w': color_scheme = "Oranges"
        elif gradient == 'f2100w': color_scheme = "Reds"

        # Create scatter plot coloured by log flux
        sc = ax.scatter(zreds, logmasses, c=log_flux_array, cmap=color_scheme, s=60, alpha=0.8, edgecolor='black')
        
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(rf'$\log_{{10}}(F_{{\mathrm{{{gradient.upper()}}}}})\ [µJy]$')        #cbar.set_label("log$_{10}$(F770W flux) [μJy]")
        save_path = save_path + f'zM_{gradient}.png'
        ax.set_xlim(1.5, 3.75)
        ax.set_ylim(9, 12)
    
    elif gradient in ['nsig_f770w', 'nsig_f1000w', 'nsig_f1800w', 'nsig_f2100w']:
        fig, ax = plt.subplots(figsize=(6, 4))
        nsig_array = []
        mask = []
        band = gradient.split('_')[1].upper()
        
        for det_dict, nsig_dict in zip(detections, data):
            if det_dict.get(band, False) and band in nsig_dict:
                nsig_array.append(nsig_dict[band])
                mask.append(True)
            else:
                nsig_array.append(np.nan)
                mask.append(False)
        
        mask = np.array(mask)

        # Apply mask to all quantities
        nsig_array = np.array(nsig_array)[mask]
        zreds = np.array(zreds)[mask]
        logmasses = np.array(logmasses)[mask]
        
        # Create scatter plot coloured by log flux
        sc = ax.scatter(zreds, logmasses, c=nsig_array, cmap=color_scheme, s=60, alpha=0.8, vmin=-7, vmax=7, edgecolor='black')
        
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(rf'$N_\sigma$ ({band})')
        save_path = save_path + f'zM_{gradient}_r.png'
        ax.set_xlim(1.5, 3.75)
        ax.set_ylim(9, 12)
        
    
    else:
        print("⚠️Gradient has to be set to either absolute, relative or flux.")
        return None
          
        
    ax.set_xlabel('Redshift (z)', fontsize=14)
    ax.set_ylabel('log$_{10}$(M$_*$/M$_\\odot$)', fontsize=14)
    #ax.set_title('z-M Parameter Space', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as {save_path}")


def plot_nsigma_vs_params(nsig, band, log_ssfr, dust, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Left: Nsigma vs sSFR
    sc1 = axes[0].scatter(log_ssfr, nsig, c=dust, cmap="viridis", alpha=0.7, edgecolor='black')
    axes[0].axhline(0, ls="--", c="grey")
    axes[0].set_xlabel(r'$\log(\mathrm{sSFR}_{100}\,[\mathrm{yr}^{-1}])$')
    axes[0].set_ylabel(r'$N_\sigma$')
    axes[0].set_title(f'{band}: ' + r'$\mathrm{N_\sigma}$ vs sSFR')
    cb1 = fig.colorbar(sc1, ax=axes[0])
    cb1.set_label("Dust attenuation (dust2)")
    
    # --- Right: Nsigma vs Dust
    sc2 = axes[1].scatter(dust, nsig, c=log_ssfr, cmap="plasma", alpha=0.7, edgecolor='black')
    axes[1].axhline(0, ls="--", c="grey")
    axes[1].set_xlabel(r'Dust attenuation ($\mathrm{dust2}$)')
    axes[1].set_ylabel(r'$N_\sigma$')
    axes[1].set_title(f'{band}: ' + r'$\mathrm{N_\sigma}$ vs $\mathrm{A_V}$')
    cb2 = fig.colorbar(sc2, ax=axes[1])
    cb2.set_label(r'$\log(\mathrm{sSFR}_{100})$')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def plot_extremes(objid, base_paths, add_fit=False, save_path=None):
    """
    Plot dynamic cutouts + Prospector fit for one galaxy.
    
    Parameters
    ----------
    objid : int or str
        Galaxy ID
    base_paths : dict
        Dictionary of paths with keys: "miri", "nircam", "prospector"
    show_fit : bool, optional
        Decide whether or not to add the PROSPECTOR fit to the figure
    save_path : bool, optional
        Specify save path for the figure
    """

    # --- Paths ---
    nircam_path = os.path.join(base_paths["nircam"], f"{objid}_F444W_cutout.fits")
    miri_paths = {
        band: os.path.join(base_paths["miri"], f"{objid}_{band}.h5")
        for band in ["F1800W", "F2100W"]
    }
    
    prospector_path = os.path.join(base_paths["prospector"], f"{objid}.png")

    # --- Check availability ---
    available_miri = [b for b, p in miri_paths.items() if os.path.exists(p)]
    if not available_miri:
        print(f"Skipping {objid} — no MIRI cutouts found")
        return

    nircam_exists = os.path.exists(nircam_path)
    prospector_exists = os.path.exists(prospector_path)

    # --- Case 1: one MIRI band ---
    if len(available_miri) == 1:
        if add_fit == True:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"width_ratios":[1,1,1.7]})
            f444w_ax, miri_ax, prosp_ax = axes
        else:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            f444w_ax, miri_ax = axes

        if nircam_exists:
            show_fits_cutout(nircam_path, f444w_ax, f"{objid} - F444W")
        else:
            f444w_ax.axis("off")

        show_h5_cutout(miri_paths[available_miri[0]], miri_ax, f"{objid} - {available_miri[0]}")

        if add_fit == True:
            if prospector_exists:
                show_png(prospector_path, prosp_ax, f"{objid} - Prospector Fit")
            else:
                prosp_ax.axis("off")

    # --- Case 2: both MIRI bands ---
    elif len(available_miri) == 2:
        if add_fit == True:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={"width_ratios":[1,1,1,1.7]})
            ax_f444w, ax_f1800w, ax_f2100w, ax_prosp = axes
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            ax_f444w, ax_f1800w, ax_f2100w = axes

        if nircam_exists:
            show_fits_cutout(nircam_path, ax_f444w, f"{objid} - F444W")
        else:
            ax_f444w.axis("off")

        show_h5_cutout(miri_paths["F1800W"], ax_f1800w, f"{objid} - F1800W")
        show_h5_cutout(miri_paths["F2100W"], ax_f2100w, f"{objid} - F2100W")

        if add_fit:
            if prospector_exists:
                show_png(prospector_path, ax_prosp, f"{objid} - Prospector Fit")
            else:
                ax_prosp.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# --- Helpers ---
def show_fits_cutout(path, ax, title):
    img = fits.getdata(path)
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(img)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
    ax.imshow(img, origin="lower", cmap="inferno", norm=norm)
    ax.set_title(title)
    ax.axis("off")

def show_h5_cutout(path, ax, title):
    import h5py
    with h5py.File(path, "r") as f:
        img_bkgsub = f["background_subtracted"][:]
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(img_bkgsub)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
    ax.imshow(img_bkgsub, origin="lower", cmap="inferno", norm=norm)
    ax.set_title(title)
    ax.axis("off")

def show_png(path, ax, title):
    img = imread(path)
    ax.imshow(img)
    #ax.set_title(title)
    ax.axis("off")
