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
import pandas as pd

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
        suffix = ""
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
    
    #for a,b in zip(results['theta_labels'],map_parameters):
    #    print(a, b)
    
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
    
    print("Mean ratio between photometries: ", corr_factor)
    print("Standard deviation: ", corr_uncertainty)
    
    #for f, phot in zip(obs['filters'], maggies):
    #    print(f"{f.name}: {phot} maggies")
        
    #for f, phot in zip(obs['filters_all'], obs['maggies_all']):
    #    print(f"{f.name}: {phot} maggies")
        
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
        
        

def load_and_display(objid, mod=None, mod_err=None):
    path_to_pkl = f'/Users/benjamincollins/University/Master/Red_Cardinal/prospector/pickle_files/{objid}.pkl'
    with open(path_to_pkl, 'rb') as f:
        fit_data = pkl.load(f)
    
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
    fig, ax = plt.subplots(figsize=(10, 6))

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
    ax.set_xlabel('Observed Wavelength (µm)')
    ax.set_ylabel('Flux (µJy)')
    ax.set_xlim(0.4, 35)    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"✅ Plot saved to {filename}")
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
    x_min, x_max = np.percentile(df['N_sigma'], [1, 99])  
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
        
        ax.hist(subset['N_sigma'], bins=bins, color=colors[i], alpha=0.7, edgecolor='black')
        ax.set_title(f'{bands[i]}')
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(r'$N_\sigma$')
        ax.set_ylabel('Number of galaxies')

        nsigmas = subset['N_sigma']
        # Add compact statistics
        mean_ratio = np.mean(nsigmas)
        std_ratio = np.std(nsigmas)
        N = len(subset['galaxy_id'].unique())
        num = f'N = {N}'

        median_ratio = np.median(nsigmas)
        
        stats_text = f'μ={mean_ratio:.2f}\nσ={std_ratio:.2f}\nMed={median_ratio:.2f}\n\n{num}'
            
        ax.text(0.8, 0.71, stats_text, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Annotate in the top-right corner (adjust x,y if needed)
        #ax.text(0.95, 0.95, f'N = {n_galaxies}', 
        #        transform=ax.transAxes, ha='right', va='top',
        #        fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        
    #plt.suptitle(r'$N_\sigma$ distribution for each MIRI filter', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save single combined figure
    filename = os.path.join(out_dir, 'Nsigma_all_filters_v2.png')
    plt.savefig(filename, dpi=300)
    plt.show()
    
    for f in filters:
        subset = df[df['filter_name'] == f]

        plt.figure(figsize=(6,4))
        plt.hist(subset['N_sigma'], bins=bins, color='skyblue', alpha=0.7, edgecolor='black')
        plt.xlabel(r'$N_\sigma$')
        plt.ylabel('Number of galaxies')
        plt.title(rf'$N_\sigma$ distribution for {f}')
        plt.xlim(x_min, x_max)
        plt.tight_layout()
        
        # Count how many galaxies are in this filter
        n_galaxies = len(subset['galaxy_id'].unique())
        # Annotate in the top-right corner (adjust x,y if needed)
        plt.text(0.95, 0.95, f'N = {n_galaxies}',
            transform=plt.gca().transAxes,  # coordinates relative to the axes (0–1)
            ha='right', va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        
        # Save each histogram with the filter name
        filename = os.path.join(out_dir, f'{f}_Nsigma.png')
        plt.savefig(filename, dpi=300)
        plt.show()
        plt.close()
    
    # Plot histograms for galaxies    
    galaxies = df['galaxy_id'].unique()

    x_min, x_max = df['N_sigma'].min(), df['N_sigma'].max()
    
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
        filename = os.path.join(out_dir, f'{gal}_Nsigma_notitle.png')
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





import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from matplotlib.colors import ListedColormap
"""
def setup_publication_style():
    """Set up matplotlib for publication-quality plots"""
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

def plot_mass_vs_redshift(zreds, masses, ndetections, save_path='mass_vs_redshift.png'):
    """
    Plot stellar mass vs redshift with detection status
    
    Parameters:
    -----------
    zreds : array-like
        Redshift values
    masses : array-like
        Stellar masses in solar masses
    ndetections : array-like
        Number of MIRI detections per galaxy
    save_path : str
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    detected = ndetections > 0
    
    # Plot non-detected galaxies first (so detected ones appear on top)
    ax.scatter(zreds[~detected], masses[~detected], 
              s=80, alpha=0.6, color='#808080',
              edgecolor='black', linewidth=0.5,
              label=f'No detections (N={np.sum(~detected)})', zorder=2)
    
    ax.scatter(zreds[detected], masses[detected], 
              s=80, alpha=0.8, color='#ff7f0e', 
              edgecolor='black', linewidth=0.5,
              label=f'MIRI detections (N={np.sum(detected)})', zorder=3)
    
    ax.set_xlabel('Redshift (z)', fontsize=14)
    ax.set_ylabel('Stellar Mass (M$_\\odot$)', fontsize=14)
    ax.set_yscale('log')
    ax.set_title('Stellar Mass vs Redshift', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Add sample statistics as text
    ax.text(0.02, 0.98, f'Total: {len(zreds)} galaxies\nDetected: {np.sum(detected)} ({100*np.sum(detected)/len(zreds):.1f}%)', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as {save_path}")

def plot_main_sequence(masses, sfr100, ndetections, color_scheme='viridis', save_path='main_sequence.png'):
    """
    Plot the star-forming main sequence
    
    Parameters:
    -----------
    masses : array-like
        Stellar masses in solar masses
    sfr100 : array-like
        Star formation rates (100 Myr)
    ndetections : array-like
        Number of MIRI detections per galaxy
    color_scheme : str
        Color scheme to use
    save_path : str
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = get_color_scheme(color_scheme)
    cmap = ListedColormap(colors)
    
    sc = ax.scatter(masses, sfr100, c=ndetections, 
                   cmap=cmap, s=80, alpha=0.8,
                   edgecolor='black', linewidth=0.5,
                   vmin=0, vmax=len(colors)-1)
    
    ax.set_xlabel('Stellar Mass (M$_\\odot$)', fontsize=14)
    ax.set_ylabel('SFR$_{100 Myr}$ (M$_\\odot$ yr$^{-1}$)', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Star-Forming Main Sequence', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Number of MIRI Detections', fontsize=12)
    cbar.set_ticks(np.arange(0, len(colors)))
    
    # Add main sequence line (optional - you can customize this)
    mass_range = np.logspace(9, 11.5, 100)
    
    # Typical main sequence relation: log(SFR) = 0.8*log(M) - 8.5 (adjust as needed)
    ms_sfr = 0.8 * np.log10(mass_range) - 6.5
    
    ax.plot(mass_range, 10**ms_sfr, 'k--', alpha=0.5, linewidth=2, label='Main Sequence')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as {save_path}")

def plot_z_mass_parameter_space(zreds, logmasses, ndetections, color_scheme='viridis', save_path='z_mass_parameter_space.png'):
    """
    Plot z-M parameter space
    
    Parameters:
    -----------
    zreds : array-like
        Redshift values
    logmasses : array-like
        Log stellar masses
    ndetections : array-like
        Number of MIRI detections per galaxy
    color_scheme : str
        Color scheme to use
    save_path : str
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = get_color_scheme(color_scheme)
    cmap = ListedColormap(colors)
    
    sc = ax.scatter(zreds, logmasses, c=ndetections, 
                   cmap=cmap, s=80, alpha=0.8,
                   edgecolor='black', linewidth=0.5,
                   vmin=0, vmax=len(colors)-1)
    
    ax.set_xlabel('Redshift (z)', fontsize=14)
    ax.set_ylabel('log$_{10}$(M$_*$/M$_\\odot$)', fontsize=14)
    ax.set_title('z-M Parameter Space', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Number of MIRI Detections', fontsize=12)
    cbar.set_ticks(np.arange(0, len(colors)))
    
    # Add mass completeness line (optional - customize as needed)
    z_range = np.linspace(zreds.min(), zreds.max(), 100)
    # Example completeness limit: log(M) = 9.5 + 0.3*z (adjust to your survey)
    #completeness_limit = 9.5 + 0.3 * z_range
    #ax.plot(z_range, completeness_limit, 'r--', alpha=0.7, linewidth=2, 
    #        label='Mass Completeness Limit')
    #ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as {save_path}")

def plot_all_galaxy_plots(zreds, logmasses, masses, sfr100, ndetections, 
                         color_scheme='viridis', save_dir='./'):
    """
    Generate all three plots at once
    
    Parameters:
    -----------
    zreds, logmasses, masses, sfr100, ndetections : array-like
        Your galaxy data arrays
    color_scheme : str
        Color scheme for the plots
    save_dir : str
        Directory to save plots
    """
    print(f"Generating galaxy plots with {len(zreds)} galaxies...")
    print(f"Redshift range: {zreds.min():.2f} - {zreds.max():.2f}")
    print(f"Mass range: {masses.min():.2e} - {masses.max():.2e} M☉")
    print(f"SFR range: {sfr100.min():.2f} - {sfr100.max():.2f} M☉/yr")
    print(f"Detections: {np.sum(ndetections > 0)}/{len(ndetections)} galaxies detected")
    print()
    
    # Set up publication style
    setup_publication_style()
    
    # Generate all plots
    plot_mass_vs_redshift(zreds, masses, ndetections, 
                         save_path=f'{save_dir}/mass_vs_redshift.png')
    
    plot_main_sequence(masses, sfr100, ndetections, color_scheme,
                      save_path=f'{save_dir}/main_sequence.png')
    
    plot_z_mass_parameter_space(zreds, logmasses, ndetections, color_scheme,
                               save_path=f'{save_dir}/z_mass_parameter_space.png')
    
    print("All plots generated successfully!")
