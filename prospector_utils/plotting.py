import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, ListedColormap
import pickle as pkl
from scipy.stats import norm, median_abs_deviation

from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

from astropy.table import Table

from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch

from matplotlib.image import imread




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
    
    # Get current labels to prevent duplicates
    _, labels = ax.get_legend_handles_labels()
    
    for i, filt in enumerate(obs['filters_all']):
        
        wave = obs['phot_wave_all'][i] * 1e-4  # convert to µm
        flux = obs['maggies_all'][i] * factor  # µJy
        err  = obs['maggies_unc_all'][i] * factor  # µJy
        
        uplims = False
        
        name = filt.name.lower()
        
        if 'acs_wfc' in name:
            style = instrument_styles['acs']
        elif 'wfc3_ir' in name:
            style = instrument_styles['wfc3']
        elif 'miri' in name or any(m in name for m in ['f770w', 'f1000w', 'f1800w', 'f2100w']):
            style = instrument_styles['miri']
            # Improved Upper Limit Logic for MIRI
            if (flux / err < 3.0):
                uplims = True
                flux = 3 * err # Plot at 3-sigma
                err = flux * 0.4 # Small arrow size for visualisation
            
        elif 'nircam' in name or ('jwst' in name and 'f' in name and 'w' in name):
            style = instrument_styles['nircam']
        else:
            continue  # skip unknown filters


        ax.errorbar(
            wave, flux, yerr=err,
            fmt=style['marker'],
            color=style['color'],
            markeredgecolor=style.get('edgecolor', 'none'),
            alpha=style.get('alpha', 1.0),
            markersize=10,
            uplims=uplims, # This creates the actual downward arrow
            label=style['label'] if style['label'] not in labels else None
        )
        
        # Update labels list to prevent duplicates in current loop
        if style['label'] not in labels:
            labels.append(style['label'])


def plot_reconstructed_fit(filename, plot_dir):
    
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
    phot_wave = model['phot_wave']
    phot_wave_miri = model['phot_wave_miri']
    
    obs = fit_data['obs']
    maggies_to_muJy = fit_data['maggies_to_muJy']
    
    # Convert to µJy
    lower_scaled = spec_16th * maggies_to_muJy    
    median_scaled = spec_median * maggies_to_muJy
    upper_scaled = spec_84th * maggies_to_muJy
    spec_scaled = spec_best * maggies_to_muJy
    
    phot_wave_microns = phot_wave * 1e-4  # convert to µm
    phot_wave_miri_microns = phot_wave_miri * 1e-4  # convert to µm
    
    phot_scaled = phot * maggies_to_muJy
    phot_miri_scaled = phot_miri * maggies_to_muJy
    phot_miri_err_scaled = phot_miri_err * maggies_to_muJy
    
    wave_spec_rs = wave_spec * 1e-4 * (1+zred)
    
    # Initialise the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot shaded region for 1σ uncertainty
    ax.fill_between(wave_spec_rs, lower_scaled, upper_scaled, color='crimson', alpha=0.2, label='1σ uncertainty')
    
    for spec in sample_specs:
        ax.plot(wave_spec_rs, spec*maggies_to_muJy, color='crimson', alpha=0.15, lw=0.8)
    
    #ax.plot(wave_spec_rs, lower_scaled, color='blue', lw=0.8, label='16th percentile')
    #ax.plot(wave_spec_rs, upper_scaled, color='blue', lw=0.8, label='84th percentile')
    #########       PLOT THE BEST FIT      #########
    
    ax.plot(wave_spec_rs, spec_scaled, '-', color='crimson', alpha=0.8, lw=1.5, label='Best-fit model')
    
    #########    PLOT MODEL PHOTOMETRY     #########
    
    ax.plot(phot_wave_microns, phot_scaled, 'd', markersize=6, color='black', label='Model photometry')
    ax.errorbar(phot_wave_miri_microns, phot_miri_scaled, yerr=phot_miri_err_scaled, fmt='d', markersize=6, color='blue')
    #ax.plot(phot_wave_miri_microns, phot_miri_scaled, 'd', markersize=6, color='black')
    
    #########  PLOT MEASURED PHOTOMETRY    #########

    plot_photometry(ax, obs)
    
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
    if gid in [7549, 7696, 8013, 9395, 10339, 10400, 11142, 11247, 11494, 12133, 12175, 12332, 21472, 21477]:
        ax.legend(loc="lower right")
    else:
        ax.legend(loc="upper left")
    #ax.set_title(f"Galaxy {objid} at z={np.round(zred,2)}", fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=13)
    
    zred_rounded = np.round(zred,2)
    #plt.title(f"Galaxy {objid} at z={zred_rounded}")
    plt.tight_layout()
    
    os.makedirs(plot_dir, exist_ok=True)
    fname = os.path.join(plot_dir, f'{gid}.png')
    plt.savefig(fname)
    plt.show()
    plt.close()
    
    return






def plot_quality_stats(pickle_dir, out_dir, bins=25, plot_nsigmas=True, plot_logratios=True, plot_chi2red=True):
    """
    Create a histogram of the N_sigma values for all galaxies and for all bands as stored in the csv file.
    """
    
    pickle_files = glob.glob(f'{pickle_dir}/*.pkl')
    
    all_data = []
    reduced_chi2_list = []

    # 1. DATA EXTRACTION
    for filename in pickle_files:
        with open(filename, 'rb') as f:
            data = pkl.load(f)
            
        gid = data['id']
        fit_quality = data.get('fit_quality', {})
        
        # Store global per-galaxy stats
        if 'chi2_red' in fit_quality:
            reduced_chi2_list.append({
                'galaxy_id': gid,
                'reduced_chi2': fit_quality['chi2_red'][0],
                'n_filters': len(fit_quality) - 1 # Assuming only 'chi2_red' is non-filter
            })

        # Store per-band stats
        for i, (key, val) in enumerate(fit_quality.items()):
            if isinstance(val, dict): # This identifies the filter entries
                all_data.append({
                    'galaxy_id': gid,
                    'filter_name': key,
                    'N_sigma': val.get('n_sigma'),
                    'flux': val.get('flux'),
                    'flux_err': val.get('flux_err'),
                    'frac_diff': val.get('frac_diff')
                })

    df = pd.DataFrame(all_data)
    chi2_df = pd.DataFrame(reduced_chi2_list)
    
    print("Extracted data!")
    
    """    
    fit_quality['chi2_red'] = chi2_red,
    
    for i, filt in enumerate(obs_miri['filters_miri']):
        
        # Extracts 'F770W' from 'jwst_f770w'
        name = filt.name.split('_')[-1].upper()
        
        fit_quality[name] = {
            'n_sigma': N_sigma[i],
            'frac_diff' : perc[i],
            'flux': miri_flux[i] * maggies_to_muJy,
            'flux_err': miri_err[i] * maggies_to_muJy
        }
    """
        
    # Create output folder
    os.makedirs(out_dir, exist_ok=True)

    # Compute global x-axis limits (clip outliers if needed)
    #x_min, x_max = np.percentile(df['N_sigma'], [1, 99])  

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
    #filters_sorted = sorted(filters, key=lambda f: filter_wavelengths.get(f, np.inf))
    
    if plot_nsigmas:
        # PLOT 1: N_SIGMA HISTOGRAMS
        fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=False, sharey=True)
        axes = axes.flatten()  # easier to index

        print("Initialised axes")

        for i, (ax, band) in enumerate(zip(axes, bands)):
            subset = df[df['filter_name'] == band]
            nsigmas = subset['N_sigma']
            if len(nsigmas) == 0: continue
            
            ax.set_title(f'{band}')
            #ax.set_xlim(x_min, x_max)
            ax.set_xlabel(r'$N_\sigma$')
            ax.set_ylabel('Number of galaxies')
            
            #if i in [0,1]: ax.set_ylim(0, 24)
            #elif i in [2,3]: ax.set_ylim(0,12)

            # Add compact statistics
            mean_ratio = np.mean(nsigmas)
            median_ratio = np.median(nsigmas)
            std_ratio = np.std(nsigmas)
            mad_ratio = median_abs_deviation(nsigmas)
            N = len(subset['galaxy_id'].unique())
            num = f'N = {N}'
            
            x_min = -8.5
            x_max = 8.5
            bins = np.linspace(x_min, x_max, 25)
            
            counts, bin_edges, _ = ax.hist(nsigmas, bins=bins, color=colors[i], alpha=0.7, edgecolor='black')

            x = np.linspace(x_min, x_max, 500)
            gaussian_norm = norm.pdf(x, loc=0, scale=1)
            gaussian_obs = norm.pdf(x, loc=median_ratio, scale=mad_ratio)

            # Scale Gaussians to match histogram counts
            gaussian_norm_scaled = gaussian_norm * len(nsigmas) * (bin_edges[1] - bin_edges[0])
            gaussian_obs_scaled = gaussian_obs * len(nsigmas) * (bin_edges[1] - bin_edges[0])
            
            ax.plot(x, gaussian_norm_scaled, 'gray', lw=2, alpha=1, label=r'$\mathcal{N}(0,1)$')
            ax.plot(x, gaussian_obs_scaled, colors[i], lw=2, alpha=1, label=r'$\mathcal{N}'+f'({median_ratio:.2f},{mad_ratio:.2f})$')
            
            median_ratio = np.median(nsigmas)
            
            stats_text = f'Med = {median_ratio:.2f}\nMAD = {mad_ratio:.2f}\n{num}'
            ax.legend()
            ax.text(0.78, 0.84, stats_text, transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Annotate in the top-right corner (adjust x,y if needed)
            #ax.text(0.95, 0.95, f'N = {n_galaxies}', 
            #        transform=ax.transAxes, ha='right', va='top',
            #        fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            
        #plt.suptitle(r'$N_\sigma$ distribution for each MIRI filter', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # Save single combined figure
        filename = os.path.join(out_dir, 'Nsigma_gauss.png')
        plt.savefig(filename, dpi=300)
        plt.show()
        
    
    
    if plot_logratios:
        # PLOT 2: LOG RATIOS
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=True)
        axes = axes.flatten()  # easier to index

        for i, (ax, band) in enumerate(zip(axes, bands)):
            subset = df[df['filter_name'] == band]
            
            f_obs = subset['flux']
            err = subset['flux_err']
            frac_diff = subset['frac_diff']
            
            # SNR filter: Keep only detections > 3-sigma
            snr_mask = (f_obs / err) >= 3.0
            
            # Model reconstruction
            # Using the simplified: f_model = f_obs * (1 - frac_diff)
            # We filter out frac_diff >= 1 to avoid log(0) or log(negative)
            valid_mask = (frac_diff < 1.0) & snr_mask
            
            # Calculate ratios only for valid entries
            # log10(f_obs / (f_obs * (1-frac_diff))) reduces to -log10(1-frac_diff)
            log_ratios = -np.log10(1.0 - frac_diff[valid_mask])
            
            ax.set_title(f'{band}')
            #ax.set_xlim(x_min, x_max)
            ax.set_xlabel('Flux ratio (dex)')
            ax.set_ylabel('Number of galaxies')
            
            #if i in [0,1]: ax.set_ylim(0, 24)
            #elif i in [2,3]: ax.set_ylim(0,12)
            
            # Add compact statistics
            mean_logr = np.mean(log_ratios)
            std_logr = np.std(log_ratios)
            median_logr = np.median(log_ratios)
            mad_logr = median_abs_deviation(log_ratios)
            N = len(log_ratios)
            num = f'N = {N}'
            
            x_min = -1.2
            x_max = 1.2
            bins = np.linspace(x_min, x_max, 25)
            
            counts, _, _ = ax.hist(log_ratios, bins=bins, color=colors[i], alpha=0.7, edgecolor='black')

            ymax = 25 # for all plots
            ax.set_ylim(0,25)
            ax.vlines(median_logr, ymin=0, ymax=ymax, color='darkred', alpha=0.8, linestyle='-', linewidth=2, label=f'Median: {median_logr:.2f}')
#            if i == 2: 
 #               stats_text += ' (*)'
  #              print(log_ratios[log_ratios > 1])
            ax.text(0.025, 0.92, num, transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            ax.plot([],[], label=f'MAD: {mad_logr:.2f}', alpha=0)  # dummy plot for legend
            ax.legend()
            
        #plt.suptitle(r'$N_\sigma$ distribution for each MIRI filter', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # Save single combined figure
        filename = os.path.join(out_dir, 'log_ratios.png')
        plt.savefig(filename, dpi=300)
        plt.show()
    


    if plot_chi2red:
        
        # Compute reduced chi^2 per galaxy
        
        chi2_red = chi2_df['reduced_chi2']
        
        # --- 1. Clipping & Statistics ---
        chi2_red = chi2_df['reduced_chi2']
        q95 = chi2_red.quantile(0.95)
        filtered = chi2_df[chi2_red <= q95]
        
        mean_val = chi2_red.mean()
        median_val = chi2_red.median()
        mad_val = median_abs_deviation(chi2_red.dropna())

        fig, axes = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios":[1.25,0.75]})
        
        # --- Left: Nsigma vs sSFR
        counts, bins, patches = axes[0].hist(filtered['reduced_chi2'], bins=25, alpha=0.7, edgecolor='black', range=(0, q95))
        
        axes[0].set_xlabel(r'Reduced $\chi^2$')
        axes[0].set_ylabel('Number of galaxies')
        
        # Count how many chi2 values are in the histogram
        chi2_values = len(chi2_df)
        num = f'\nN = {len(filtered)}/{chi2_values}\n(95th perctile)'
        # Annotate in the top-right corner (adjust x,y if needed)
        
        # plot vertical lines for mean and median
        # Plot vertical lines for mean and median
        ymax = counts.max() * 1.1
        axes[0].vlines(mean_val, ymin=0, ymax=ymax, color='red', alpha=0.8, linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[0].vlines(median_val, ymin=0, ymax=ymax, color='darkred', alpha=0.8, linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
        axes[0].plot([],[], label=num, alpha=0)  # dummy plot for legend
        axes[0].set_ylim(0, ymax)
        axes[0].legend()    
        

        # Scatter plot with filtered data
        #plt.scatter(filtered['n_filters'], filtered['reduced_chi2'], alpha=0.7)    
        axes[1].scatter(filtered['n_filters'], filtered['reduced_chi2'], alpha=0.7)    
        axes[1].set_xlabel('Number of photometric data points')
        axes[1].set_ylabel(r'Reduced $\chi^2$')
        #plt.title(r'Reduced $\chi^2$ vs. number of MIRI bands')
        axes[1].axhline(1, color='orange', linestyle='--', label='Unity')
        axes[1].legend()    
        
        plt.tight_layout()
        filename = os.path.join(out_dir, 'reduced_chi2.png')
        plt.savefig(filename, dpi=300)
        print(f"✅ Saved reduced chi^2 plots to {filename}")
        plt.show()
        
        threshold = 30  # user-specified value
        high_chi2_ids = chi2_df.loc[chi2_df['reduced_chi2'] > threshold, 'galaxy_id'].tolist()
        print(f"{len(high_chi2_ids)} galaxies have reduced χ² > {threshold}")
        print("These galaxies are:", high_chi2_ids)
        print("Their χ² values are:", chi2_df.loc[chi2_df['reduced_chi2'] > threshold, 'reduced_chi2'].tolist())

    return 
    
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



def plot_main_sequence_from_pickles(pickle_dir, zred_ms, ms_type='Leja', out_dir='/Users/benjamincollins/University/Master/Red_Cardinal/prospector_v2/sample_plots/'):
    """
    Reads pickle files and plots the star-forming main sequence.
    """
    
    pickle_files = glob.glob(f'{pickle_dir}/*.pkl')
    
    # Lists to store extracted data
    masses = []
    sfrs = []
    ids = []
    fit_qual = []
    
    # 1. Extraction Loop
    for f_path in pickle_files:
        with open(f_path, 'rb') as f:
            data = pkl.load(f)
        
        props = data['galaxy_properties']
        ids.append(data['id'])
        masses.append(10**props['logmass'])
        sfrs.append(props['sfr_100myr'])
        fit_qual.append(data['fit_quality'])

    # Convert to arrays
    logM = np.log10(masses)
    logSFR_sample = np.log10(sfrs)
    logM_grid = np.linspace(np.min(logM)-0.5, np.max(logM)+0.5, 200)

    # 2. Main Sequence Calculations
    t = cosmo.age(zred_ms).to(u.Gyr).value

    if ms_type == 'Leja':        
        a = -0.06707 + 0.3684 * zred_ms - 0.1047 * zred_ms**2
        b = 0.8552 - 0.1010 * zred_ms - 0.001816 * zred_ms*2
        c = 0.2148 + 0.8137 * zred_ms - 0.08052 * zred_ms**2
        log_Mt = 10.29 - 0.1284 * zred_ms + 0.1203 * zred_ms**2
        logSFR_MS = np.where(logM_grid > log_Mt, a*(logM_grid-log_Mt)+c, b*(logM_grid-log_Mt)+c)

    elif ms_type == 'Speagle':
        # Speagle+14 coefficients
        slope = 0.84 - 0.026 * t
        intercept = -(6.51 - 0.11 * t)

        # 1-sigma errors
        slope_err = 0.02 + 0.003 * t
        intercept_err = 0.24 + 0.03 * t

        # Main sequence
        logSFR_MS = slope * logM_grid + intercept
        logSFR_high = (slope + slope_err) * logM_grid + (intercept + intercept_err)
        logSFR_low  = (slope - slope_err) * logM_grid + (intercept - intercept_err)

    else:
        print("⚠️ Error: ms_type needs to be either 'Leja' or 'Speagle'. Aborting.")   
        return 

    # 3. Plotting Logic
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap('YlOrBr')
    # Colourise by detection fraction!
    n_obs = []
    n_det = []
    for key, fq in fit_qual.values():
        # Count filters (keys like 'F770W') excluding global stats like 'chi2_red'
        count = sum(1 for k, v in fq.items() if isinstance(v, dict))
        n_obs.append(count)
        
        # Count only filters where SNR > 3
        count = sum(1 for k, v in fq.items() 
                    if isinstance(v, dict) and 
                    v.get('snr', 0) > 3.0)
        n_det.append(count)
    
    N_detected = np.array(n_det)
    N_available = np.array(n_obs)
    f_det = N_detected / N_available  # fraction 0-1  
    
    # Scatter plot
    sc = ax.scatter(logM, logSFR_sample, c=f_det, cmap=cmap, s=60, edgecolor='black', norm=Normalize(vmin=0, vmax=1))
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    cbar.set_label('Number of MIRI detections (relative)')
        
    if ms_type == 'Leja':
        # MS line
        ax.plot(logM_grid, logSFR_MS, 'k--', color='black', alpha=0.7, label=f'Leja+22 MS (z={zred_ms:.2f})', linewidth=2)
        ax.plot(logM_grid, logSFR_MS - 1.0, 'k:', alpha=0.7, label='1 dex below MS')
        filename = 'mein_sequence_leja.png'
        #ax.plot(10.720281148198858, 0.5723139475044815, color='red', alpha=0.2)
    elif ms_type == 'Speagle':
        # MS line and shaded 1-sigma region
        ax.plot(logM_grid, logSFR_MS, 'k--', alpha=0.5, label=f'Speagle+14 MS (z={zred_ms:.2f})')
        ax.fill_between(logM_grid, logSFR_low, logSFR_high, color='gray', alpha=0.15, label='1σ uncertainty')
        filename = 'main_sequence_speagle.png'
        
    # Labels and legend
    ax.set_xlabel('log$_{10}$(M$_*$/M$_\\odot$)', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\mathrm{SFR} / $M$_\odot\,\mathrm{yr}^{-1})$', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as {save_path}")
        

        
        
        
def plot_main_sequence(masses, sfr100, zred_ms, detections, data=None, ms_type='Speagle', color_scheme='viridis', gradient='absolute', save_path='main_sequence.png'):
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

    # Mass grid in log10(M)
    logM_grid = np.linspace(np.min(logM)-0.1, np.max(logM)+0.1, 200)

    logSFR_sample = np.log10(sfr100) # SFR in solar masses per year

    # Median redshift for MS line
    t = cosmo.age(zred_ms).to(u.Gyr).value  # cosmic time in Gyr

    if ms_type == 'Leja':        
        
        a = -0.06707 + 0.3684 * zred_ms - 0.1047 * zred_ms**2
        b = 0.8552 - 0.1010 * zred_ms - 0.001816 * zred_ms*2
        c = 0.2148 + 0.8137 * zred_ms - 0.08052 * zred_ms**2
        log_Mt = 10.29 - 0.1284 * zred_ms + 0.1203 * zred_ms**2
        
        logSFR_MS = np.zeros_like(logM_grid)
        
        for i, logM_i in enumerate(logM_grid):
            if logM_i > log_Mt:
                logSFR_MS[i] = a * (logM_i - log_Mt) + c
            else:
                logSFR_MS[i] = b * (logM_i - log_Mt) + c        
        
    elif ms_type == 'Speagle':
        # Speagle+14 coefficients
        slope = 0.84 - 0.026 * t
        intercept = -(6.51 - 0.11 * t)

        # 1-sigma errors
        slope_err = 0.02 + 0.003 * t
        intercept_err = 0.24 + 0.03 * t

        # Main sequence
        logSFR_MS = slope * logM_grid + intercept
        logSFR_high = (slope + slope_err) * logM_grid + (intercept + intercept_err)
        logSFR_low  = (slope - slope_err) * logM_grid + (intercept - intercept_err)
    
    # Now let's introduce the plot   
    
    if gradient == 'absolute':
        fig, ax = plt.subplots(figsize=(6, 4))
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
        cbar.set_label('Number of MIRI detections (absolute)')
        cbar.ax.set_yticklabels([str(i) for i in range(5)])    # ensure labels 0..4
        save_path += f'sfms_{gradient}'
        
    elif gradient == 'relative':  
        fig, ax = plt.subplots(figsize=(7, 4))
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
        cbar.set_label('Number of MIRI detections (relative)')
        save_path += f'sfms_{gradient}'
    
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
        cbar.set_label(rf'$\log_{{10}}(F) [µJy]$', fontsize=9)        #cbar.set_label("log$_{10}$(F770W flux) [μJy]")
        #ax.set_xlim(8, 13)
        ax.set_ylim(-1, 4)
        save_path += f'sfms_{gradient}'
        
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
        save_path += f'sfms_{gradient}'
        
    else:
        print("⚠️Gradient has to be set to either absolute or relative.")
        return None

    if ms_type == 'Leja':
        # MS line
        ax.plot(logM_grid, logSFR_MS, 'k--', color='black', alpha=0.7, label=f'Leja+22 MS (z={zred_ms:.2f})', linewidth=2)
        ax.plot(logM_grid, logSFR_MS - 1.0, 'k:', alpha=0.7, label='1 dex below MS')
        save_path += f'_Leja.png'
        #ax.plot(10.720281148198858, 0.5723139475044815, color='red', alpha=0.2)
    elif ms_type == 'Speagle':
        # MS line and shaded 1-sigma region
        ax.plot(logM_grid, logSFR_MS, 'k--', alpha=0.5, label=f'Speagle+14 MS (z={zred_ms:.2f})')
        ax.fill_between(logM_grid, logSFR_low, logSFR_high, color='gray', alpha=0.15, label='1σ uncertainty')
        save_path += f'_Speagle.png'

    # Labels and legend
    ax.set_xlabel('log$_{10}$(M$_*$/M$_\\odot$)', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\mathrm{SFR} / $M$_\odot\,\mathrm{yr}^{-1})$', fontsize=14)
    ax.legend()
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
        fig, ax = plt.subplots(figsize=(6, 4))
        cmap = plt.get_cmap(color_scheme, 5)  # 5 discrete colors: 0,1,2,3,4
        bounds = np.arange(-0.5, 5.5, 1)
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=5)
        N_detected = []
        for det in detections: 
            N_detected.append(sum(det.values()))
        N_detected = np.array(N_detected)
        sc = plt.scatter(zreds, logmasses, c=N_detected, cmap=cmap, norm=norm, s=40, alpha=0.8, edgecolor='black')
        
        cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(0, 5))   # ticks at 0,1,2,3,4
        cbar.set_label('# of MIRI detections (absolute)', fontsize=14)
        cbar.ax.set_yticklabels([str(i) for i in range(5)])    # ensure labels 0..4
        
        print("Detected galaxies: ", np.sum(N_detected > 0), " out of ", len(zreds))
        
        detected = N_detected > 0
        print("Detected galaxies: ", np.sum(detected), " out of ", len(zreds))
        # Add sample statistics as text
        #ax.text(0.61, 0.98, f'Total: {len(zreds)} galaxies\nDetected: {np.sum(detected)} ({100*np.sum(detected)/len(zreds):.1f}%)', 
        #        transform=ax.transAxes, verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        save_path = save_path + f'zM_{gradient}_paper_v2.png'
        
    elif gradient == 'relative': 
        fig, ax = plt.subplots(figsize=(6, 4)) 
        N_detected = []
        N_available = []
        for det in detections: 
            N_detected.append(sum(det.values()))
            N_available.append(len(det.values()))
        N_detected = np.array(N_detected)
        N_available = np.array(N_available)
        f_det = N_detected / N_available  # fraction 0-1  
        
        print("Detected galaxies: ", np.sum(N_detected > 0), " out of ", len(zreds))
        
        cmap = plt.get_cmap(color_scheme)

        sc = ax.scatter(zreds, logmasses, c=f_det, cmap=cmap, s=40, edgecolor='black', norm=Normalize(vmin=0, vmax=1))

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
        cbar.set_label('# of MIRI detections (relative)', fontsize=14)
        
        detected = N_detected > 0
        # Add sample statistics as text
        #ax.text(0.6, 0.98, f'Total: {len(zreds)} galaxies\nDetected: {np.sum(detected)} ({100*np.sum(detected)/len(zreds):.1f}%)', 
        #        transform=ax.transAxes, verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        save_path = save_path + f'zM_{gradient}_paper_v2.png'
        
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
        print(flux_array)
        zreds = np.array(zreds)[mask]
        logmasses = np.array(logmasses)[mask]

        # Convert flux to log scale
        log_flux_array = np.log10(flux_array)

        if gradient == 'f770w': color_scheme = "Blues"
        elif gradient == 'f1000w': color_scheme = "Greens"
        elif gradient == 'f1800w': color_scheme = "Oranges"
        elif gradient == 'f2100w': color_scheme = "Reds"

        print("Non-finite zreds:", np.sum(~np.isfinite(zreds)))
        print("Non-finite logmasses:", np.sum(~np.isfinite(logmasses)))
        print("Non-finite log_flux_array:", np.sum(~np.isfinite(log_flux_array)))

        invalid = ~np.isfinite(zreds) | ~np.isfinite(logmasses) | ~np.isfinite(log_flux_array)
        print("Indices of invalid points:", np.where(invalid)[0])
        print("Example invalid entries:")
        print("zreds:", zreds[invalid])
        print("log_flux_array:", log_flux_array[invalid])

        # Create scatter plot coloured by log flux
        sc = ax.scatter(zreds, logmasses, c=log_flux_array, cmap=color_scheme, s=60, alpha=0.8, edgecolor='black')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(rf'$\log_{{10}}(F) [µJy]$', fontsize=12)        #cbar.set_label("log$_{10}$(F770W flux) [μJy]")
        
        stats_text = f'N = {len(zreds)}'
        ax.text(0.82, 0.9, stats_text, transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        save_path = save_path + f'zM_{gradient}.png'
        ax.set_title(f'{band}', fontsize=14)
        #ax.set_xlim(1.5, 3.75)
        #ax.set_ylim(9, 12)
    
    elif gradient in ['nsig_f770w', 'nsig_f1000w', 'nsig_f1800w', 'nsig_f2100w']:
        fig, ax = plt.subplots(figsize=(6, 4))
        nsig_array = []
        mask = []
        band = gradient.split('_')[1].upper()
        
        for nsig_dict in data:
            if band in nsig_dict:
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
        stats_text = f'N = {len(zreds)}'
        ax.text(0.82, 0.9, stats_text, transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(rf'$N_\sigma$ ({band})')
        save_path = save_path + f'zM_{gradient}_r.png'
        ax.set_xlim(1.5, 3.75)
        ax.set_ylim(8.75, 12)
        
    
    else:
        print("⚠️Gradient has to be set to either absolute, relative or flux.")
        return None
          
        
    ax.set_xlabel('Redshift (z)', fontsize=16)
    ax.set_ylabel('log$_{10}$(M$_*$/M$_\\odot$)', fontsize=16)
    #ax.set_title('z-M Parameter Space', fontsize=16, fontweight='bold', pad=20)
    #ax.set_xticklabels(ax.get_xticks(), fontsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    #ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as {save_path}")


def plot_nsigma_vs_params(nsig1, nsig2, log_ssfr, dust, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes = axes.flatten()
    # --- Left: Nsigma vs sSFR
    sc1 = axes[0].scatter(log_ssfr, nsig1, c=dust, cmap="viridis", alpha=0.7, edgecolor='black')
    axes[0].axhline(0, ls="--", c="grey")
    axes[0].set_xlabel(r'$\log(\mathrm{sSFR}_{100}\,[\mathrm{yr}^{-1}])$')
    axes[0].set_ylabel(r'$N_\sigma$')
    axes[0].set_title(f'F1800W: ' + r'$\mathrm{N_\sigma}$ vs sSFR')
    axes[0].set_xlim(-9.75, -7.75)
    cb1 = fig.colorbar(sc1, ax=axes[0])
    cb1.set_label("Dust attenuation (dust2)")
    
    # --- Right: Nsigma vs Dust
    sc2 = axes[1].scatter(dust, nsig1, c=log_ssfr, cmap="plasma", alpha=0.7, edgecolor='black')
    axes[1].axhline(0, ls="--", c="grey")
    axes[1].set_xlabel(r'Dust attenuation $A_V$')
    axes[1].set_ylabel(r'$N_\sigma$')
    axes[1].set_title(f'F1800W: ' + r'$\mathrm{N_\sigma}$ vs $\mathrm{A_V}$')
    axes[1].set_xlim(-0.1, 3)
    cb2 = fig.colorbar(sc2, ax=axes[1])
    cb2.set_label(r'$\log(\mathrm{sSFR}_{100})$')
    
    # --- Left: Nsigma vs sSFR
    sc3 = axes[2].scatter(log_ssfr, nsig2, c=dust, cmap="viridis", alpha=0.7, edgecolor='black')
    axes[2].axhline(0, ls="--", c="grey")
    axes[2].set_xlabel(r'$\log(\mathrm{sSFR}_{100}\,[\mathrm{yr}^{-1}])$')
    axes[2].set_ylabel(r'$N_\sigma$')
    axes[2].set_title(f'F2100W: ' + r'$\mathrm{N_\sigma}$ vs sSFR')
    axes[2].set_xlim(-9.75, -7.75)
    cb1 = fig.colorbar(sc1, ax=axes[2])
    cb1.set_label("Dust attenuation (dust2)")
    
    # --- Right: Nsigma vs Dust
    sc4 = axes[3].scatter(dust, nsig2, c=log_ssfr, cmap="plasma", alpha=0.7, edgecolor='black')
    axes[3].axhline(0, ls="--", c="grey")
    axes[3].set_xlabel(r'Dust attenuation $A_V$')
    axes[3].set_ylabel(r'$N_\sigma$')
    axes[3].set_title(f'F2100W: ' + r'$\mathrm{N_\sigma}$ vs $\mathrm{A_V}$')
    axes[3].set_xlim(-0.1, 3)
    cb2 = fig.colorbar(sc2, ax=axes[3])
    cb2.set_label(r'$\log(\mathrm{sSFR}_{100})$')
    
    for ax in axes:
        ax.set_ylim(-6,6)
    
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
        #for band in ["F770W", "F1000W", "F1800W", "F2100W"]
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
    
