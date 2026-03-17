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


quiescent = [7549, 8013, 8469, 9395, 10128, 10339, 10400, 10565, 10592, 11142, 11494, 16419, 18668, 21477]
below_ms = [10600, 18977, 21451]
no_spec = [9517, 9809, 11051, 11451, 12133, 17713, 17984, 20195, 20693, 20720, 21472, 22990]

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
        'miri':    {'color': 'firebrick',    'marker': 'p', 'edgecolor': 'black',    'alpha': 0.7, 'label': 'JWST MIRI (not used in fit)', 'ms': 10}
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


def plot_reconstructed_fit(filename, plot_dir=None):
    
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
    ax.set_xlabel('Observed Wavelength [µm]', fontsize=13)
    ax.set_ylabel('Flux [µJy]', fontsize=13)
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
    
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        fname = os.path.join(plot_dir, f'{gid}.png')
        plt.savefig(fname)
        print(f"Plot saved to {fname}")
    plt.show()
    plt.close()
    
    return


def plot_miri_fit(filename, plot_dir=None):
    
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
    phot_wave = model['phot_wave']
    
    obs = fit_data['obs']
    maggies_to_muJy = fit_data['maggies_to_muJy']
    
    # Convert to µJy
    lower_scaled = spec_16th * maggies_to_muJy    
    median_scaled = spec_median * maggies_to_muJy
    upper_scaled = spec_84th * maggies_to_muJy
    spec_scaled = spec_best * maggies_to_muJy
    
    phot_wave_microns = phot_wave * 1e-4  # convert to µm
    
    phot_scaled = phot * maggies_to_muJy
    
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
    
    #########  PLOT MEASURED PHOTOMETRY    #########

    # Define the style per instrument
    instrument_styles = {
        'acs':     {'color': 'royalblue',   'marker': 'o', 'edgecolor': 'black', 'label': 'HST/ACS', 'ms': 10},
        'wfc3':    {'color': 'limegreen',  'marker': 'o', 'edgecolor': 'black', 'label': 'HST/WFC3', 'ms': 10},
        'nircam':  {'color': 'orange', 'marker': 'p', 'edgecolor': 'black',    'alpha': 0.7, 'label': 'JWST/NIRCam', 'ms': 10},
        'miri':    {'color': 'firebrick',    'marker': 'p', 'edgecolor': 'black',    'alpha': 0.7, 'label': 'JWST/MIRI', 'ms': 10}
    }
    
    # Get current labels to prevent duplicates
    _, labels = ax.get_legend_handles_labels()
    
    for i, filt in enumerate(obs['filters_all']):
        
        wave = obs['phot_wave'][i] * 1e-4  # convert to µm
        flux = obs['maggies'][i] * maggies_to_muJy  # µJy
        err  = obs['maggies_unc'][i] * maggies_to_muJy  # µJy
        
        name = filt.name.lower()
        
        # Improved Upper Limit Logic for MIRI
        uplims = False
        
        if (flux / err < 3.0):
            uplims = True
            flux = 3 * err # Plot at 3-sigma
            err = flux * 0.4 # Small arrow size for visualisation
        
        if 'acs_wfc' in name:
            style = instrument_styles['acs']
        elif 'wfc3_ir' in name:
            style = instrument_styles['wfc3']
        elif 'miri' in name or any(m in name for m in ['f770w', 'f1000w', 'f1800w', 'f2100w']):
            style = instrument_styles['miri']            
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
    ax.set_xlabel('Observed Wavelength [µm]', fontsize=13)
    ax.set_ylabel('Flux [µJy]', fontsize=13)
    ax.set_xlim(0.4, 35)#200)    # Change x range    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc="upper left")
    #ax.set_title(f"Galaxy {objid} at z={np.round(zred,2)}", fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=13)
    
    zred_rounded = np.round(zred,2)
    #plt.title(f"Galaxy {objid} at z={zred_rounded}")
    plt.tight_layout()
    
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        fname = os.path.join(plot_dir, f'{gid}.png')
        plt.savefig(fname)
        print(f"Plot saved to {fname}")
    plt.show()
    plt.close()
    
    return


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



def plot_sample_from_pickles(pickle_dir, out_dir='/Users/benjamincollins/University/Master/Red_Cardinal/prospector/sample_plots/'):
    """
    Plot z-M parameter space colour-coded by nsigma.
    """
    
    pickle_files = glob.glob(f'{pickle_dir}/*.pkl')
    
    # Color schemes based on band
    cmaps = {'F770W': 'Blues', 'F1000W': 'Greens', 'F1800W': 'Oranges', 'F2100W': 'Reds'}
    
    # Lists to store extracted data
    logmasses = []
    zreds = []
    ids = []
    fit_qual = []
    
    # 1. Extraction Loop
    for f_path in pickle_files:
        with open(f_path, 'rb') as f:
            data = pkl.load(f)
        
        if data['id'] in no_spec:
            continue
        
        props = data['galaxy_properties']
        ids.append(data['id'])
        logmasses.append(props['logmass'])
        zreds.append(data['zred'])
        fit_qual.append(data['fit_quality'])
    
    # Convert to numpy arrays for masking
    zreds = np.array(zreds)
    logmasses = np.array(logmasses)
    
    os.makedirs(out_dir, exist_ok=True)        
    
    for band in cmaps.keys():
        fig, ax = plt.subplots(figsize=(6, 4))
        
        flux_array = []
        mask = []
        
        for fq in fit_qual:
            if band in fq:
                snr = fq[band]['snr']                
                if snr > 3.0:
                    # We use obs_flux from your new fit_quality dict
                    flux_array.append(fq[band]['obs_flux'])
                    mask.append(True)
                else:
                    mask.append(False)
            else:
                mask.append(False)
                
        mask = np.array(mask)
              
        # Apply mask to all plotting arrays
        plot_z = zreds[mask]
        plot_m = logmasses[mask]
        # Convert flux to log10(muJy) - assuming it's already muJy from your pickle loop
        plot_color = np.log10(np.array(flux_array)*1e6)
        
        color_scheme = cmaps.get(band, 'viridis')
        
        sc = ax.scatter(plot_z, plot_m, c=plot_color, cmap=color_scheme, s=60, alpha=0.8, edgecolor='black')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(rf'$\log_{{10}}(Flux) [\mu Jy]$', fontsize=14)
        
        filename = f'zM_flux_{band}.png'
        
        stats_text = f'N = {len(plot_z)}'
        ax.text(0.83, 0.92, stats_text, transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        save_path = os.path.join(out_dir, filename)
        ax.set_title(f'{band}', fontsize=14)
        ax.set_xlim(1.25, 3.75)
        ax.set_ylim(8.5, 12.5)
        
        ax.set_xlabel('Redshift (z)', fontsize=16)
        ax.set_ylabel('log$_{10}$(M$_*$/M$_\\odot$)', fontsize=16)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_yticklabels(ax.get_yticks(), fontsize=12)
        ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved figure to {save_path}")

        
        
    
        fig, ax = plt.subplots(figsize=(6, 4))
        nsig_array = []
        mask = []
        
        for fq in fit_qual:
            if band in fq:
                nsig_array.append(fq[band]['n_sigma'])
                mask.append(True)
            else:
                mask.append(False)
                
        mask = np.array(mask)
        plot_z = zreds[mask]
        plot_m = logmasses[mask]
        plot_color = np.array(nsig_array)
        
        # Use a diverging colormap for residuals (Red-Blue)
        sc = ax.scatter(plot_z, plot_m, c=plot_color, cmap='seismic', s=60, 
                        alpha=0.8, edgecolor='black', vmin=-6, vmax=6)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(rf'$N_\sigma$', fontsize=14)
        
        filename = f'zM_nsigma_{band}.png'
            
        stats_text = f'N = {len(plot_z)}'
        ax.text(0.83, 0.92, stats_text, transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        save_path = os.path.join(out_dir, filename)
        ax.set_title(f'{band}', fontsize=14)
        ax.set_xlim(1.25, 3.75)
        ax.set_ylim(8.5, 12.5)
        
        ax.set_xlabel('Redshift (z)', fontsize=16)
        ax.set_ylabel('log$_{10}$(M$_*$/M$_\\odot$)', fontsize=16)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.set_yticklabels(ax.get_yticks(), fontsize=12)
        ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved figure to {save_path}")


def plot_nsigma_vs_params(pickle_dir, out_dir='/Users/benjamincollins/University/Master/Red_Cardinal/prospector/sample_plots/'):
    """
    Plots N_sigma residuals against physical parameters (sSFR and Dust)
    for MIRI filters F1800W and F2100W.
    """
    
    pickle_files = glob.glob(f'{pickle_dir}/*.pkl')

    # 1. Efficient Extraction
    # We use a nested dict to keep everything organized by band
    bands = ['F1800W', 'F2100W']
    extracted = {b: {'nsig': [], 'ssfr': [], 'dust': []} for b in bands}

    for f_path in pickle_files:
        with open(f_path, 'rb') as f:
            data = pkl.load(f)
        
        # Exclude quiescent galaxies of Bugiani et al. (2025)
        if data['id'] in [7549, 8013, 8469, 9395, 10128, 10339, 10400, 10565, 10592, 11142, 11494, 16419, 18668, 21477]:
            continue
        
        # Below -1 dex of the SFMS of Leja et al. (2022)
        if data['id'] in [21472, 12133, 18977, 20720,  9809]:
            continue
        
        fq = data.get('fit_quality', {})
        props = data.get('galaxy_properties', {})
        
        # Pre-calculate common parameters to save CPU cycles
        log_m = props.get('logmass', np.nan)
        sfr = props.get('sfr_100myr', np.nan)
        ssfr = np.log10(sfr / 10**log_m) if log_m and sfr else np.nan
        dust_val = props.get('dust2', np.nan)

        for b in bands:
            if b in fq:
                extracted[b]['nsig'].append(fq[b]['n_sigma'])
                extracted[b]['ssfr'].append(ssfr)
                extracted[b]['dust'].append(dust_val)

    # 2. Setup Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # Define the configurations for each of the 4 subplots
    # (Band, X-key, Color-key, cmap)
    configs = [
        ('F1800W', 'ssfr', 'dust', 'jet'),
        ('F1800W', 'dust', 'ssfr', 'plasma'),
        ('F2100W', 'ssfr', 'dust', 'jet'),
        ('F2100W', 'dust', 'ssfr', 'plasma')
    ]

    for i, (band, x_key, c_key, cmap) in enumerate(configs):
        ax = axes[i]
        d = extracted[band]
        
        # Convert to numpy arrays for the specific band
        x = np.array(d[x_key])
        y = np.array(d['nsig'])
        c = np.array(d[c_key])

        # Remove any NaNs that might have sneaked in
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
        
        sc = ax.scatter(x[mask], y[mask], c=c[mask], cmap=cmap, 
                        alpha=0.7, edgecolor='black', s=55)

        # Labels and Style
        ax.axhline(0, ls="--", c="grey", alpha=0.5)
        ax.set_title(f"{band}: $N_\sigma$ vs {x_key.upper()}", fontsize=14)
        ax.set_ylabel(r"$N_\sigma$", fontsize=14)
        ax.set_ylim(-6, 6)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)

        # X-Axis specific formatting
        if x_key == 'ssfr':
            ax.set_xlabel(r"$\log(\mathrm{sSFR}_{100})$", fontsize=14)
            ax.set_xlim(-11, -7.5)
        else:
            ax.set_xlabel(r"Dust Attenuation ($\mathrm{A_V}$)", fontsize=14)
            ax.set_xlim(-0.1, 3.0)

        # Colorbar
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r"$\mathrm{A_V}$" if c_key == 'dust' else "sSFR", fontsize=14)
        cb.ax.tick_params(labelsize=12)
        
        # Count label
        ax.text(0.03, 0.92, f'N = {np.sum(mask)}', transform=ax.transAxes, 
                fontsize=13, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    filename = os.path.join(out_dir, 'nsigma_vs_params_v2.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as {filename}")
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
    
