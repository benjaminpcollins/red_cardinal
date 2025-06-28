#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIRI Utils Astrometric Offset Module
====================================

This module provides functions for computing and visualising astrometric offsets between NIRCam and MIRI cutouts, 
which is critical for ensuring accurate positional alignment in multi-wavelength analyses. 
The module includes tools for:
- Computing centroids for cutouts with optional smoothing to reduce noise.
- Saving alignment figures to visually inspect centroid matching.
- Calculating RA/Dec offsets for entire galaxy catalogues.
- Exporting statistical summaries of offset distributions.
- Shifting MIRI FITS files to correct systematic positional offsets.

Dependencies
------------
Dependencies:
- astropy (for FITS I/O, WCS transformations, and coordinate calculations)
- matplotlib (for visualisation)
- photutils (for centroiding)
- scipy (for image smoothing)
- numpy (for array manipulations)
- json (for exporting statistics)
 
Requirements
------------
- astropy
- numpy
- scipy

Usage
-----
- Ensure cat is defined with the expected structure (including 'id', 'ra', 'dec' columns and offset placeholders).
- Call the compute_offset function to compute centroids and offsets for each galaxy in the catalogue.
- Use save_alignment_figure for visual verification of alignment.
- Export summary statistics with write_offset_stats.
- Shift MIRI cutouts to correct for systematic offsets using shift_miri_fits.

Note
----
The functions in this module assume the use of PRIMER or COSMOS-Web data products. Paths should be adjusted accordingly.


Author: Benjamin P. Collins
Date: May 15, 2025
Version: 2.0
"""

import numpy as np
import scipy
import os
import json
import warnings

from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, FITSFixedWarning
import astropy.units as u
from astropy.nddata import Cutout2D
from photutils import centroids
from .cutout_tools import load_cutout

# Suppress common WCS-related warnings that don't affect functionality
warnings.simplefilter("ignore", category=FITSFixedWarning)

def compute_centroid(cutout, smooth_sigma, good_frac_cutout, smooth_miri):
    """Compute the centroid of a given cutout image using quadratic fitting.
    
    Parameters:
    cutout (Cutout2D): The 2D image cutout for centroid computation.
    smooth_sigma (float): Sigma for Gaussian smoothing.
    good_frac_cutout (float): Fraction of the cutout used for centroid fitting.
    smooth_miri (bool): Whether to apply additional smoothing to MIRI data.
    
    Returns:
    SkyCoord: The computed centroid in world coordinates, or None if the centroid could not be determined.
    """
    if cutout is None:
        return None

    # Decide whether to smooth MIRI or not
    if smooth_miri == True:
        smoothed_data = scipy.ndimage.gaussian_filter(cutout.data, smooth_sigma)
    else: 
        smoothed_data = cutout.data
    
    # Makes sure the boxsize is an odd number
    search_boxsize = int(np.floor(good_frac_cutout * cutout.shape[0]) // 2 * 2 + 1)

    centroid_pix = centroids.centroid_quadratic(
        smoothed_data,
        xpeak=cutout.shape[0] // 2,
        ypeak=cutout.shape[1] // 2,
        search_boxsize=search_boxsize,
        fit_boxsize=5
    )

    return cutout.wcs.pixel_to_world(centroid_pix[0], centroid_pix[1]) if not np.isnan(centroid_pix).any() else None

def save_alignment_figure(g, cutout_nircam, cutout_miri, centroid_nircam, centroid_miri, output_dir, survey, filter):
    """Save a side-by-side comparison of NIRCam and MIRI cutouts, with centroids marked.
    
    Parameters:
    g (dict): Galaxy metadata including ID.
    cutout_nircam (Cutout2D): NIRCam cutout image.
    cutout_miri (Cutout2D): MIRI cutout image.
    centroid_nircam (SkyCoord): Centroid of the NIRCam cutout.
    centroid_miri (SkyCoord): Centroid of the MIRI cutout.
    output_dir (str): Directory to save the figure.
    survey (str): Survey name.
    filter (str): MIRI filter used.
    """
    
    fig, axs = plt.subplots(1, 2, figsize=[10, 5])

    axs[0].imshow(scipy.ndimage.gaussian_filter(cutout_nircam.data, 1.0), origin='lower')
    axs[0].plot(*cutout_nircam.wcs.world_to_pixel(centroid_nircam), 'x', color='red')
    axs[0].set(title=f"NIRCam F444W Reference {g['id']}")

    axs[1].imshow(scipy.ndimage.gaussian_filter(cutout_miri.data, 1.0), origin='lower')
    axs[1].plot(*cutout_miri.wcs.world_to_pixel(centroid_miri), 'o', color='orange')
    axs[1].set(title=f"MIRI {filter} Cutout {g['id']}")
    
    # show expected position of the centroid
    expected_position_pix = cutout_miri.wcs.world_to_pixel(centroid_nircam)
    axs[1].plot(expected_position_pix[0], expected_position_pix[1], 'x', color='red')

    output_path = os.path.join(output_dir, f"{g['id']}_{filter}_offset_{survey}.pdf")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path)
    plt.close()

    #print(f"Saved figure: {output_path}")


def compute_offset(cutout_folder, output_folder, cat, survey, filter, obs="", save_fig=True, smooth_miri=True, use_filters=False):
    """Computes the astrometric offset between NIRCam and MIRI for each galaxy."""
    
    for i, g in enumerate(cat):
        #print(f"Processing galaxy {g['id']}...")

        ref_position = SkyCoord(ra=g['ra'], dec=g['dec'], unit=u.deg)
        cutout_size = (2.5 * u.arcsec, 2.5 * u.arcsec)
        smooth_sigma, good_frac_cutout = 1.0, 0.7

        if g['id'] == 21451: # exclude bright source nearby
            good_frac_cutout = 0.4
        if g['id'] == 9986: # exclude bright source nearby
            good_frac_cutout = 0.4
        if g['id'] == 11451: # exclude bright source nearby
            good_frac_cutout = 0.4
        
        # Load MIRI cutout
        cutout_miri_path = os.path.join(cutout_folder, f"{g['id']}_{filter}_cutout_{survey}{obs}.fits")
        miri_data, miri_wcs = load_cutout(cutout_miri_path)
        if miri_data is None:
            continue
        cutout_miri = Cutout2D(miri_data, ref_position, cutout_size, wcs=miri_wcs)

        # Load NIRCam cutout
        nircam_path = f"/home/bpc/University/master/Red_Cardinal/NIRCam/F444W_cutouts/{g['id']}_F444W_cutout.fits"
        nircam_data, nircam_wcs = load_cutout(nircam_path)
        if nircam_data is None:
            continue
        cutout_nircam = Cutout2D(nircam_data, ref_position, cutout_size, wcs=nircam_wcs)

        # Compute centroids
        centroid_nircam = compute_centroid(cutout_nircam, smooth_sigma, good_frac_cutout, smooth_miri)
        centroid_miri = compute_centroid(cutout_miri, smooth_sigma, good_frac_cutout, smooth_miri)

        if centroid_nircam is None or centroid_miri is None:
            print("Centroid not found for one or both cutouts. Skipping.")
            continue
        
        if save_fig == True:
            # Save alignment figure
            output_dir = os.path.join(output_folder, f"{survey}{obs}/")
            os.makedirs(output_dir, exist_ok=True)
            save_alignment_figure(g, cutout_nircam, cutout_miri, centroid_nircam, centroid_miri, output_dir, survey, filter)

        if use_filters == True:
            filter_l = '_' + filter.lower()
        else:
            filter_l = ''
        
        # Compute offsets
        dra, ddec = centroid_nircam.spherical_offsets_to(centroid_miri)
        cat[f'{survey}{obs}{filter_l}_dra'][i] = dra.to(u.arcsec).value
        cat[f'{survey}{obs}{filter_l}_ddec'][i] = ddec.to(u.arcsec).value

        #print(f"Offset: ΔRA = {dra.to(u.arcsec)}, ΔDec = {ddec.to(u.arcsec)}")



def write_offset_stats(df, dra, ddec, output_dir, survey, filter):
    """Write mean and std of astrometric offsets to a JSON file.

    Args:
        df (pandas.DataFrame): The dataframe with 'dra' and 'ddec' columns.
        survey (string): Name of the survey (primer or cweb)
        obs (string): Number of the observation
        output_dir (str): Directory where the stats file will be saved.
        filename (str): Name of the JSON file.
    """

    # Calculate means and standard deviations
    stats = {
        "dra_mean": df[dra].mean(),
        "ddec_mean": df[ddec].mean(),
        "dra_std": df[dra].std(),
        "ddec_std": df[ddec].std()
    }

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    filename = f"offset_{survey}_{filter}_stats.json"
    # Write to JSON
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"Offset statistics written to {output_path}")



def visualise_offsets(df, survey, output_dir, exclude_ids, filter, use_filters=False):
    """
    Produces three types of plots for the astrometric offsets (Scatter, Quiver, and Histogram)
    and returns the filtered DataFrame for further analysis.

    Args:
        df (pandas DataFrame): The complete dataframe with all offsets stored.
        survey (str): The name of the survey (primer or cweb plus observation number)
        output_dir (str): Path to the output directory.
        exclude_ids (list[int]): A list of galaxy IDs to be excluded from analysis.

    Returns:
        pandas DataFrame: The filtered DataFrame for further analysis.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Exclude specific galaxy IDs
    df = df[~df['id'].isin(exclude_ids)].copy()

    # Find corresponding ddec column
    if use_filters == True:
        col1 = survey + f'_{filter.lower()}_dra'
        col2 = survey + f'_{filter.lower()}_ddec'
    else:
        col1 = survey + '_dra'
        col2 = survey + '_ddec'

    # Remove rows where col1 is exactly 0.0
    df_new = df[df[col1] != 0.0].copy()

    # Determine survey name
    if 'primer' in col1:
        survey_cap = 'PRIMER' 
    elif 'cosmos' in col1:
        survey_cap = 'COSMOS-3D'
    else: 
        survey_cap = 'COSMOS-Web'

    # ---- Compute Statistics ----
    write_offset_stats(df_new, col1, col2, output_dir, survey, filter)
    
    # ---- Scatter Plot ----
    plot_dir = os.path.join(output_dir, 'scatter_plots/')
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df_new[col1], df_new[col2], s=10, alpha=0.7)
    ax.set_xlabel('ΔRA (arcsec)')
    ax.set_ylabel('ΔDec (arcsec)')
    ax.set_title(f'{survey_cap} Astrometric Offset\n{filter} MIRI vs F444W NIRCam')

    scatter_path = os.path.join(plot_dir, f'{survey}_offset_{filter}_scatter.png')
    fig.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---- Quiver Plot ----
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.quiver(df_new['ra'], df_new['dec'], df_new[col1], df_new[col2], angles='xy', scale_units='xy', scale=1)
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.set_title(f'{survey_cap} Astrometric Offset\n{filter} MIRI vs F444W NIRCam')

    # Calculate and adjust axis limits to fit all arrows
    ra_min, ra_max = df_new['ra'].min(), df_new['ra'].max()
    dec_min, dec_max = df_new['dec'].min(), df_new['dec'].max()
    arrow_max = np.sqrt(df_new[col1]**2 + df_new[col2]**2).max()

    ax.set_xlim(ra_min - arrow_max, ra_max + arrow_max)
    ax.set_ylim(dec_min - arrow_max, dec_max + arrow_max)

    quiver_path = os.path.join(plot_dir, f'{survey}_offset_{filter}_arrows.png')
    fig.savefig(quiver_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---- Histogram Plot ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    axs[0].hist(df_new[col1], bins=15, edgecolor='black', alpha=0.7)
    axs[0].set_title("ΔRA (arcsec)")
    axs[0].set_xlabel("Offset (arcsec)")

    axs[1].hist(df_new[col2], bins=15, edgecolor='black', alpha=0.7)
    axs[1].set_title("ΔDec (arcsec)")
    axs[1].set_xlabel("Offset (arcsec)")

    hist_path = os.path.join(plot_dir, f'{survey}_offset_{filter}_hist.png')
    fig.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Rename columns for consistency
    df_new = df_new.rename(columns={col1: 'dra', col2: 'ddec'})
    
    return df_new


def shift_miri_fits(fits_file, dra_mean, ddec_mean):
    """
    Shifts the WCS of a MIRI fits file to compensate for a systematic astrometric offset.

    Args:
        fits_file (str): Path to the input MIRI fits file.
        dra_mean (float): Mean RA offset in arcseconds.
        ddec_mean (float): Mean Dec offset in arcseconds.
        output_dir (str): Directory in which the corrected fits file is saved.

    Returns:
        None
    """
    
    with fits.open(fits_file, mode='update') as hdul:
        for hdu in hdul:
            try:
                wcs = WCS(hdu.header)
                if wcs.has_celestial:
                    # Apply shift in degrees
                    hdu.header['CRVAL1'] -= dra_mean / 3600.0
                    hdu.header['CRVAL2'] -= ddec_mean / 3600.0
                    print(f"✅ Updated WCS in: {fits_file}")
                    break  # Only shift first valid celestial WCS
            except Exception:
                continue
        else:
            raise RuntimeError("No celestial WCS found to update.")



# Define a function to read the json files
def get_mean_stats(filename):
    with open(filename, "r") as f:
        stats = json.load(f)
    dra_mean = stats["dra_mean"]
    ddec_mean = stats["ddec_mean"]
    return dra_mean, ddec_mean


# Function to plot offsets in polar coordinates
def plot_astrometric_offsets(df1, df2, label1, label2, output_dir, band):
    # Convert RA/Dec offsets to polar coordinates
    def to_polar(dra, ddec):
        r = np.sqrt(dra**2 + ddec**2)
        theta = np.arctan2(ddec, dra)  # angle from x-axis (RA), in radians
        return r, theta

    r1, theta1 = to_polar(df1['dra'], df1['ddec'])
    r2, theta2 = to_polar(df2['dra'], df2['ddec'])

    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.scatter(theta1, r1, s=10, alpha=0.6, label=label1)
    ax.scatter(theta2, r2, s=10, alpha=0.6, label=label2)

    # Plot mean offset vectors
    for r, t, label, col in zip(
        [r1, r2],
        [theta1, theta2],
        [label1, label2],
        ['tab:blue', 'tab:orange']
    ):
        r_mean = np.mean(r)
        t_mean = np.arctan2(np.mean(np.sin(t)), np.mean(np.cos(t)))
        ax.scatter(t_mean, r_mean, color=col, label=f'{label} mean')

    ax.set_title(f'Astrometric Offsets (F444W → {band})', fontsize=14)
    ax.set_rmax(0.5)
    ax.set_rticks([0.1, 0.2, 0.3, 0.4, 0.5])  # arcsec
    ax.set_rlabel_position(135)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Save
    figname = output_dir + f'polar_offset_{band}.png'
    print(f'Figure saved to {figname}')
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()