"""
MIRI Utils Photometry Pipeline
==============================

Functions for performing aperture photometry on MIRI (Mid-Infrared Instrument) images.

This module provides tools for:
- Background estimation and subtraction using 2D plane fitting
- Aperture photometry with elliptical apertures
- PSF-based aperture corrections
- Flux and uncertainty calculations
- Visualisation of photometry results

The primary workflow involves:
1. Loading FITS image data and error maps
2. Estimating background using sigma-clipped plane fitting
3. Measuring source flux within defined apertures
4. Applying aperture corrections based on PSF models
5. Converting measurements to physical units (Jy, AB magnitudes)

Example usage
-------------
    from miri_utils.photometry import perform_photometry
    
    perform_photometry(
        cutout_files=['data/12345_F770W.fits', 'data/12345_F1800W.fits'],
        aperture_table='data/apertures.csv',
        output_folder='results/'
    )

Author: Benjamin P. Collins
Date: May 15, 2025
Version: 1.0
"""

import os
import glob
import numpy as np
import pandas as pd
import warnings
import json
import matplotlib.pyplot as plt
import astropy.units as u

from PIL import Image
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from astropy.table import Table, MaskedColumn
from astropy.stats import SigmaClip
from photutils.aperture import EllipticalAperture, EllipticalAnnulus, aperture_photometry

from .cutout_tools import load_cutout

# Suppress common WCS-related warnings that don't affect functionality
warnings.simplefilter("ignore", category=FITSFixedWarning)


def adjust_aperture(galaxy_id, filter, survey, obs, output_folder, save_plot=False):
    
    # --- Load the FITS table ---
    #table_path =  '/home/bpc/University/master/Red_Cardinal/Flux_Aperture_PSFMatched_AperCorr_old.fits'
    #aperture_table = Table.read(table_path)
    table_path =  '/home/bpc/University/master/Red_Cardinal/aperture_table.csv'
    df = pd.read_csv(table_path)
    
    # --- Select the galaxy by ID ---
    #row = aperture_table[aperture_table['ID'] == galaxy_id][0]
    galaxy_id = int(galaxy_id)
    row = df[df['ID'] == galaxy_id].iloc[0]

    # --- Read in rotation angle of MIRI FITS file ---
    angle_file = '/home/bpc/University/master/Red_Cardinal/rotation_angles.json'
    with open(angle_file, "r") as f:
        angles = json.load(f)
    angle = angles[f"angle_{survey}{obs}"]
    
    # --- Read WCS from NIRCam image ---
    nircam_path = f"/home/bpc/University/master/Red_Cardinal/NIRCam/F444W_cutouts/{galaxy_id}_F444W_cutout.fits"
    nircam_data, nircam_wcs = load_cutout(nircam_path)

    # --- Convert NIRCam pixel coordinates to sky ---
    sky_coord = nircam_wcs.pixel_to_world(row['Apr_Xcenter'], row['Apr_Ycenter'])
    
    # --- Open MIRI cutout image ---
    miri_path = f"/home/bpc/University/master/Red_Cardinal/cutouts_phot/{galaxy_id}_{filter}_cutout_{survey}{obs}.fits"
    miri_data, miri_wcs = load_cutout(miri_path)

    # --- Convert sky coords to MIRI pixel coordinates ---
    miri_x, miri_y = miri_wcs.world_to_pixel(sky_coord)

    # --- Create elliptical region in MIRI pixel space ---
    nircam_scale = 0.03    # arcsec/pixel
    miri_scale = 0.11092  # arcsec per pixel
    
    # arcsec/pixel
    scale_factor = nircam_scale / miri_scale
    
    # --- Specify parameters for the ellips ---
    width = row['Apr_A'] * scale_factor
    height = row['Apr_B'] * scale_factor
    theta = -row['Apr_Theta']
    theta_new = ((theta - angle) % 180) * u.deg
    
    # --- Create region file and check if folder exists ---
    os.makedirs(output_folder, exist_ok=True)
    reg_file = os.path.join(output_folder, f'regions/{galaxy_id}_{survey}{obs}_aperture.reg') 
    
    # --- Write to DS9-compatible region file ---
    with open(reg_file, "w") as fh:
        fh.write("# Region file format: DS9 version 4.1\n")
        fh.write("global color=red dashlist=8 3 width=2 font=\"helvetica 10 normal\" "
                "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        fh.write("image\n")
        fh.write(f"ellipse({miri_x:.2f},{miri_y:.2f},{width:.2f},{height:.2f},{theta_new:.2f})\n")
    
    if save_plot:
        
        # --- Clean and prepare the MIRI data for plotting ---
        miri_clean = np.copy(miri_data)
        finite_vals = miri_clean[np.isfinite(miri_clean)].flatten()
        
        # Sort and get lowest 80% values
        sorted_vals = np.sort(finite_vals)
        cutoff_index = int(0.8 * len(sorted_vals))
        background_vals = sorted_vals[:cutoff_index]
        background_mean = np.mean(background_vals)

        # Replace NaNs or infs with background mean
        miri_clean[~np.isfinite(miri_clean)] = background_mean

        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(miri_clean, origin='lower', cmap='gray', vmin=np.percentile(miri_clean, 5), vmax=np.percentile(miri_clean, 99))

        # Original ellipse (without rotation correction)
        ellipse_original = Ellipse(
            xy=(miri_x, miri_y),
            width=width,
            height=height,
            angle=theta,  # Just the original θ!
            edgecolor='red',
            facecolor='none',
            lw=2,
            label='Original Ellipse'
        )
        ax.add_patch(ellipse_original)

        ellipse = Ellipse(
            xy=(miri_x, miri_y),
            width=width,
            height=height,
            angle=theta_new.to_value(u.deg),
            edgecolor='blue',
            linestyle='--',  # maybe dashed to differentiate
            facecolor='none',
            lw=2,
            label='Rotated Ellipse'
        )
        ax.add_patch(ellipse)
        
        ax.set_title(f"Galaxy {galaxy_id} - {filter} ({survey}{obs})")
        ax.set_xlim(miri_x - 30, miri_x + 30)
        ax.set_ylim(miri_y - 30, miri_y + 30)
        ax.legend(loc='upper right')
        
        # Save figure
        png_path = os.path.join(output_folder, f'masks/{galaxy_id}_{survey}{obs}_aperture_overlay.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    # Collect modified aperture data
    aperture_info = {          
    'Apr_A': width,            # Rescaled aperture
    'Apr_B': height,
    'Apr_Xcenter': miri_x,
    'Apr_Ycenter': miri_y,
    'Apr_Theta': theta_new.to_value(u.deg),
    'ID': galaxy_id
    }
    
    return aperture_info




def estimate_background(galaxy_id, filter_name, image_data, aperture_params, sigma=3.0, 
                        annulus_factor=3.0, fig_path=None):
    """
    Estimate background using a global 2D plane fit, then extract statistics from 
    an elliptical annulus.
    
    Parameters
    ----------
    galaxy_id : str
        The ID of the galaxy
    filter_name : str
        The band which is being observed
    image_data : ndarray
        The 2D image data
    aperture_params : dict
        Dictionary containing aperture parameters (x_center, y_center, a, b, theta)
    sigma : float
        Sigma clipping threshold
    annulus_factor : float
        Factor by which to scale the inner ellipse to create the outer ellipse
    visualise : bool, optional
        If True, display visualisation plots
        
    Returns
    -------
    - background_plane : ndarray
        2D background model
    - background_median : float
        median background value within the annulus
    - background_std : float
        standard deviation of background model within the annulus (excluding clipped data)
    - background_region_mask : ndarray
        boolean mask showing the region used for background stats

    """
    
    x_center = aperture_params['x_center']
    y_center = aperture_params['y_center']
    a = aperture_params['a']
    b = aperture_params['b']
    theta = aperture_params['theta']    # in radians
    
    # Create source aperture
    source_aperture = EllipticalAperture(
        positions=(x_center, y_center),
        a=a,
        b=b,
        theta=theta
    )
    
    # Create mask for the source
    source_mask = source_aperture.to_mask(method='center').to_image(image_data.shape)
    source_mask_bool = source_mask.astype(bool)
    
    # Apply initial source mask to the data
    masked_data = np.copy(image_data)
    masked_data[source_mask_bool] = np.nan
    
    # Apply sigma clipping to the remaining background
    sigma_clip = SigmaClip(sigma=sigma)
    clipped_data = sigma_clip(masked_data)
    
    # Create a mask for sigma-clipped pixels (clipped_data.mask is True for clipped values)
    sigma_clipped_mask = clipped_data.mask if hasattr(clipped_data, 'mask') else np.isnan(clipped_data)
    
    # Create a global mask for all valid background pixels (not in source, not sigma-clipped)
    global_mask = ~source_mask_bool & ~np.isnan(image_data) & ~sigma_clipped_mask
    
    # Define indices for the full image
    y, x = np.indices(image_data.shape)
    
    # Extract coordinates and values of all valid background pixels for global fitting
    x_vals = x[global_mask].flatten()
    y_vals = y[global_mask].flatten()
    z_vals = image_data[global_mask].flatten()
    
    # Check if we have enough pixels for fitting
    if len(z_vals) < 3:
        raise ValueError("Not enough background pixels for fitting. Try adjusting parameters.")
    
    # Fit a 2D plane (ax + by + c) to all valid background pixels
    A = np.vstack([x_vals, y_vals, np.ones_like(x_vals)]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z_vals, rcond=None)
    alpha, beta, gamma = coeffs
    
    # Create the 2D background plane for the entire image
    background_plane = alpha * x + beta * y + gamma
    
    # Define the background region
    region_name = "Annulus"
    
    # Set minimum and maximum sizes for the annulus
    min_pixels = 300  # Minimum number of pixels in the annulus for reliable background estimation
    max_annulus_size = 35  # Roughly half of the 75x75 image size for large apertures
    step_factor = 0.15  # Step size for expanding the annulus
    max_attempts = 10
    attempt = 0
    
    # Start with a slightly larger annulus than the source aperture
    a_in = a * 2
    b_in = b * 2
    a_out = a_in * annulus_factor
    b_out = b_in * annulus_factor
    
    if galaxy_id in ['12020', '17669', '7136']:
        a_in *= 1.2
        b_in *= 1.2
        a_out *= 1.3
        b_out *= 1.3
    if galaxy_id in ['9871', '11136', '11137', '11494', '12340', '12717', '17793', '16874', '17517', '20397']:
        a_out *= 0.7
        b_out *= 0.7
    
    # Adjust the outer size until the annulus contains enough pixels
    while attempt < max_attempts:
        # Create the annulus
        annulus = EllipticalAnnulus(
            positions=(x_center, y_center),
            a_in=a_in,
            b_in=b_in,
            a_out=min(a_out, max_annulus_size),
            b_out=min(b_out, max_annulus_size),
            theta=theta
        )
        
        # Create mask for the annulus region
        annulus_mask = annulus.to_mask(method='center').to_image(image_data.shape)
        background_region_mask = annulus_mask.astype(bool) & ~np.isnan(image_data)
        pixel_count = np.sum(background_region_mask)

        # If the annulus has enough pixels, break the loop
        if pixel_count >= min_pixels:
            break
        
        # Expand the annulus for the next attempt
        a_out += a_in * step_factor
        b_out += b_in * step_factor
        attempt += 1
        
    
    # Gather all the pixels within the background region that were NOT sigma-clipped
    bkg_region_valid_pixels = ~sigma_clipped_mask & background_region_mask
    residual_data = image_data - background_plane
    num_valid_pixels = np.sum(bkg_region_valid_pixels)
    
    # Calculate background statistics
    background_residuals = residual_data[bkg_region_valid_pixels]
    background_std = np.std(background_residuals) * np.sqrt(num_valid_pixels)
    
    # Extract background plane values for the background region and calculate the median
    bkg_plane_values = background_plane[background_region_mask]
    background_median = np.median(bkg_plane_values)
    
    # Print background statistics
    print(f"Background Statistics:")
    print(f"  Global 2D Plane coefficients: a={alpha:.6e}, b={beta:.6e}, c={gamma:.6f}")
    print(f"  {region_name} region background median: {background_median:.6f}")
    print(f"  {region_name} region background std dev: {background_std:.6f}")
    
    # Create a mask visualisation
    mask_vis = np.zeros_like(image_data, dtype=int)
    mask_vis[~sigma_clipped_mask] = 1  # Pixels excluded by sigma clipping
    mask_vis[background_region_mask] = 2  # Pixels in the annulus/rectangle
    mask_vis[source_mask_bool] = 3  # Source pixels
    
    # Store visualization data
    vis_data = {
        'galaxy_id': galaxy_id,
        'filter': filter_name,
        'original_data': image_data,
        'background_plane': background_plane,
        'background_subtracted': residual_data,
        'mask_vis': mask_vis,
        'sigma_clipped_mask': sigma_clipped_mask,
        'background_region_mask': background_region_mask,
        'source_mask': source_mask_bool,
        'aperture_params': aperture_params,
        'a_in': a_in,
        'b_in': b_in,
        'a_out': a_out,
        'b_out': b_out,
        'sigma': sigma,
        'region_name': region_name,
        'coeffs': (alpha, beta, gamma)
    }
    
    # If requested, visualize the results
    if fig_path:
        visualise_background(vis_data, fig_path=fig_path)
    
    return background_median, background_std



def visualise_background(vis_data, fig_path=None):
    """
    Create visualisations from the background estimation data.
    
    Parameters
    ----------
    vis_data : dict
        Dictionary containing all data needed for visualisation
    fig_path : str, optional
        Path to save the visualisation figure
    """
    # Extract data from the dictionary
    image_data = vis_data['original_data']
    background_plane = vis_data['background_plane']
    background_subtracted = vis_data['background_subtracted']
    mask_vis = vis_data['mask_vis']
    aperture_params = vis_data['aperture_params']
    sigma = vis_data['sigma']
    region_name = vis_data['region_name']
    galaxy_id = vis_data['galaxy_id']
    filter = vis_data['filter']
    
    # Create aperture objects for plotting
    x_center = aperture_params['x_center']
    y_center = aperture_params['y_center']
    a = aperture_params['a']
    b = aperture_params['b']
    theta = aperture_params['theta']
    
    source_aperture = EllipticalAperture(
        positions=(x_center, y_center),
        a=a,
        b=b,
        theta=theta
    )
    
    # Create visualisations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Original data with aperture
    vmin = np.nanpercentile(image_data, 5)
    vmax = np.nanpercentile(image_data, 95)
    
    im0 = axes[0, 0].imshow(image_data, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    plt.colorbar(im0, ax=axes[0, 0], label='Flux [MJy/(sr pixel)]')
    
    # Plot the source aperture
    source_aperture.plot(ax=axes[0, 0], color='red', lw=1.5)
    
    # Plot the background region
    annulus = EllipticalAnnulus(
        positions=(x_center, y_center),
        a_in  = vis_data['a_in'],
        b_in  = vis_data['b_in'],
        a_out = vis_data['a_out'],
        b_out = vis_data['b_out'],
        theta = theta
    )
    annulus.plot(ax=axes[0, 0], color='white', lw=1.5)
        
    axes[0, 0].set_title("Original Data with Aperture and Annulus")
    
    # Background-subtracted data
    vmin2 = np.nanpercentile(background_subtracted, 5)
    vmax2 = np.nanpercentile(background_subtracted, 95)
    
    im1 = axes[0, 1].imshow(background_subtracted, origin='lower', cmap='magma', vmin=vmin2, vmax=vmax2)
    plt.colorbar(im1, ax=axes[0, 1], label='Background-subtracted Flux [MJy/(sr pixel)]')
    source_aperture.plot(ax=axes[0, 1], color='red', lw=1.5)
    axes[0, 1].set_title("Background-subtracted Data with Aperture")
    
    # Global 2D background plane
    im2 = axes[1, 0].imshow(background_plane, origin='lower', cmap='viridis')
    plt.colorbar(im2, ax=axes[1, 0], label='Background Flux [MJy/(sr pixel)')
    axes[1, 0].set_title("Global 2D Background Plane")
        
    # Mask visualisation
    cmap = plt.cm.get_cmap('viridis', 4)
    im3 = axes[1, 1].imshow(mask_vis, origin='lower', cmap=cmap, vmin=-0.5, vmax=3.5)
    cbar = plt.colorbar(im3, ax=axes[1, 1], ticks=[0, 1, 2, 3])
    cbar.set_ticklabels([f'Excluded\n(σ={sigma})', 'Used for fitting', 
                         f'{region_name} region', 'Source'])
    axes[1, 1].set_title("Pixel Masks")
    
    fig.suptitle(f'{filter}', fontsize=18)#, fontweight='bold')
    plt.tight_layout()
    
    if fig_path:
        os.makedirs(fig_path, exist_ok=True)
        if filter == 'F1800W':
            filepath = os.path.join(fig_path, f'{galaxy_id}_{filter}.png')
        else: 
            filepath = os.path.join(fig_path, f'{galaxy_id}.png')
        plt.savefig(filepath, dpi=150)
        plt.close(fig)

def get_psf(filter_name, psf_dir='/home/bpc/University/master/Red_Cardinal/WebbPSF/'):
    """
    Read MIRI PSF file for the specified filter.
    
    Parameters
    ----------
    filter_name : str
        Name of the filter
    psf_dir : str
        Directory containing PSF files
        
    Returns
    -------
    psf_data : ndarray
        PSF data
    """
    psf_file = os.path.join(psf_dir, f'PSF_MIRI_{filter_name}.fits')    
    with fits.open(psf_file) as psf:
        return psf[0].data

def get_aperture_params(galaxy_id, aperture_table):
    """
    Retrieve aperture parameters from the CSV table.
    
    Parameters
    ----------
    galaxy_id : str
        ID of the galaxy
    aperture_table : str
        Path to CSV table with aperture parameters
        
    Returns
    -------
    dict
        Dictionary with aperture parameters
    """
    df = pd.read_csv(aperture_table)
    row = df[df['ID'] == int(galaxy_id)].iloc[0]
    
    return {
        'x_center': row['Apr_Xcenter'],
        'y_center': row['Apr_Ycenter'], 
        'a': row['Apr_A'] / 2,  # Converting diameter to radius
        'b': row['Apr_B'] / 2,  # Converting diameter to radius
        'theta': (row['Apr_Theta'] * u.deg).to_value(u.rad)  # Convert to radians
    }

def calculate_aperture_correction(psf_data, aperture_params):
    """
    Calculate aperture correction factor for given PSF and aperture.
    
    Parameters
    ----------
    psf_data : ndarray
        PSF data
    aperture_params : dict
        Aperture parameters
        
    Returns
    -------
    correction_factor : float
        Aperture correction factor
    """
    aperture = EllipticalAperture(
        positions=(psf_data.shape[1] / 2, psf_data.shape[0] / 2),
        a=aperture_params['a'],
        b=aperture_params['b'],
        theta=aperture_params['theta']
    )
    total_flux = np.sum(psf_data)
    phot_table = aperture_photometry(psf_data, aperture)
    flux_in_aperture = phot_table['aperture_sum'][0]
    return total_flux / flux_in_aperture

def measure_flux(image_data, error_map, background_median, background_std, aperture_params):
    """
    Calculate flux and uncertainty from aperture photometry.
    
    Parameters
    ----------
    image_data : ndarray
        Image data
    background_median : float
        Median background level
    error_map : ndarray
        Error map data
    background_std : float
        Standard deviation of background
    aperture_params : dict
        Aperture parameters
        
    Returns
    -------
    dict
        Dictionary with flux measurements and uncertainties
    """
    # Subtract the median background value within the annulus from the data
    data_bkgsub = image_data - background_median
    
    # Create aperture
    aperture = EllipticalAperture(
        positions=(aperture_params['x_center'], aperture_params['y_center']),
        a=aperture_params['a'],
        b=aperture_params['b'],
        theta=aperture_params['theta']
    )
    
    # Sum flux within the aperture - baclground subtracted data
    phot_table = aperture_photometry(data_bkgsub, aperture, method='exact')
    flux = phot_table['aperture_sum'][0]    # in MJy/sr
    
    # Mask for aperture
    aperture_mask = aperture.to_mask(method='exact')
    mask = aperture_mask.to_image(data_bkgsub.shape)
    
    # Flux uncertainty from ERR extension
    error_map = np.nan_to_num(error_map, nan=0.0, posinf=0.0, neginf=0.0)
    image_errors = error_map * mask
    sum_image_errors = np.sqrt(np.sum(image_errors**2))
    
    # Number of pixels within the aperture
    n_pix = aperture.area
    
    # To obtain the background flux within the aperture we multiply the median background within the annulus
    # by the number of pixels within the aperture    
    background_flux = n_pix * background_median
    
    # Total flux uncertainty
    total_flux_error = np.sqrt(sum_image_errors**2 + background_std**2)
    
    # Median error of the error map within the aperture
    median_error = np.median(error_map[mask>0])    
    
    # Convert everything to from MJy/sr to Jy
    miri_scale = 0.11092  # arcsec per pixel
    miri_scale_rad = miri_scale / 206265
    omega_pix = miri_scale_rad**2
    conversion_factor = 1e6 * omega_pix
    
    # Now everything is in Jy!!
    return {
        'flux': flux * conversion_factor,
        'flux_error': total_flux_error * conversion_factor,
        'background_flux': background_flux * conversion_factor,
        'median_error': median_error * conversion_factor,
        'pixel_count': n_pix
    }

# --- Main Loop ---

def perform_photometry(cutout_files, aperture_table, output_folder, psf_data, create_plots=False):
    """
    Main function to perform photometry on a list of cutout files.
    
    Parameters
    ----------
    cutout_files : list
        List of paths to cutout FITS files
    aperture_table : str
        Path to CSV table with aperture parameters
    output_folder : str
        Path to output folder
    psf_dir : str
        Directory containing PSF files
    create_plots : bool, optional
        Decide whether plots should be made
    """
    results = []
    
    for fits_path in cutout_files:
        # Extract ID and filter from filename
        fits_name = os.path.basename(fits_path)
        galaxy_id = fits_name.split('_')[0]
        filter_name = fits_name.split('_')[1]
        
        print(f'Processing galaxy {galaxy_id} with filter {filter_name}...')
        
        # Load image data
        with fits.open(fits_path) as hdul:
            image_data = hdul['SCI'].data if 'SCI' in hdul else hdul[1].data
            image_error = hdul['ERR'].data if 'ERR' in hdul else hdul[2].data
            
        # Get aperture parameters
        aperture_params = get_aperture_params(galaxy_id, aperture_table)
        
        # Set sigma-clipping threshold based on galaxy ID and filter
        sigma = 2.8  # Default value
        
        # Special cases for certain galaxies
        if galaxy_id in ['12332', '12282', '10314', '12164', '18332', '21452', '21477', 
                        '21541', '22606', '10592', '11136', '11142', '11420', '11451', 
                        '11494', '11716', '13103', '16419', '19042']:
            sigma = 2.0
            
        # Additional adjustments for F770W filter
        if filter_name == 'F770W' and galaxy_id in ['7136', '7904', '7922', '8469', '11716', 
                                                   '16424', '17000', '17669', '11137']:
            sigma = 2.0
        
        # Setup paths for visualisation
        #vis_path = os.path.join(output_folder, 'mosaic_fits')
        #os.makedirs(vis_path, exist_ok=True)
        
        if create_plots:
            fig_path = os.path.join(output_folder, 'mosaic_plots')
            os.makedirs(fig_path, exist_ok=True)
        else:
            fig_path = None
        
        # Estimate background with 2D-plane fit
        background_median, background_std = estimate_background(
            galaxy_id, 
            filter_name, 
            image_data, 
            aperture_params,
            sigma=sigma, 
            annulus_factor=3.0, 
            fig_path=fig_path
        )                                                                   
        
        # Measure flux
        flux_measurements = measure_flux(
            image_data, 
            image_error,
            background_median,  
            background_std, 
            aperture_params
        )

        # Get PSF and calculate aperture correction
        correction_factor = calculate_aperture_correction(psf_data, aperture_params)

        # Apply aperture correction
        corrected_flux = flux_measurements['flux'] #* correction_factor
        corrected_flux_error = flux_measurements['flux_error'] #* correction_factor
        corrected_background_flux = flux_measurements['background_flux'] #* correction_factor
        corrected_background_error = background_std #* correction_factor
        
        # --- Convert fluxes into AB magnitudes ---
        if corrected_flux > 0:
            ab_mag = -2.5 * np.log10(corrected_flux) + 8.90
        else: ab_mag = np.nan
        
        # Append results
        results.append({
            'ID': int(galaxy_id),
            'Flux': corrected_flux,
            'Flux_Err': corrected_flux_error,
            'Image_Err': flux_measurements['median_error'] * correction_factor,
            'Flux_BKG': corrected_background_flux,
            'Flux_BKG_Err': corrected_background_error,
            'AB_Mag': ab_mag,
            'N_PIX': flux_measurements['pixel_count'],
            'Apr_A': aperture_params['a'] * 2,  # Convert back to diameter for output
            'Apr_B': aperture_params['b'] * 2,  # Convert back to diameter for output
            'Apr_Xcenter': aperture_params['x_center'],
            'Apr_Ycenter': aperture_params['y_center'],
            'Apr_Theta': (aperture_params['theta'] * u.rad).to_value(u.deg)  # Convert to degrees for output
        })
    
    # Save to output table (assuming it's a pandas DataFrame)
    os.makedirs(os.path.join(output_folder, 'results'), exist_ok=True)
    output_path = os.path.join(output_folder, f'results/photometry_table_{filter_name}.csv')
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    
def combine_figures(fig_path='/home/bpc/University/master/Red_Cardinal/photometry/mosaic_plots/'):
    """Function that scans a directory for plots in different filters and
       combines them if available.   
    """
    print(f"Scanning {fig_path} for galaxy images to combine...")

    # Get all F1800W images
    f1800w_pngs = glob.glob(os.path.join(fig_path, '*_F1800W.png'))

    # Track how many images we've combined
    combined_count = 0

    for f1800w_png in f1800w_pngs:
        # Extract galaxy ID from filename
        galaxy_id = os.path.basename(f1800w_png).replace('_F1800W.png', '')
        f770w_png = os.path.join(fig_path, f'{galaxy_id}.png')
        
        # Check if the standard file exists
        if os.path.exists(f770w_png):
            try:
                # Open both images
                img_f770w = Image.open(f770w_png)
                img_f1800w = Image.open(f1800w_png)
                
                # Get dimensions
                width1, height1 = img_f770w.size
                width2, height2 = img_f1800w.size
                
                # Create a new image with enough width for both images
                combined_width = width1 + width2
                combined_height = max(height1, height2)
                combined_img = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
                
                # Paste both images
                combined_img.paste(img_f770w, (0, 0))
                combined_img.paste(img_f1800w, (width1, 0))
                
                # Save combined image
                save_png = os.path.join(fig_path, f'{galaxy_id}.png')
                combined_img.save(save_png)
                
                # Delete the F1800W file
                os.remove(f1800w_png)
                
                combined_count += 1
                print(f"Combined images for galaxy {galaxy_id}")
                
            except Exception as e:
                print(f"Error combining images for galaxy {galaxy_id}: {e}")

    print(f"Combined {combined_count} galaxy image pairs.")

    # Check if any F1800W images remain (no matching F770W image)
    remaining_f1800w = glob.glob(os.path.join(fig_path, '*_F1800W.png'))
    if remaining_f1800w:
        print(f"Note: {len(remaining_f1800w)} F1800W images have no matching standard image.")
 
 
 
 
def create_fits_table_from_csv(f770w_csv_path, f1800w_csv_path=None, output_file='Flux_Aperture_PSFMatched_AperCorr_MIRI.fits'):
    """
    Create a FITS table by combining photometry data from F770W and F1800W CSV files.

    Parameters:
    -----------
    f770w_csv_path : str
        Path to the CSV file containing F770W photometry results.
    f1800w_csv_path : str, optional
        Path to the CSV file containing F1800W photometry results.
        If None or file doesn't exist, only F770W data will be used.
    output_file : str
        Output FITS file name.
    """
    # Load F770W CSV file (required)
    try:
        df_f770w = pd.read_csv(f770w_csv_path)
        print(f"Loaded F770W data with {len(df_f770w)} rows")
    except FileNotFoundError:
        raise FileNotFoundError(f"F770W CSV file not found: {f770w_csv_path}")

    # Load F1800W CSV file (optional)
    df_f1800w = None
    if f1800w_csv_path and os.path.exists(f1800w_csv_path):
        df_f1800w = pd.read_csv(f1800w_csv_path)
        print(f"Loaded F1800W data with {len(df_f1800w)} rows")
    else:
        print("No F1800W data provided or file not found")

    # Get all galaxy IDs from F770W - these are our primary sources
    galaxy_ids = df_f770w['ID'].unique()
    print(f"Found {len(galaxy_ids)} unique galaxy IDs")

    # Initialize the table
    table = Table()

    # Define which columns should be masked
    masked_columns = ['Flux_Err', 'Flux_BKG', 'AB_Mag', 'Flux_BKG_Err']

    # These columns will have 2 values per row (one for each filter)
    array_columns = ['Flux', 'Flux_Err', 'Image_Err', 'Flux_BKG', 'Flux_BKG_Err', 
                        'AB_Mag', 'N_PIX']

    # These columns will be scalar (one value per galaxy)
    scalar_columns = ['Apr_A', 'Apr_B', 'Apr_Xcenter', 'Apr_Ycenter', 'Apr_Theta']

    # Prepare data for each column
    column_data = {col: [] for col in array_columns + scalar_columns}

    # Process each galaxy ID
    for gid in galaxy_ids:
        # Get F770W data for this galaxy
        f770w_row = df_f770w[df_f770w['ID'] == gid].iloc[0]
        
        # Get F1800W data for this galaxy if available
        f1800w_row = None
        if df_f1800w is not None and gid in df_f1800w['ID'].values:
            f1800w_row = df_f1800w[df_f1800w['ID'] == gid].iloc[0]
        
        # Process array columns (with values for both filters)
        for col in array_columns:
            # Always have F770W data
            filter_values = [f770w_row[col]]
            
            # Add F1800W data if available, otherwise add NaN
            if f1800w_row is not None:
                filter_values.append(f1800w_row[col])
            else:
                filter_values.append(np.nan)
            
            column_data[col].append(filter_values)
        
        # Process scalar columns (single value per galaxy)
        for col in scalar_columns:
            column_data[col].append(f770w_row[col])

    # Add ID column (one ID per galaxy)
    id_bytes = [str(gid).encode('ascii') for gid in galaxy_ids]
    table.add_column(id_bytes, name='ID')

    # Add array columns
    for col in array_columns:
        if col in masked_columns:
            # Create masked column
            masked_col = MaskedColumn(column_data[col], name=col, dtype=np.float64)
            
            # Mask NaN values
            for i, row in enumerate(column_data[col]):
                for j, val in enumerate(row):
                    if np.isnan(val):
                        masked_col.mask[i, j] = True
            
            # Additional masking for AB_Mag based on flux values
            if col == 'AB_Mag':
                for i, flux_row in enumerate(column_data['Flux']):
                    for j, flux_val in enumerate(flux_row):
                        # Mask AB_Mag where flux is invalid
                        if flux_val <= 0 or np.isnan(flux_val):
                            masked_col.mask[i, j] = True
            
            table.add_column(masked_col)
        else:
            # Regular column
            table.add_column(column_data[col], name=col)

    # Add scalar columns
    for col in scalar_columns:
        table.add_column(column_data[col], name=col)

    # Print table info
    print("\nFITS Table Summary:")
    print(table.info())

    # Write to FITS file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    table.write(output_file, format='fits', overwrite=True)
    print(f"\nSaved FITS table to {output_file}")

    return table


def combine_filter_csv_to_fits(results_folder):
    """
    Combine filter-specific CSV files into a single FITS table.

    Parameters:
    -----------
    output_folder : str
        Base folder containing the results folder with CSV files
    """
    # CSV files for each filter
    f770w_csv = os.path.join(results_folder, 'photometry_table_F770W.csv')
    f1800w_csv = os.path.join(results_folder, 'photometry_table_F1800W.csv')

    # Output FITS file
    fits_output = os.path.join(results_folder, 'Flux_Aperture_PSFMatched_AperCorr_MIRI.fits')

    # Create the combined FITS table
    table = create_fits_table_from_csv(f770w_csv, f1800w_csv, fits_output)

    return table