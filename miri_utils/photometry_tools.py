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
Version: 2.0
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
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.aperture import EllipticalAperture, EllipticalAnnulus, aperture_photometry
from photutils.segmentation import detect_sources, SegmentationImage
from photutils.centroids import centroid_com


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
    width = row['Apr_A'] * scale_factor * 2
    height = row['Apr_B'] * scale_factor * 2
    theta = -row['Apr_Theta']
    theta_new = ((theta - angle) % 180) * u.deg
    
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
        #png_path = os.path.join(output_folder, f'masks/{galaxy_id}_{survey}{obs}_aperture_overlay.png')
        #plt.savefig(png_path, dpi=150, bbox_inches='tight')
        #plt.close(fig)
        
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




def estimate_background(galaxy_id, filter_name, image_data, aperture_params, sigma_val, 
                        annulus_factor, fig_path=None):
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
    
    # Step 1: Get initial background estimate using sigma clipping
    # This doesn't require knowing sources beforehand
    mean_bg, median_bg, std_bg = sigma_clipped_stats(
        image_data, sigma=2.5, maxiters=5, mask=source_mask_bool | np.isnan(image_data)
    )

    # Step 2: Create segmentation map based on initial background estimate
    # Typically threshold is set at median + (N * std) above background
    threshold = median_bg + (sigma_val * std_bg)  # sigma threshold, adjust as needed
    masked_for_segm = np.copy(image_data)
    masked_for_segm[source_mask_bool] = median_bg
    segm = detect_sources(masked_for_segm, threshold, npixels=5)

    # Convert segmentation to a boolean mask (True where sources are detected)
    if segm is not None:
        segm_mask = segm.data > 0
    else:
        segm_mask = np.zeros_like(image_data, dtype=bool)

    # Step 3: Combine the known source mask with the segmentation mask
    combined_mask = source_mask_bool | segm_mask | np.isnan(image_data)

    # Create a global mask for all valid background pixels (not in any source)
    global_mask = ~combined_mask
    
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
    max_annulus_size = 35  # Roughly half of the 75x75 image size for large apertures
    base_factor = 2.2
  
    ecc = a/b
    
    a_in = a + 8
    b_in = b + 8/ecc
    
    scale_factor = max_annulus_size / a_in
    
    a_out = a_in * scale_factor
    b_out = b_in * scale_factor + 2*ecc   # account for eccentricity
    
    
    # Create the annulus
    annulus = EllipticalAnnulus(
        positions=(x_center, y_center),
        a_in=a_in,
        b_in=b_in,
        a_out=a_out,
        b_out=b_out,
        theta=theta
    )
    
    # Create mask for the annulus region
    annulus_mask = annulus.to_mask(method='center').to_image(image_data.shape)
    background_region_mask = annulus_mask.astype(bool) & ~np.isnan(image_data)

    # Gather all the pixels within the background region that were NOT sigma-clipped
    bkg_region_valid_pixels = ~segm_mask & background_region_mask
    residual_data = image_data - background_plane
    
    # Calculate background statistics
    background_residuals = residual_data[bkg_region_valid_pixels]
    sigma_bkg = np.std(background_residuals)
    
    # The following should be the 1-sigma uncertainty on the total background flux in the aperture
    background_std = sigma_bkg * np.sqrt(source_aperture.area)
    
    # Previous approach -------------------------------------
    #num_valid_pixels = np.sum(bkg_region_valid_pixels)
    #background_std = sigma_bkg * np.sqrt(num_valid_pixels)
    # -------------------------------------------------------
    
    # Extract background plane values for the background region and calculate the median
    bkg_plane_values = background_plane[background_region_mask]
    background_median = np.median(bkg_plane_values)
    
    # Create a mask visualisation
    mask_vis = np.zeros_like(image_data, dtype=int)
    mask_vis[~combined_mask] = 1  # Excluded pixels
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
        'segmentation_mask': segm_mask,
        'background_region_mask': background_region_mask,
        'source_mask': source_mask_bool,
        'aperture_params': aperture_params,
        'a_in': a_in,
        'b_in': b_in,
        'a_out': a_out,
        'b_out': b_out,
        'sigma': sigma_val,
        'region_name': region_name,
        'coeffs': (alpha, beta, gamma)
    }
    
    # If requested, visualize the results
    if fig_path:
        visualise_background(vis_data, fig_path=fig_path)
    
    return background_median, background_std, background_plane



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
    segm_mask = vis_data['segmentation_mask']
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
    # Overlay the segmentation mask as white outlines
    axes[0, 0].contour(segm_mask, levels=[0.5], colors='blue', linewidths=1.5)
        
    axes[0, 0].set_title("Original Data with Aperture and Masked Regions")
    
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
        return psf[3].data  # Recommended extension!

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
        'a': row['Apr_A'] / 2,  # Converting diameter to major axis length
        'b': row['Apr_B'] / 2,  # Converting diameter to minor axis length
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
    # Check for proper normalisation
    total_flux = np.sum(psf_data)
    if not np.isclose(total_flux, 1.0, atol=1e-3):
        print(f"Warning: PSF not normalised (sum = {total_flux:.6f}). Normalising now.")
        psf_data /= total_flux
        
    # Find centroid of the PSF
    x_cen, y_cen = centroid_com(psf_data)
    
    aperture = EllipticalAperture(
        positions=(x_cen, y_cen),
        a=aperture_params['a'],
        b=aperture_params['b'],
        theta=aperture_params['theta']
    )
    
    # Ensure background is subtracted
    psf_data -= np.median(psf_data[psf_data < np.percentile(psf_data, 10)])
    
    # Calculate flux in aperture using exact method
    phot_table = aperture_photometry(psf_data, aperture, method='exact')
    flux_in_aperture = phot_table['aperture_sum'][0]
    
    return total_flux / flux_in_aperture

def measure_flux(image_data, error_map, background_plane, background_std, aperture_params):
    """
    Calculate flux and uncertainty from aperture photometry.
    
    Parameters
    ----------
    image_data : ndarray
        Image data
    background_plane : ndarray
        2D-plane fit of the local background 
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
    
    # Create aperture
    aperture = EllipticalAperture(
        positions=(aperture_params['x_center'], aperture_params['y_center']),
        a=aperture_params['a'],
        b=aperture_params['b'],
        theta=aperture_params['theta']
    )
    
    # Subtract the median background value within the annulus from the data
    data_bkgsub = image_data - background_plane
    
    # Sum flux within the aperture - baclground subtracted data
    phot_table = aperture_photometry(data_bkgsub, aperture, method='exact')
    flux = phot_table['aperture_sum'][0]    # in MJy/sr
    
    # Mask for aperture
    aperture_mask = aperture.to_mask(method='exact')
    mask = aperture_mask.to_image(data_bkgsub.shape)
    
    # Flux uncertainty from ERR extension
    error_map = np.nan_to_num(error_map, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Square the errors, sum them, then take square root
    aperture_pixels = mask > 0
    sum_image_errors = np.sqrt(np.sum(error_map[aperture_pixels]**2))
    
    # To obtain the background flux within the aperture we multiply the median background within the annulus
    # by the number of pixels within the aperture    
    background_flux = np.sum(background_plane[aperture_pixels])    
    
    # Total flux uncertainty
    total_flux_error = np.sqrt(sum_image_errors**2 + background_std**2)
    
    # Median error of the error map within the aperture
    median_error = np.median(error_map[aperture_pixels])    
    
    # Convert everything to from MJy/sr to Jy
    miri_pixel_scale = 0.11092  # arcsec per pixel
    miri_scale_rad = miri_pixel_scale / 206265    # convert to rad per pixel
    pixel_area_sr = miri_scale_rad**2   # convert to sr per pixel
    
    # Convert MJy/sr to Jy
    conversion_factor = 1e6 * pixel_area_sr
    
    # Now everything is in Jy!!
    return {
        'flux': flux * conversion_factor,
        'flux_error': total_flux_error * conversion_factor,
        'background_flux': background_flux * conversion_factor,
        'median_error': median_error * conversion_factor,
        'pixel_count': aperture.area
    }

# --- Main Loop ---

def perform_photometry(cutout_files, aperture_table, output_folder, psf_data, suffix='', fig_path=None, sigma=3.0, annulus_factor=3.0, apply_aper_corr=True):
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
    apply_aper_corr : bool, optional
        Decide whether aperture correction should be applied
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
        
        if fig_path is not None:
            os.makedirs(fig_path, exist_ok=True)
        
        # Estimate background with 2D-plane fit
        background_median, background_std, background_plane = estimate_background(
            galaxy_id, 
            filter_name, 
            image_data, 
            aperture_params,
            sigma_val=sigma, 
            annulus_factor=annulus_factor, 
            fig_path=fig_path
        )                                                                   
        
        # Measure flux
        flux_measurements = measure_flux(
            image_data, 
            image_error,
            background_plane,  # formerly background_median
            background_std, 
            aperture_params
        )

        # Get PSF and calculate aperture correction
        correction_factor = calculate_aperture_correction(psf_data, aperture_params)
        correction_factor_copy = correction_factor
        
        # Overwrite aperture correction factor if no correction should be applied
        if apply_aper_corr == False:
            correction_factor = 1.0
            
        # Apply aperture correction
        corrected_flux = flux_measurements['flux'] * correction_factor
        corrected_flux_error = flux_measurements['flux_error'] * correction_factor
        corrected_background_flux = flux_measurements['background_flux'] * correction_factor
        corrected_background_error = background_std * correction_factor
        corrected_median_error = flux_measurements['median_error'] * correction_factor
        
        # --- Convert fluxes into AB magnitudes ---
        if corrected_flux > 0:
            # constant is 8.90 for Jy and 23.90 for µJy
            ab_mag = -2.5 * np.log10(corrected_flux) + 8.90
        else: ab_mag = np.nan
        
        # Append results
        results.append({
            'ID': int(galaxy_id),
            'Flux': corrected_flux,
            'Flux_Err': corrected_flux_error,
            'Image_Err': corrected_median_error,
            'Flux_BKG': corrected_background_flux,
            'Flux_BKG_Err': corrected_background_error,
            'AB_Mag': ab_mag,
            'Apr_Corr': correction_factor_copy,
            'N_PIX': flux_measurements['pixel_count'],
            'Apr_A': aperture_params['a'] * 2,  # Convert back to diameter for output
            'Apr_B': aperture_params['b'] * 2,  # Convert back to diameter for output
            'Apr_Xcenter': aperture_params['x_center'],
            'Apr_Ycenter': aperture_params['y_center'],
            'Apr_Theta': (aperture_params['theta'] * u.rad).to_value(u.deg)  # Convert to degrees for output
        })
    
    # Save to output table (assuming it's a pandas DataFrame)
    os.makedirs(os.path.join(output_folder, 'results'), exist_ok=True)
    output_path = os.path.join(output_folder, f'results/phot_table_{filter_name}{suffix}.csv')
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
                        'AB_Mag', 'Apr_Corr']

    # These columns will be scalar (one value per galaxy)
    scalar_columns = ['N_PIX', 'Apr_A', 'Apr_B', 'Apr_Xcenter', 'Apr_Ycenter', 'Apr_Theta']

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


def combine_filter_csv_to_fits(f770w_fname, f1800w_fname, fits_table_name):
    """
    Combine filter-specific CSV files into a single FITS table.

    Parameters:
    -----------
    f770w_fname : str
        Filename of the F770W CSV table containing the aperture photometry results
    f1800w_fname : str
        Filename of the F1800W CSV table containing the aperture photometry results
    fits_table_name : str
        Filename of the output FITS table
    suffix : str, optional
        Suffix to the final output table to keep track of versions locally
    """
    # Define results folder as a static variable
    results_folder = '/home/bpc/University/master/Red_Cardinal/photometry/results/'
    
    # CSV files for each filter
    f770w_csv = os.path.join(results_folder, f770w_fname)
    f1800w_csv = os.path.join(results_folder, f1800w_fname)

    # Output FITS file
    fits_output = os.path.join(results_folder, fits_table_name)

    # Create the combined FITS table
    table = create_fits_table_from_csv(f770w_csv, f1800w_csv, fits_output)

    return table





def compare_aperture_statistics(table_small_path, table_big_path, fig_path, summary_doc_path, scaling=None):
    """
    Compare and contrast two photometric tables WITHOUT APERTURE CORRECTION APPLIED
    and create a comprehensive summary plot of all important statistics and write
    the output to a text file.

    Args:
        table_small_path (str):
            Path to table using small apertures
        table_big_path (str): 
            Path to table using big apertures
        fig_path (str):
            Output path of the summary plot
        summary_doc_path (str): 
            Output path of the summary text file
        scaling (str) optional:
            'log' for logarithmic, default is linear
    """
    # Enhanced Aperture Photometry Comparison
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.table import Table
    from matplotlib.patches import Rectangle
    import seaborn as sns

    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")

    # Load the two tables
    table_small_path = '/home/bpc/University/master/Red_Cardinal/photometry/results/Flux_SmallAperture_NoCorr_MIRI.fits'
    table_big_path = '/home/bpc/University/master/Red_Cardinal/photometry/results/Flux_BigAperture_NoCorr_MIRI.fits'

    table_small = Table.read(table_small_path)
    table_big = Table.read(table_big_path)

    # Convert ID columns to string for alignment
    ids1 = [id.decode() if isinstance(id, bytes) else str(id) for id in table_small['ID']]
    ids2 = [id.decode() if isinstance(id, bytes) else str(id) for id in table_big['ID']]

    # Match common IDs
    common_ids = sorted(set(ids1) & set(ids2))
    print(f"Found {len(common_ids)} common galaxies")

    # Prepare data structures
    bands = ['F770W', 'F1800W']
    data_comparison = {
        'ID': [],
        'Band': [],
        'Flux_Small_Raw': [],
        'Flux_Big_Raw': [],
        'Flux_Small_Corrected': [],
        'Flux_Big_Corrected': [],
        'Apr_Corr_Small': [],
        'Apr_Corr_Big': [],
        'Flux_Ratio': [],
        'Corrected_Flux_Ratio': [],
        'Flux_Difference': [],
        'Corrected_Flux_Difference': []
    }

    # Collect all data for comprehensive analysis
    for idx, band in enumerate([0, 1]):  # F770W = 0, F1800W = 1
        for gid in common_ids:
            i1 = ids1.index(gid)
            i2 = ids2.index(gid)

            # Raw fluxes (convert to µJy)
            flux_small = table_small['Flux'][i1][idx] * 1e6
            flux_big = table_big['Flux'][i2][idx] * 1e6
            
            # Aperture corrections
            corr_small = table_small['Apr_Corr'][i1][idx] if 'Apr_Corr' in table_small.colnames else np.nan
            corr_big = table_big['Apr_Corr'][i2][idx] if 'Apr_Corr' in table_big.colnames else np.nan
            
            # Skip if any crucial value is invalid
            if not (np.isfinite(flux_small) and np.isfinite(flux_big) and 
                    np.isfinite(corr_small) and np.isfinite(corr_big)):
                continue
                
            # Calculate corrected fluxes
            flux_small_corr = flux_small * corr_small
            flux_big_corr = flux_big * corr_big
            
            # Store all data
            data_comparison['ID'].append(gid)
            data_comparison['Band'].append(bands[idx])
            data_comparison['Flux_Small_Raw'].append(flux_small)
            data_comparison['Flux_Big_Raw'].append(flux_big)
            data_comparison['Flux_Small_Corrected'].append(flux_small_corr)
            data_comparison['Flux_Big_Corrected'].append(flux_big_corr)
            data_comparison['Apr_Corr_Small'].append(corr_small)
            data_comparison['Apr_Corr_Big'].append(corr_big)
            data_comparison['Flux_Ratio'].append(flux_big / flux_small)
            data_comparison['Corrected_Flux_Ratio'].append(flux_big_corr / flux_small_corr)
            data_comparison['Flux_Difference'].append(flux_big - flux_small)
            data_comparison['Corrected_Flux_Difference'].append(flux_big_corr - flux_small_corr)

    # Convert to arrays for easier handling
    for key in data_comparison:
        data_comparison[key] = np.array(data_comparison[key])

    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(20, 16))

    # 1. Raw vs Corrected Flux Comparison (Scatter plots)
    for i, band in enumerate(bands):
        mask = data_comparison['Band'] == band
        
        # Raw fluxes
        ax1 = plt.subplot(4, 4, 1 + i*2)
        plt.scatter(data_comparison['Flux_Small_Raw'][mask], 
                    data_comparison['Flux_Big_Raw'][mask], 
                    alpha=0.7, s=30)
        
        # Add 1:1 line
        min_flux = min(np.min(data_comparison['Flux_Small_Raw'][mask]), 
                    np.min(data_comparison['Flux_Big_Raw'][mask]))
        max_flux = max(np.max(data_comparison['Flux_Small_Raw'][mask]), 
                    np.max(data_comparison['Flux_Big_Raw'][mask]))
        plt.plot([min_flux, max_flux], [min_flux, max_flux], 'r--', alpha=0.8, label='1:1')
        if scaling: plt.loglog()
        plt.xlabel(f'{band} Small Aperture Raw Flux [µJy]')
        plt.ylabel(f'{band} Large Aperture Raw Flux [µJy]')
        plt.title(f'{band} Raw Flux Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Corrected fluxes
        ax2 = plt.subplot(4, 4, 2 + i*2)
        plt.scatter(data_comparison['Flux_Small_Corrected'][mask], 
                    data_comparison['Flux_Big_Corrected'][mask], 
                    alpha=0.7, s=30, color='orange')
        
        min_flux_corr = min(np.min(data_comparison['Flux_Small_Corrected'][mask]), 
                            np.min(data_comparison['Flux_Big_Corrected'][mask]))
        max_flux_corr = max(np.max(data_comparison['Flux_Small_Corrected'][mask]), 
                            np.max(data_comparison['Flux_Big_Corrected'][mask]))
        plt.plot([min_flux_corr, max_flux_corr], [min_flux_corr, max_flux_corr], 'r--', alpha=0.8, label='1:1')
        if scaling: plt.loglog()
        plt.xlabel(f'{band} Small Aperture Corrected Flux [µJy]')
        plt.ylabel(f'{band} Large Aperture Corrected Flux [µJy]')
        plt.title(f'{band} Corrected Flux Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 2. Flux Ratios (Large/Small aperture)
    for i, band in enumerate(bands):
        mask = data_comparison['Band'] == band
        
        # Raw flux ratios
        ax3 = plt.subplot(4, 4, 5 + i*2)
        plt.hist(data_comparison['Flux_Ratio'][mask], bins=25, alpha=0.7, 
                color='skyblue', edgecolor='black')
        plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Unity')
        plt.axvline(np.median(data_comparison['Flux_Ratio'][mask]), 
                    color='orange', linestyle='-', linewidth=2, label='Median')
        plt.xlabel('Flux Ratio (Large/Small)')
        plt.ylabel('Number of Sources')
        plt.title(f'{band} Raw Flux Ratio Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        ratio_stats = f"Median: {np.median(data_comparison['Flux_Ratio'][mask]):.3f}\n" + \
                    f"Mean: {np.mean(data_comparison['Flux_Ratio'][mask]):.3f}\n" + \
                    f"Std: {np.std(data_comparison['Flux_Ratio'][mask]):.3f}"
        plt.text(0.95, 0.95, ratio_stats, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Corrected flux ratios
        ax4 = plt.subplot(4, 4, 6 + i*2)
        plt.hist(data_comparison['Corrected_Flux_Ratio'][mask], bins=25, alpha=0.7, 
                color='lightcoral', edgecolor='black')
        plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Unity')
        plt.axvline(np.median(data_comparison['Corrected_Flux_Ratio'][mask]), 
                    color='orange', linestyle='-', linewidth=2, label='Median')
        plt.xlabel('Corrected Flux Ratio (Large/Small)')
        plt.ylabel('Number of Sources')
        plt.title(f'{band} Corrected Flux Ratio Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        corr_ratio_stats = f"Median: {np.median(data_comparison['Corrected_Flux_Ratio'][mask]):.3f}\n" + \
                        f"Mean: {np.mean(data_comparison['Corrected_Flux_Ratio'][mask]):.3f}\n" + \
                        f"Std: {np.std(data_comparison['Corrected_Flux_Ratio'][mask]):.3f}"
        plt.text(0.95, 0.95, corr_ratio_stats, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. Aperture Correction Comparison
    for i, band in enumerate(bands):
        mask = data_comparison['Band'] == band
        
        ax5 = plt.subplot(4, 4, 9 + i*2)
        plt.scatter(data_comparison['Apr_Corr_Small'][mask], 
                    data_comparison['Apr_Corr_Big'][mask], 
                    alpha=0.7, s=30, color='green')
        
        min_corr = min(np.min(data_comparison['Apr_Corr_Small'][mask]), 
                    np.min(data_comparison['Apr_Corr_Big'][mask]))
        max_corr = max(np.max(data_comparison['Apr_Corr_Small'][mask]), 
                    np.max(data_comparison['Apr_Corr_Big'][mask]))
        plt.plot([min_corr, max_corr], [min_corr, max_corr], 'r--', alpha=0.8, label='1:1')
        if scaling: plt.loglog()
        plt.xlabel(f'{band} Small Aperture Correction')
        plt.ylabel(f'{band} Large Aperture Correction')
        plt.title(f'{band} Aperture Corrections')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Corrected flux differences
        ax6 = plt.subplot(4, 4, 10 + i*2)
        plt.hist(data_comparison['Corrected_Flux_Difference'][mask], bins=25, alpha=0.7, 
                color='purple', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        plt.axvline(np.median(data_comparison['Corrected_Flux_Difference'][mask]), 
                    color='orange', linestyle='-', linewidth=2, label='Median')
        plt.xlabel('Corrected Flux Difference [µJy]')
        plt.ylabel('Number of Sources')
        plt.title(f'{band} Corrected Flux Difference (Large - Small)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        diff_stats = f"Median: {np.median(data_comparison['Corrected_Flux_Difference'][mask]):.2f} µJy\n" + \
                    f"Mean: {np.mean(data_comparison['Corrected_Flux_Difference'][mask]):.2f} µJy\n" + \
                    f"Std: {np.std(data_comparison['Corrected_Flux_Difference'][mask]):.2f} µJy"
        plt.text(0.95, 0.95, diff_stats, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 4. Flux vs Ratio relationships (to identify systematic trends)
    for i, band in enumerate(bands):
        mask = data_comparison['Band'] == band
        
        ax7 = plt.subplot(4, 4, 13 + i*2)
        plt.scatter(data_comparison['Flux_Small_Raw'][mask], 
                    data_comparison['Flux_Ratio'][mask], 
                    alpha=0.6, s=30, c=data_comparison['Apr_Corr_Small'][mask], 
                    cmap='viridis')
        plt.colorbar(label='Small Aperture Correction')
        plt.axhline(1.0, color='red', linestyle='--', alpha=0.8, label='Unity')        
        if scaling: plt.xscale('log')
        plt.xlabel(f'{band} Small Aperture Raw Flux [µJy]')
        plt.ylabel('Flux Ratio (Large/Small)')
        plt.title(f'{band} Flux Ratio vs Brightness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        ax8 = plt.subplot(4, 4, 14 + i*2)
        plt.scatter(data_comparison['Flux_Small_Corrected'][mask], 
                    data_comparison['Corrected_Flux_Ratio'][mask], 
                    alpha=0.6, s=30, c=data_comparison['Apr_Corr_Big'][mask], 
                    cmap='plasma')
        plt.colorbar(label='Large Aperture Correction')
        plt.axhline(1.0, color='red', linestyle='--', alpha=0.8, label='Unity')
        if scaling: plt.xscale('log')
        plt.xlabel(f'{band} Small Aperture Corrected Flux [µJy]')
        plt.ylabel('Corrected Flux Ratio (Large/Small)')
        plt.title(f'{band} Corrected Flux Ratio vs Brightness')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.suptitle('Comprehensive Aperture Photometry Comparison', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(fig_path, dpi=150)
    plt.show()
    plt.close()

    with open(summary_doc_path, "w") as file:
        file.write("\n" + "="*80 + "\n")
        file.write("COMPREHENSIVE APERTURE COMPARISON SUMMARY\n")
        file.write("="*80 + "\n")
        
        for band in bands:
            mask = data_comparison['Band'] == band
            file.write(f"\n📊 {band} FILTER:\n")
            file.write("-" * 40 + "\n")
            
            # Raw flux statistics
            flux_ratio_med = np.median(data_comparison['Flux_Ratio'][mask])
            flux_ratio_mean = np.mean(data_comparison['Flux_Ratio'][mask])
            flux_ratio_std = np.std(data_comparison['Flux_Ratio'][mask])
            
            file.write("Raw Flux Ratios (Large/Small):\n")
            file.write(f"  Median: {flux_ratio_med:.3f} ± {flux_ratio_std:.3f}\n")
            file.write(f"  Mean:   {flux_ratio_mean:.3f}\n")
            file.write(f"  Range:  {np.min(data_comparison['Flux_Ratio'][mask]):.3f} - {np.max(data_comparison['Flux_Ratio'][mask]):.3f}\n")
            
            # Corrected flux statistics
            corr_ratio_med = np.median(data_comparison['Corrected_Flux_Ratio'][mask])
            corr_ratio_mean = np.mean(data_comparison['Corrected_Flux_Ratio'][mask])
            corr_ratio_std = np.std(data_comparison['Corrected_Flux_Ratio'][mask])
            
            file.write("\nCorrected Flux Ratios (Large/Small):\n")
            file.write(f"  Median: {corr_ratio_med:.3f} ± {corr_ratio_std:.3f}\n")
            file.write(f"  Mean:   {corr_ratio_mean:.3f}\n")
            file.write(f"  Range:  {np.min(data_comparison['Corrected_Flux_Ratio'][mask]):.3f} - {np.max(data_comparison['Corrected_Flux_Ratio'][mask]):.3f}\n")
            
            # Aperture correction comparison
            small_corr_med = np.median(data_comparison['Apr_Corr_Small'][mask])
            big_corr_med = np.median(data_comparison['Apr_Corr_Big'][mask])
            
            file.write("\nAperture Corrections:\n")
            file.write(f"  Small aperture median: {small_corr_med:.3f}\n")
            file.write(f"  Large aperture median: {big_corr_med:.3f}\n")
            file.write(f"  Difference (Large-Small): {big_corr_med - small_corr_med:.3f}\n")
            
            # Final corrected flux differences
            corr_diff_med = np.median(data_comparison['Corrected_Flux_Difference'][mask])
            corr_diff_mean = np.mean(data_comparison['Corrected_Flux_Difference'][mask])
            corr_diff_std = np.std(data_comparison['Corrected_Flux_Difference'][mask])
            
            file.write("\nFinal Corrected Flux Differences (Large - Small) [µJy]:\n")
            file.write(f"  Median: {corr_diff_med:.2f} ± {corr_diff_std:.2f}\n")
            file.write(f"  Mean:   {corr_diff_mean:.2f}\n")
            
            # Percentage of sources where large aperture gives higher flux
            higher_flux_pct = np.sum(data_comparison['Corrected_Flux_Difference'][mask] > 0) / np.sum(mask) * 100
            file.write(f"  Sources with higher flux in large aperture: {higher_flux_pct:.1f}%\n")

        file.write(f"\nTotal sources analyzed: {len(common_ids)}\n")
        file.write("="*80 + "\n")