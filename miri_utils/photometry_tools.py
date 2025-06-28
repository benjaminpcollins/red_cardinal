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
import h5py
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
from collections import defaultdict


from .cutout_tools import load_cutout

# Suppress common WCS-related warnings that don't affect functionality
warnings.simplefilter("ignore", category=FITSFixedWarning)


def adjust_aperture(galaxy_id, filter, survey, obs, output_folder, mask_folder='masks', save_plot=False):
    
    # --- Load the FITS table ---
    table_path =  '/home/bpc/University/master/Red_Cardinal/catalogues/Flux_Aperture_PSFMatched_AperCorr_old.fits'
    aperture_table = Table.read(table_path)
    #table_path =  '/home/bpc/University/master/Red_Cardinal/catalogues/aperture_table.csv'
    #df = pd.read_csv(table_path)
    
    # --- Select the galaxy by ID ---
    matches = aperture_table[aperture_table['ID'] == galaxy_id]
    if len(matches) == 0:
        print(f"Galaxy ID {galaxy_id} not found in the table.")
        return None
    else:
        row = matches[0]
    #row = df[df['ID'] == galaxy_id].iloc[0]

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
    pixel_conversion = scale_factor
    
    ####################################################
    ### Additional rescaling of the NIRCam apertures ###
    ####################################################
    
    if int(galaxy_id) == 12332:
        scale_factor *= 1.0 # no additional scaling to avoid contaminating source
    elif int(galaxy_id) in [7136, 7904, 7922, 11136, 16419, 21452]:
        scale_factor *= 1.6
    elif int(galaxy_id) in [7934, 10314, 10592, 18332]:
        scale_factor *= 1.8
    else:
        scale_factor *= 2.0
    
    # --- Specify parameters for the ellipse ---
    width = row['Apr_A'] * scale_factor
    height = row['Apr_B'] * scale_factor
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
            width=row['Apr_A'] * pixel_conversion,     # Here we just
            height=row['Apr_B'] * pixel_conversion,    # leave the 
            angle=theta,                               # original values
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
        mask_dir = os.path.join(output_folder, mask_folder)
        os.makedirs(mask_dir, exist_ok=True)
        png_path = os.path.join(mask_dir, f'{galaxy_id}_{survey}{obs}_aperture_overlay.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    # Collect modified aperture data including full context
    aperture_info = {
        'ID': galaxy_id,
        'Filter': filter,
        'Survey': survey,
        'Obs': obs,
        'Apr_A': width,
        'Apr_B': height,
        'Apr_Xcenter': miri_x,
        'Apr_Ycenter': miri_y,
        'Apr_Theta': theta_new.to_value(u.deg)
    }
    
    return aperture_info




def estimate_background(galaxy_id, filter_name, image_data, aperture_params, sigma_val):
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
        
    Returns
    -------
    - background_std : float
        standard deviation of background model within the annulus (excluding clipped data)
    - background_median : float
        median background value within the annulus
    - background_plane : ndarray
        2D background model

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
    total_mask = source_mask_bool | np.isnan(image_data)
    
    # Step 1: Get initial background estimate using sigma clipping
    # This doesn't require knowing sources beforehand
    mean_bg, median_bg, std_bg = sigma_clipped_stats(
        image_data, sigma=2.5, maxiters=5, mask=total_mask
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
    
    a_in = a + 8
    b_in = b + 8
    
    # Check how much space there is to the boundary
    delta_x = min(x_center, 72-x_center)
    delta_y = min(y_center, 72-y_center)
    
    # Maximum theoretically possible radius for a circle
    r_max = max(delta_x, delta_y) - 2   # for boundary buffer
    
    # Specify a_out and b_out
    a_out = r_max
    b_out = 0.9*r_max
    
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
    annulus_mask_bool = annulus_mask.astype(bool)
    background_region_mask = annulus_mask_bool & ~np.isnan(image_data) & ~segm_mask

    # Gather all the pixels within the background region that were NOT sigma-clipped
    bkg_region_valid_pixels = ~segm_mask & background_region_mask
    data_bkgsub = image_data - background_plane
    
    # Calculate background statistics
    background_residuals = data_bkgsub[bkg_region_valid_pixels]
    sigma_bkg = np.std(background_residuals)
    
    # The following should be the 1-sigma uncertainty on the total background flux in the aperture
    background_std = sigma_bkg * np.sqrt(source_aperture.area)
    
    # Extract background plane values for the background region and calculate the median
    background_median = np.median(background_residuals)
    
    # Create a mask visualisation
    mask_vis = np.zeros_like(image_data, dtype=int)
    mask_vis[~combined_mask] = 1  # Excluded pixels
    mask_vis[background_region_mask] = 2  # Pixels in the annulus/rectangle
    mask_vis[source_mask_bool] = 3  # Source pixels
    
    # Store visualisation data
    vis_data = {
        'galaxy_id': galaxy_id,
        'filter': filter_name,
        'original_data': image_data,
        'background_plane': background_plane,
        'background_subtracted': data_bkgsub,
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
    
    # Save visualisation data to .h5 file
    vis_dir = '/home/bpc/University/master/Red_Cardinal/photometry/vis_data'
    os.makedirs(vis_dir, exist_ok=True)
    
    vis_path = os.path.join(vis_dir, f'{galaxy_id}_{filter_name}.h5')
    save_vis(vis_data, vis_path)
    
    return background_median, background_std, background_plane

def save_vis(vis_data, filename):
    """
    Save visualisation data to HDF5 file.
    
    Parameters:
    -----------
    vis_data : dict
        Dictionary containing visualization data
    filename : str
        Output filename (should end with .h5 or .hdf5)
    """
    with h5py.File(filename, 'w') as f:
        # Save arrays with compression
        for key in ['original_data', 'background_plane', 'background_subtracted', 
                   'mask_vis', 'segmentation_mask', 'background_region_mask', 
                   'source_mask']:
            if key in vis_data and vis_data[key] is not None:
                f.create_dataset(key, data=vis_data[key], compression='gzip', compression_opts=6)
        
        # Save scalars
        for key in ['galaxy_id', 'a_in', 'b_in', 'a_out', 'b_out', 'sigma']:
            if key in vis_data and vis_data[key] is not None:
                f.attrs[key] = vis_data[key]
        
        # Save strings
        for key in ['filter', 'region_name']:
            if key in vis_data and vis_data[key] is not None:
                f.attrs[key] = vis_data[key].encode('utf-8') if isinstance(vis_data[key], str) else vis_data[key]
        
        # Save coefficients tuple
        if 'coeffs' in vis_data and vis_data['coeffs'] is not None:
            f.create_dataset('coeffs', data=np.array(vis_data['coeffs']))
        
        # Save aperture_params dict as JSON string
        if 'aperture_params' in vis_data and vis_data['aperture_params'] is not None:
            f.attrs['aperture_params'] = json.dumps(vis_data['aperture_params'])
        
        # Add metadata
        f.attrs['created_date'] = np.string_(str(np.datetime64('now')))
        f.attrs['data_type'] = np.string_('galaxy_visualization_data')

def load_vis(filename):
    """
    Load visualisation data from HDF5 file.
    
    Parameters:
    -----------
    filename : str
        Input filename
        
    Returns:
    --------
    dict : Loaded visualisation data
    """
    vis_data = {}
    
    with h5py.File(filename, 'r') as f:
        # Load arrays
        for key in ['original_data', 'background_plane', 'background_subtracted', 
                   'mask_vis', 'segmentation_mask', 'background_region_mask', 
                   'source_mask']:
            if key in f:
                vis_data[key] = f[key][:]
        
        # Load coefficients
        if 'coeffs' in f:
            vis_data['coeffs'] = tuple(f['coeffs'][:])
        
        # Load scalars from attributes
        for key in ['galaxy_id', 'a_in', 'b_in', 'a_out', 'b_out', 'sigma']:
            if key in f.attrs:
                vis_data[key] = f.attrs[key]
        
        # Load strings from attributes
        for key in ['filter', 'region_name']:
            if key in f.attrs:
                val = f.attrs[key]
                if isinstance(val, bytes):
                    vis_data[key] = val.decode('utf-8')
                else:
                    vis_data[key] = val
        
        # Load aperture_params dict
        if 'aperture_params' in f.attrs:
            vis_data['aperture_params'] = json.loads(f.attrs['aperture_params'])
    
    return vis_data

def create_mosaics(input_dir, mosaic_dir=None, plane_sub_dir=None):
    all_files = glob.glob(os.path.join(input_dir, '*.h5'))
    all_ids = np.unique([os.path.basename(f).split('_')[0] for f in all_files])

    def plot_aperture_overlay(ax, data, aperture, cmap='magma', label='', percentile=(5, 95)):
        vmin, vmax = np.nanpercentile(data, percentile)
        im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        aperture.plot(ax=ax, color='blue', lw=4)
        ax.set_title(label)
        return im
    
    if plane_sub_dir:
        os.makedirs(plane_sub_dir, exist_ok=True)

        for file in all_files:
            vis = load_vis(file)
            image_data = vis['original_data']
            background_plane = vis['background_plane']
            background_subtracted = vis['background_subtracted']
            aperture_params = vis['aperture_params']
            filter = vis['filter']
            galaxy_id = vis['galaxy_id']

            aperture = EllipticalAperture(
                positions=(aperture_params['x_center'], aperture_params['y_center']),
                a=aperture_params['a'],
                b=aperture_params['b'],
                theta=aperture_params['theta']
            )

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            im0 = plot_aperture_overlay(axes[0], image_data, aperture, label="Original Data with Aperture")
            im1 = axes[1].imshow(background_plane, origin='lower', cmap='viridis')
            axes[1].set_title("Global 2D Background Plane")
            im2 = plot_aperture_overlay(axes[2], background_subtracted, aperture, label="Background-subtracted Data")

            for ax, im, label in zip(axes, [im0, im1, im2], [
                'Flux [MJy/(sr pixel)]',
                'Background Flux [MJy/(sr pixel)]',
                'Background-subtracted Flux [MJy/(sr pixel)]'
            ]):
                plt.colorbar(im, ax=ax, label=label)

            fig.suptitle(f'{filter}', fontsize=18)
            plt.tight_layout()
            plt.savefig(os.path.join(plane_sub_dir, f'{galaxy_id}_{filter}_planefit.png'), dpi=150)
            plt.close(fig)

    if mosaic_dir:
        os.makedirs(mosaic_dir, exist_ok=True)

        for gid in all_ids:
            print(f"Processing galaxy {gid}...")
            vis_files = glob.glob(os.path.join(input_dir, f'{gid}*.h5'))
            vis_list = [load_vis(f) for f in vis_files]

            filter_order = ['F770W', 'F1000W', 'F1800W', 'F2100W']
            vis_dict = {v['filter']: v for v in vis_list}
            vis_sorted = [vis_dict[f] for f in filter_order if f in vis_dict]

            num = len(vis_sorted)
            fig, axes = plt.subplots(3, num, figsize=(4*num, 12))

            if num == 1:
                axes = np.expand_dims(axes, axis=1)

            for ii, vis in enumerate(vis_sorted):
                ap_params = vis['aperture_params']
                aperture = EllipticalAperture(
                    positions=(ap_params['x_center'], ap_params['y_center']),
                    a=ap_params['a'],
                    b=ap_params['b'],
                    theta=ap_params['theta']
                )

                # Top: original + aperture
                plot_aperture_overlay(axes[0, ii], vis['original_data'], aperture, label=vis['filter'])

                # Middle: background plane
                axes[1, ii].imshow(vis['background_plane'], origin='lower', cmap='viridis')
                axes[1, ii].set_title("Background")

                # Bottom: mask visualisation
                cmap = plt.cm.get_cmap('viridis', 4)
                axes[2, ii].imshow(vis['mask_vis'], origin='lower', cmap=cmap, vmin=-0.5, vmax=3.5)
                axes[2, ii].set_title("Mask")

            plt.tight_layout()
            plt.savefig(os.path.join(mosaic_dir, f'{gid}.png'), dpi=150)
            plt.close(fig)            
            
        


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
    source_aperture.plot(ax=axes[0, 0], color='blue', lw=4)
    
    # Overlay the segmentation mask as white outlines
    axes[0, 0].contour(segm_mask, levels=[0.5], colors='green', linewidths=2.5)
        
    axes[0, 0].set_title("Original Data with Aperture and Masked Regions")
    
    # Background-subtracted data
    vmin2 = np.nanpercentile(background_subtracted, 5)
    vmax2 = np.nanpercentile(background_subtracted, 95)
    
    im1 = axes[0, 1].imshow(background_subtracted, origin='lower', cmap='magma', vmin=vmin2, vmax=vmax2)
    plt.colorbar(im1, ax=axes[0, 1], label='Background-subtracted Flux [MJy/(sr pixel)]')
    source_aperture.plot(ax=axes[0, 1], color='blue', lw=4)
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
        filepath = os.path.join(fig_path, f'{galaxy_id}_{filter}.png')
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

def get_aperture_params(galaxy_id, filter, aperture_table):
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
    
    # Look for the unique combination of ID and filter
    row = df[(df['ID'] == int(galaxy_id)) & (df['Filter'] == filter)].iloc[0]
    
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
    psf_data
        Loaded PSF data
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

def measure_flux(image_data, error_map, background_median, background_std, background_plane, aperture_params):
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
    background_flux = background_median * aperture.area
    
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

def perform_photometry(cutout_files, aperture_table, output_folder, suffix='', 
                       sigma=2.0, apply_aper_corr=True):
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
    suffix : str, optional
        Choose a custom filename extension to track version history
    sigma : float, optional
        Manually set threshold for the sigma clipping. Default value is 2.0
    apply_aper_corr : bool, optional
        Decide whether aperture correction should be applied
    """
    results = []
    
    psf_f770w  = get_psf('F770W')
    psf_f1000w = get_psf('F1000W')
    psf_f1800w = get_psf('F1800W')
    psf_f2100w = get_psf('F2100W')
    
    psf_dict = {
        'F770W':  psf_f770w,
        'F1000W': psf_f1000w,
        'F1800W': psf_f1800w,
        'F2100W': psf_f2100w
    }
    
    #########################################################
    #####         Initialise Filtering Arrays           #####
    #########################################################    
    exclude_all = [18094, 19307]
    exclude_filters = {
        'F770W': [16424],
        'F1000W': [],
        'F1800W': [12202, 12332, 16419],
        'F2100W': [7102, 16874]
    }
    art_filters = {
        'F770W': [7185, 8013, 8469, 8500, 8843, 9517, 11136,
                    11137, 11494, 11716, 16516, 17793, 19098, 21451],
        'F1000W': [],
        'F1800W': [7102, 11716, 12202, 17793, 19098, 21451],
        'F2100W': [11723, 12175, 12213, 16874, 17984]
    }
    has_companion = [7136, 7904, 7922, 7934, 8469, 10314,
                        16424, 17517, 18332, 21452]
    
    
    for fits_path in cutout_files:
        # Extract ID and filter from filename
        fits_name = os.path.basename(fits_path)
        galaxy_id = fits_name.split('_')[0]
        filter_name = fits_name.split('_')[1]
        psf_data = psf_dict[filter_name]
        
        #########################################################
        #####               Filtering Section               #####
        #########################################################
        
        companion_flag = False
        artefact_flag = False
        
        # Galaxies to exclude from analysis
        if int(galaxy_id) in exclude_all:
            continue
        
        # Galaxies that have companions that could cause contamination
        if int(galaxy_id) in has_companion:
            companion_flag = True
        
        # Galaxies that should be excluded in only one filter or show weird detector artefacts
        if filter_name in exclude_filters:
            # Exclude from analysis
            if int(galaxy_id) in exclude_filters[filter_name]:
                continue
        
        if filter_name in art_filters:
            # Artefact
            if int(galaxy_id) in art_filters[filter_name]:
                artefact_flag = True
        
        #########################################################
        #####               Start Photometry                #####
        #########################################################
        
        print(f'Processing galaxy {galaxy_id} with filter {filter_name}...')
        
        # Load image data
        with fits.open(fits_path) as hdul:
            image_data = hdul['SCI'].data if 'SCI' in hdul else hdul[1].data
            image_error = hdul['ERR'].data if 'ERR' in hdul else hdul[2].data
            
        # Get aperture parameters
        aperture_params = get_aperture_params(galaxy_id, filter_name, aperture_table)
        
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
            
        
        #########################################################
        #####              Estimate Background              #####
        #########################################################
        
        background_median, background_std, background_plane = estimate_background(
            galaxy_id, 
            filter_name, 
            image_data, 
            aperture_params,
            sigma_val=sigma, 
        )                                                                   
        
        #########################################################
        #####                 Measure Flux                  #####
        #########################################################
        
        flux_measurements = measure_flux(
            image_data, 
            image_error,
            background_median,
            background_std, 
            background_plane,
            aperture_params
        )

        #########################################################
        #####           Apply Aperture Correction           #####
        #########################################################        
        
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
            'Apr_Theta': (aperture_params['theta'] * u.rad).to_value(u.deg),  # Convert to degrees for output
            'Flag_Com': companion_flag,
            'Flag_Art': artefact_flag
        })
    
    suffix = '_' + suffix
    # Save to output table (assuming it's a pandas DataFrame)
    os.makedirs(os.path.join(output_folder, 'results'), exist_ok=True)
    output_path = os.path.join(output_folder, f'results/phot_table_{filter_name}{suffix}.csv')
    output_df = pd.DataFrame(results)
    
    output_df.to_csv(output_path, index=False) 
 
 
def create_fits_table_from_csv(csv_paths, output_file):
    """
    Create a FITS table by combining photometry data from multiple MIRI filter CSV files.
    
    Parameters:
    -----------
    csv_fnames : list of str
        List of paths to CSV files corresponding to the filters (in the same order as 'filters').
    output_file : str
        Output FITS file name.
    skip_gid : list of int
        List of galaxy IDs to skip for photometric analysis e.g. due to incomplete coverage
    """
    
    # Check which filters are used based on the csv files
    filters = []
    for csv_file in csv_paths:
        filter = os.path.basename(csv_file).split('_')[2]
        filters.append(filter)
        
    # Check that at least one CSV file exists
    valid_csvs = [(path, f) for path, f in zip(csv_paths, filters) if path and os.path.exists(path)]
    if not valid_csvs:
        raise ValueError("At least one valid CSV file is required.")

    # Load all valid CSVs into a dict
    dfs = {}
    all_ids = set()

    for path, filt in valid_csvs:
        df = pd.read_csv(path)
        print(f"Loaded {filt} data with {len(df)} rows")
        dfs[filt] = df
        # Convert ID column of the df to a list and create the union of this
        # list with all_ids to filter out the unique IDs - genius!
        all_ids.update(df['ID'].tolist())   

    # Sort and convert to list
    galaxy_ids = sorted(all_ids)
    print(f"Found {len(galaxy_ids)} unique galaxy IDs across all filters")
    
    # Initialize table
    table = Table()

    # Define columns
    array_columns =  [
        'Flux',             # flux in µJy
        'Flux_Err',         # total error
        'Image_Err',        # median per-pixel uncertainty of the 
                            # 'ERR'-extension within the aperture
        'Flux_BKG',         # background flux within the aperture
        'Flux_BKG_Err',     # uncertainty on background
        'AB_Mag',           # AB magnitude
        'Apr_Corr',         # aperture correction
    
        # Aperture position/orientation - varies with WCS per filter
        'Apr_Xcenter',      # aperture X-centre in MIRI pixels
        'Apr_Ycenter',      # aperture Y-centre
        'Apr_Theta',        # rotation angle in degrees
        'Flag_Art'          # flag for weird detector artefacts
    ]
    
    scalar_columns = [ 
        'Apr_A',            # semi-major axis in MIRI pixels 
        'Apr_B',            # semi-minor axis
        'N_PIX',            # pixel count within the aperture
        'Flag_Com'         # flag for companion in the frame
    ]
    
    # Specify columns that might contain masked values and should be ignored by astropy
    masked_columns = ['Flux_Err', 'Flux_BKG', 'AB_Mag', 'Flux_BKG_Err']

    # Prepare data structures
    column_data = {col: [] for col in array_columns + scalar_columns}
    filters_data = []  # Separate handling for filters

    for gid in galaxy_ids:
        row_data = {col: [] for col in array_columns}
        
        # Determine base row from the first available filter table for scalar values
        base_row = None
        for df in dfs.values():
            if gid in df['ID'].values:
                base_row = df[df['ID'] == gid].iloc[0]
                break
        
        if base_row is None:
            raise ValueError(f"Galaxy ID {gid} not found in any input CSV.")
        
        filters_present = []  # Track available filters for this galaxy
        
        # Process each filter to build array data
        for filt in filters:
            df = dfs.get(filt, None)
            if df is not None and gid in df['ID'].values:
                row = df[df['ID'] == gid].iloc[0]  
                filters_present.append(filt)    # Record that the filter is available
            else:
                row = None

            # Build array data for this filter
            for col in array_columns:
                if row is not None:
                    row_data[col].append(row[col])
                else:
                    # Handle different default values for different column types
                    if col == 'Flag_Art':
                        default_val = False  
                    else:
                        default_val = np.nan
                    if row is not None:
                        row_data[col].append(row[col])
                    else:
                        row_data[col].append(default_val)

        # Store filters as a comma-separated string instead of a list
        filters_data.append(','.join(filters_present))
        
        # Append array data to main storage
        for col in array_columns:
            column_data[col].append(row_data[col])

        # Append scalar data from base row (only process columns that are actually scalar)
        for col in scalar_columns:
            column_data[col].append(base_row[col])
    
    # Add ID column as strings (astropy will handle FITS conversion)
    table.add_column(galaxy_ids, name='ID')

    # Add array columns with masking
    for col in array_columns:
        if col in masked_columns:
            masked_col = MaskedColumn(column_data[col], name=col, dtype=np.float64)
            for i, row in enumerate(column_data[col]):
                for j, val in enumerate(row):
                    if np.isnan(val):
                        masked_col.mask[i, j] = True
            if col == 'AB_Mag':
                for i, flux_row in enumerate(column_data['Flux']):
                    for j, flux_val in enumerate(flux_row):
                        if flux_val <= 0 or np.isnan(flux_val):
                            masked_col.mask[i, j] = True
            table.add_column(masked_col)
        else:
            table.add_column(column_data[col], name=col)

    # Add filters column as strings (astropy will handle FITS conversion)
    table.add_column(filters_data, name='Filters')

    # Add scalar columns
    for col in scalar_columns:
        table.add_column(column_data[col], name=col)

    print("\nFITS Table Summary:")
    print("="*50)
    print("DATA STRUCTURE OVERVIEW:")
    print("- Array columns: Values stored per filter (length = number of filters)")
    print("- Scalar columns: Single value per galaxy ID")
    print("- Flag_Art: Boolean array, True/False per filter for each galaxy")
    print("- Flag_Com: Boolean scalar, True/False per galaxy (same across all filters)")
    print("- Filters: Comma-separated string of available filters per galaxy")
    print("="*50)
    print(table.info())

    phot_tables_dir = '/home/bpc/University/master/Red_Cardinal/photometry/phot_tables/'
    os.makedirs(phot_tables_dir, exist_ok=True)
    fits_output = os.path.join(phot_tables_dir, output_file)
    table.write(fits_output, format='fits', overwrite=True)
    print(f"\nSaved FITS table to {fits_output}")

    return table




def galaxy_statistics(table_path, fig_path=None, stats_path=None):
    """
    Analyse how many galaxies are observed in each filter, and which galaxy IDs appear per filter.

    Parameters:
    -----------
    table_path : str
        Path to the FITS table

    Returns:
    --------
    filter_id_map : dict
        Dictionary mapping each filter to a set of galaxy IDs.
    """
    table = Table.read(table_path, format='fits')

    if 'Filters' not in table.colnames or 'ID' not in table.colnames:
        raise ValueError("FITS table must contain 'Filters' and 'ID' columns.")

    filter_id_map = defaultdict(set)

    for row in table:
        gid = row['ID']
        filters_str = row['Filters']
        filters = [f.strip() for f in filters_str.split(',') if f.strip()]
        for filt in filters:
            filter_id_map[filt].add(gid)

    # Print summary
    print("\nGalaxy Filter Mapping:")
    print("=" * 30)
    for filt, ids in filter_id_map.items():
        print(f"{filt:10s}: {len(ids)} galaxies")

    if fig_path:
        plot_galaxy_filter_matrix(table_path, fig_path)
        print(f'Saved output plot to {fig_path}')
    if stats_path:
        write_galaxy_stats(table, stats_path)
        print(f'Wrote galaxy statistics to {stats_path}')
    
    return filter_id_map

def write_galaxy_stats(table, output_path):
    with open(output_path, 'w') as f:
        f.write("Galaxy filter mapping summary:\n")
        f.write("="*40 + "\n")

        # 1. Which galaxy IDs are mapped in which filters
        f.write("Per-galaxy Filter Coverage:\n")
        f.write("-" * 35 + "\n")
        for row in table:
            gid = row['ID']
            filters = [f.strip() for f in row['Filters'].split(',') if f.strip()]
            f.write(f"Galaxy {gid}: {', '.join(filters)}\n")

        # 2. Group by number of filters
        f.write("\nGalaxies by Number of Filters Covered:\n")
        f.write("-" * 45 + "\n")
        filters_per_count = defaultdict(list)

        for row in table:
            gid = row['ID']
            filters = [f.strip() for f in row['Filters'].split(',') if f.strip()]
            filters_per_count[len(filters)].append((gid, filters))

        for n in sorted(filters_per_count.keys()):
            f.write(f"\nMapped in {n} filter(s): {len(filters_per_count[n])} galaxies\n")
            for gid, filt_list in filters_per_count[n]:
                f.write(f"  {gid}: {', '.join(filt_list)}\n")


def plot_galaxy_filter_matrix(table_path, fig_path):
    """
    Visualise which galaxies are observed in which filters using a binary matrix plot.

    Parameters:
    -----------
    table_path : str
        Path to the FITS file.
    fig_path : str
        Path to the output file
    """
    table = Table.read(table_path, format='fits')
    
    # Default order from blue to red for MIRI filters
    filter_order = ['F770W', 'F1000W', 'F1800W', 'F2100W']

    # Custom pastel RGB colours from blue → red
    pastel_colours = {
        'F770W': '#a6cee3',   # light pastel blue
        'F1000W': '#b2df8a',  # soft green
        'F1800W': '#fdbf6f',  # light orange
        'F2100W': '#fb9a99'   # pastel red
    }

    galaxy_ids = [str(gid) for gid in table['ID']]
    num_galaxies = len(galaxy_ids)
    chunk_size = (num_galaxies + 3) // 4  # split into 4 roughly equal parts
    chunks = [galaxy_ids[i:i + chunk_size] for i in range(0, num_galaxies, chunk_size)]

    # Calculate size for square cells
    cell_size = 0.5  # in inches per side
    num_cols = len(filter_order)
    num_rows = chunk_size
    fig_width = cell_size * num_cols * 4  # 4 panels
    fig_height = cell_size * num_rows * 0.65    # emiprical factor for approximate squares
    
    fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes[0]  # Unpack single row
        
    if len(chunks) == 1:
        axes = [axes]  # ensure iterable

    plot_number = 0
    for ax, g_ids in zip(axes, chunks):
        matrix = np.zeros((len(g_ids), len(filter_order)), dtype=int)
        g_index_map = {gid: i for i, gid in enumerate(g_ids)}

        # Fill matrix
        for row in table:
            gid = str(row['ID'])
            if gid not in g_index_map:
                continue
            g_idx = g_index_map[gid]
            filters = [f.strip() for f in row['Filters'].split(',') if f.strip()]
            for filt in filters:
                if filt in filter_order:
                    f_idx = filter_order.index(filt)
                    matrix[g_idx, f_idx] = 1
        
        table_id_to_row = {str(row['ID']): idx for idx, row in enumerate(table)}

        # Draw coloured rectangles with artefact indication
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 1:
                    base_colour = pastel_colours[filter_order[j]]
                    # Check artefact flag for this galaxy and filter
                    flag_art = False
                    gid = g_ids[i]  # galaxy ID for this row in the matrix
                    table_row_idx = table_id_to_row[str(gid)]
                    row = table[table_row_idx]

                    # Flag_Art is expected as list or array of bools per filter
                    flag_art_array = row['Flag_Art']
                    if flag_art_array is not None and len(flag_art_array) == len(filter_order):
                        flag_art = flag_art_array[j]

                    # If artefact, adjust colour (e.g., darken by 30%)
                    if flag_art:
                        # Simple darken: convert hex to RGB, darken, back to hex
                        import matplotlib.colors as mcolors
                        rgb = np.array(mcolors.to_rgb(base_colour))
                        darker_rgb = np.clip(rgb * 0.7, 0, 1)  # darken by 30%
                        colour = darker_rgb
                    else:
                        colour = base_colour

                    ax.add_patch(
                        plt.Rectangle((j, i), 1, 1, color=colour)
                    )
        
        # Modify y-axis labels to add '*' for galaxies with a companion
        y_labels = []
        for i, gid in enumerate(g_ids):
            table_row_idx = table_id_to_row[str(gid)]
            row = table[table_row_idx]
            label = str(gid)
            if row['Flag_Com'] == True:
                label += '*'  # Append asterisk if companion flag is True
            y_labels.append(label)

        # Set y-axis tick labels with modified labels
        ax.set_yticks(np.arange(len(g_ids)) + 0.5)  # Centre labels in each row
        ax.set_yticklabels(y_labels, fontsize=8)

        
        ax.set_xlim(0, len(filter_order))
        ax.set_ylim(len(g_ids), 0)
        ax.set_xticks(np.arange(len(filter_order)) + 0.5)
        ax.set_xticklabels(filter_order, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(g_ids)) + 0.5)
        ax.set_yticklabels(y_labels, fontsize=8)
        #if plot_number == 0: ax.set_ylabel("Galaxy ID", fontsize=12)
        plot_number += 1
        
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.show()


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
        
        