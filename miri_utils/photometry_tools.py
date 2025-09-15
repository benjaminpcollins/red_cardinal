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
import random
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib.colors as mcolors
import pickle as pkl

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


def adjust_aperture(galaxy_id, filter, survey, obs, output_folder, mask_folder=None, rescale=True):
    
    # --- Load the FITS table ---
    table_path =  '/Users/benjamincollins/University/master/Red_Cardinal/photometry/phot_tables/Flux_Aperture_PSFMatched_AperCorr_old.fits'
    aperture_table = Table.read(table_path)
    #table_path =  '/Users/benjamincollins/University/master/Red_Cardinal/catalogues/aperture_table.csv'
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
    angle_file = '/Users/benjamincollins/University/master/Red_Cardinal/rotation_angles.json'
    with open(angle_file, "r") as f:
        angles = json.load(f)
    angle = angles[f"angle_{survey}{obs}"]
    
    # --- Read WCS from NIRCam image ---
    nircam_path = f"/Users/benjamincollins/University/master/Red_Cardinal/NIRCam/F444W_cutouts/{galaxy_id}_F444W_cutout.fits"
    nircam_data, nircam_wcs = load_cutout(nircam_path)

    # --- Convert NIRCam pixel coordinates to sky ---
    sky_coord = nircam_wcs.pixel_to_world(row['Apr_Xcenter'], row['Apr_Ycenter'])
    
    # --- Open MIRI cutout image ---
    miri_path = f"/Users/benjamincollins/University/master/Red_Cardinal/cutouts_phot/{galaxy_id}_{filter}_cutout_{survey}{obs}.fits"
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
    
    if rescale == True:
        if int(galaxy_id) == 12332:
            scale_factor *= 1.0 # no additional scaling to avoid contaminating source
        elif int(galaxy_id) in [7136, 7904, 7922, 11136, 16419, 21452]:
            scale_factor *= 1.6
        elif int(galaxy_id) in [7934, 10314, 10592, 18332]:
            scale_factor *= 1.8
        else:
            scale_factor *= 2.0
    else:
        scale_factor *= 1.0   # no rescaling, use original aperture sizes
        
    # --- Specify parameters for the ellipse ---
    width = row['Apr_A'] * scale_factor
    height = row['Apr_B'] * scale_factor
    theta = -row['Apr_Theta']
    theta_new = ((theta - angle) % 180) * u.deg
    
    if mask_folder:
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
        
        ax.set_title(f"Galaxy {galaxy_id} - {filter}")
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




def estimate_background(galaxy_id, filter_name, image_data, aperture_params, sigma_val, rescale=True):
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
    
    # Optional block for rescaling apertures such that the annuli are consistent
    # ==========================================    
    if rescale:
        # --- Create elliptical region in MIRI pixel space ---
        nircam_scale = 0.03    # arcsec/pixel
        miri_scale = 0.11092  # arcsec per pixel
        
        # arcsec/pixel
        scale_factor = nircam_scale / miri_scale
        
        if int(galaxy_id) == 12332:
                scale_factor *= 1.0 # no additional scaling to avoid contaminating source
        elif int(galaxy_id) in [7136, 7904, 7922, 11136, 16419, 21452]:
            scale_factor *= 1.6
        elif int(galaxy_id) in [7934, 10314, 10592, 18332]:
            scale_factor *= 1.8
        else:
            scale_factor *= 2.0
    else:
        scale_factor = 1.0
        
    a *= scale_factor
    b *= scale_factor
    # ==========================================
    
    # Define inner and outer radii for the annulus 
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
    vis_dir = '/Users/benjamincollins/University/master/Red_Cardinal/photometry/vis_data_small'
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
        f.attrs['created_date'] = str(np.datetime64('now'))
        f.attrs['data_type'] = 'galaxy_visualisation_data'


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
        #aperture.plot(ax=ax, color='blue', lw=4)
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

            if str(galaxy_id) == '7922':
                
                aperture = EllipticalAperture(
                    positions=(aperture_params['x_center'], aperture_params['y_center']),
                    a=aperture_params['a'],
                    b=aperture_params['b'],
                    theta=aperture_params['theta']
                )

                # Create figure with three subplots in a horizontal row
                fig, axes = plt.subplots(1, 3, figsize=(23, 6))

                # Plot image1 (original data) on the first subplot
                im0 = plot_aperture_overlay(axes[0], image_data, aperture, label='Original Data')

                # Plot image2 (background plane) on the second subplot
                im1 = plot_aperture_overlay(axes[1], background_plane, aperture, label='Background Fit')
                
                # Plot image3 (background-subtracted) on the third subplot
                im2 = plot_aperture_overlay(axes[2], background_subtracted, aperture, label='Background-Subtracted Data')

                # Add a minus and equals sign between the images as an annotation
                fig.text(0.32, 0.45, '$-$', fontsize=30, ha='center', va='center', rotation=0, color='black')
                fig.text(0.66, 0.45, '$=$', fontsize=30, ha='center', va='center', rotation=0, color='black')

                # Add colorbars
                for ax, im, label in zip(axes, [im0, im1, im2], [
                    'Flux [MJy/(sr pixel)]',
                    'Background Flux [MJy/(sr pixel)]',
                    'Background-subtracted Flux [MJy/(sr pixel)]'
                ]):
                    plt.colorbar(im, ax=ax)#, label=label)

                plt.subplots_adjust(wspace=5)  # This will increase the space between the subplots
                
                # Tight layout and saving the figure
                plt.tight_layout()
                plt.subplots_adjust(top=0.85)  # Adjust to prevent overlap with annotation
                #plt.suptitle(f'{filter} - Galaxy ID {galaxy_id}', fontsize=18)
                plt.savefig(os.path.join(plane_sub_dir, f'{galaxy_id}_{filter}.png'), dpi=150)
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

def get_psf(filter_name, psf_dir='/Users/benjamincollins/University/master/Red_Cardinal/WebbPSF/'):
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
    correction_factor = total_flux / flux_in_aperture
    return correction_factor

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
        'image_error': sum_image_errors * conversion_factor,
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
        corrected_image_error = flux_measurements['image_error'] * correction_factor
        
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
            'Image_Err': corrected_image_error,
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
    
    #faulty_ids = [12332, 7136, 7904, 7922, 11136, 16419, 21452, 7934, 10314, 10592, 18332]
    #galaxy_ids = [gid for gid in galaxy_ids if gid not in faulty_ids]
    #print(f"Excluding {len(faulty_ids)} faulty IDs, leaving {len(galaxy_ids)} valid galaxy IDs for photometric analysis")
    
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

    phot_tables_dir = '/Users/benjamincollins/University/master/Red_Cardinal/photometry/phot_tables/'
    os.makedirs(phot_tables_dir, exist_ok=True)
    fits_output = os.path.join(phot_tables_dir, output_file)
    table.write(fits_output, format='fits', overwrite=True)
    print(f"\nSaved FITS table to {fits_output}")

    return table




def galaxy_statistics(table_path, fig_path=None, stats_path=None, detections=None, cols=4):
    """
    Analyse how many galaxies are observed in each filter, and which galaxy IDs appear per filter.

    Parameters:
    -----------
    table_path : str
        Path to the FITS table.
    fig_path : str (optional)
        Path to the output figure.
    stats_path : str (optional)
        Path to the output statistics file.
    detections : dict (optional)
        A dictionary mapping each filter to an array of galaxy IDs that are detected.
    cols : int
        The number of columns that should be displayed.
    
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
            if detections is None or gid in detections[filt]:
                filter_id_map[filt].add(gid)

    # Print summary
    print("\nGalaxy Filter Mapping:")
    print("=" * 30)
    for filt, ids in filter_id_map.items():
        print(f"{filt:10s}: {len(ids)} galaxies")

    if fig_path:
        if detections: title = 'MIRI Detections'
        else: title = 'MIRI Coverage'
        
        plot_galaxy_filter_matrix(table_path, fig_path, None, detections, cols)
        print(f'Saved output plot to {fig_path}')
    if stats_path:
        write_galaxy_stats(table, stats_path, detections)
        print(f'Wrote galaxy statistics to {stats_path}')
    
    return filter_id_map

def write_galaxy_stats(table, output_path, detections):
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
        
        # 3. Write detection statistics
        f.write("\nGalaxy Detection Statistics:\n")
        f.write("-" * 35 + "\n")

        all_ids = set(str(row['ID']) for row in table)
        total_galaxies = len(all_ids)
        detected_in_any = set()

        for filt, det_ids in detections.items():
            # Convert to strings to match ID format
            detected = set(str(i) for i in det_ids)
            detected_in_any.update(detected)

            f.write(f"{filt}: {len(detected)} / {total_galaxies} galaxies ({(len(detected) / total_galaxies) * 100:.1f}%) detected\n")

        f.write(f"\nDetected in at least one filter: {len(detected_in_any)} / {total_galaxies} galaxies ({(len(detected_in_any) / total_galaxies) * 100:.1f}%)\n")


def plot_galaxy_filter_matrix(table_path, fig_path, title=None, detections=None, cols=4):
    """
    Visualise which galaxies are observed and detected in which filters using a binary matrix plot.

    Parameters:
    -----------
    table_path : str
        Path to the FITS file.
    fig_path : str
        Path to the output file.
    title : str, optional
        Title of the plot.
    detections : dict, optional
        Dictionary mapping filter names to lists of galaxy IDs that were detected in that filter.
    """
    table = Table.read(table_path, format='fits')
    
    filter_order = ['F770W', 'F1000W', 'F1800W', 'F2100W']
    pastel_colours = {
        'F770W': '#a6cee3',
        'F1000W': '#b2df8a',
        'F1800W': '#fdbf6f',
        'F2100W': '#fb9a99'
    }

    galaxy_ids = [str(gid) for gid in table['ID']]    
    num_galaxies = len(galaxy_ids)
    chunk_size = (num_galaxies + 3) // cols
    chunks = [galaxy_ids[i:i + chunk_size] for i in range(0, num_galaxies, chunk_size)]

    cell_size = 0.5
    num_cols = len(filter_order)
    num_rows = chunk_size
    fig_width = cell_size * num_cols * cols
    fig_height = cell_size * num_rows * 0.65
    
    fig, axes = plt.subplots(1, cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes[0]
    
    plot_number = 0
    for ax, g_ids in zip(axes, chunks):
        matrix = np.zeros((len(g_ids), len(filter_order)), dtype=int)
        g_index_map = {gid: i for i, gid in enumerate(g_ids)}
        table_id_to_row = {str(row['ID']): idx for idx, row in enumerate(table)}

        for row in table:
            gid = str(row['ID'])
            if gid not in g_index_map:
                continue
            g_idx = g_index_map[gid]
            filters = row['Filters']
            if isinstance(filters, (list, np.ndarray)):
                filters = [f.decode() if isinstance(f, bytes) else str(f) for f in filters]
            else:
                filters = [f.strip() for f in str(filters).split(',') if f.strip()]
            for filt in filters:
                if filt in filter_order:
                    if detections is None or int(gid) in detections.get(filt, []):
                        f_idx = filter_order.index(filt)
                        matrix[g_idx, f_idx] = 1

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 1:
                    base_colour = pastel_colours[filter_order[j]]
                    gid = g_ids[i]
                    row = table[table_id_to_row[gid]]

                    flag_art_array = row['Flag_Art']
                    flag_art = False
                    if flag_art_array is not None and len(flag_art_array) == len(filter_order):
                        flag_art = flag_art_array[j]

                    if flag_art:
                        rgb = np.array(mcolors.to_rgb(base_colour))
                        darker_rgb = np.clip(rgb * 0.7, 0, 1)
                        colour = darker_rgb
                    else:
                        colour = base_colour

                    ax.add_patch(
                        plt.Rectangle((j, i), 1, 1, color=colour)
                    )

        # Add labels with companion asterisk if flagged
        y_labels = []
        for i, gid in enumerate(g_ids):
            row = table[table_id_to_row[gid]]
            label = gid
            if row['Flag_Com'] == True:
                label += '*'
            y_labels.append(label)

        ax.set_xlim(0, len(filter_order))
        ax.set_ylim(len(g_ids), 0)
        ax.set_xticks(np.arange(len(filter_order)) + 0.5)
        ax.set_xticklabels(filter_order, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(g_ids)) + 0.5)
        ax.set_yticklabels(y_labels, fontsize=8)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.show()


def compare_aperture_statistics(table_small_path, table_big_path, fig_path=None, summary_doc_path=None, scaling=None):
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
    import seaborn as sns

    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")

    table_small = Table.read(table_small_path)
    table_big = Table.read(table_big_path)

    # Convert ID columns to string for alignment
    ids_small = [id.decode() if isinstance(id, bytes) else str(id) for id in table_small['ID']]
    ids_big = [id.decode() if isinstance(id, bytes) else str(id) for id in table_big['ID']]

    # Match common IDs
    common_ids = sorted(set(ids_small) & set(ids_big))
    print(f"Found {len(common_ids)} common galaxies")

    # Prepare data structures
    bands = ['F770W', 'F1000W', 'F1800W', 'F2100W']
    data_comparison = {
        'ID': [],
        'Band': [],
        'Flux_Small_Raw': [],
        'Flux_Big_Raw': [],
        'Flux_Err_Small_Raw': [],
        'Flux_Big_Raw_Err': [],
        'Flux_Small_Corrected': [],
        'Flux_Big_Corrected': [],
        'Flux_Err_Small_Corrected': [],
        'Flux_Big_Corrected_Err': [],
        'Apr_Corr_Small': [],
        'Apr_Corr_Big': [],
        'Flux_Ratio': [],
        'Corrected_Flux_Ratio': [],
        'Flux_Difference': [],
        'Corrected_Flux_Difference': []
    }

    # Collect all data for comprehensive analysis
    for idx, band in enumerate(bands):  # bands = ["F770W", "F1000W", "F1800W", "F2100W"]

        for gid in common_ids:
            index_s = ids_small.index(gid)
            index_b = ids_big.index(gid)

            # Raw fluxes (convert to µJy)
            flux_small = table_small['Flux'][index_s][idx] * 1e6
            flux_big = table_big['Flux'][index_b][idx] * 1e6
            flux_err_small = table_small['Flux_Err'][index_s][idx] * 1e6
            flux_err_big = table_big['Flux_Err'][index_b][idx] * 1e6
            
            # Aperture corrections
            corr_small = table_small['Apr_Corr'][index_s][idx] if 'Apr_Corr' in table_small.colnames else np.nan
            corr_big = table_big['Apr_Corr'][index_b][idx] if 'Apr_Corr' in table_big.colnames else np.nan
            
            # Skip if any crucial value is invalid
            if not (np.isfinite(flux_small) and np.isfinite(flux_big) and
                    (flux_small > 0) and (flux_big > 0) and
                    np.isfinite(flux_err_small) and np.isfinite(flux_err_big) and
                    np.isfinite(corr_small) and np.isfinite(corr_big)):
                continue
                
            # Calculate corrected fluxes
            flux_small_corr = flux_small * corr_small
            flux_big_corr = flux_big * corr_big
            flux_err_small_corr = flux_err_small * corr_small
            flux_err_big_corr = flux_err_big * corr_big
            
            # Store all data
            data_comparison['ID'].append(gid)
            data_comparison['Band'].append(band)            
            data_comparison['Flux_Small_Raw'].append(flux_small)
            data_comparison['Flux_Big_Raw'].append(flux_big)
            data_comparison['Flux_Err_Small_Raw'].append(flux_err_small)
            data_comparison['Flux_Big_Raw_Err'].append(flux_err_big)
            data_comparison['Flux_Small_Corrected'].append(flux_small_corr)
            data_comparison['Flux_Big_Corrected'].append(flux_big_corr)
            data_comparison['Flux_Err_Small_Corrected'].append(flux_err_small_corr)
            data_comparison['Flux_Big_Corrected_Err'].append(flux_err_big_corr)
            data_comparison['Apr_Corr_Small'].append(corr_small)
            data_comparison['Apr_Corr_Big'].append(corr_big)
            data_comparison['Flux_Ratio'].append(flux_big / flux_small)
            data_comparison['Corrected_Flux_Ratio'].append(flux_big_corr / flux_small_corr)
            data_comparison['Flux_Difference'].append(flux_big - flux_small)
            data_comparison['Corrected_Flux_Difference'].append(flux_big_corr - flux_small_corr)

    filename = os.path.join("/Users/benjamincollins/University/Master/Red_Cardinal/photometry/apertures/aperture_comparisons/comparison_data.pkl")    
    
    # Write output to a pickle file
    with open(filename, 'wb') as f:
        pkl.dump(data_comparison, f)
        print(f'Saved pickle file to {filename}')
        
    if fig_path:
        plot_aperture_comparison(data_comparison, fig_path, scaling)
        print(f'Saved output plot to {fig_path}')
        
    if summary_doc_path:
        write_aperture_summary(data_comparison, common_ids, summary_doc_path)


def plot_aperture_comparison(data_comparison, fig_path, scaling=None):
    
    # Convert to arrays for easier handling
    for key in data_comparison:
        data_comparison[key] = np.array(data_comparison[key])

    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(20, 16))

    bands = ['F770W', 'F1000W', 'F1800W', 'F2100W']
    
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
    
def plot_aperture_summary(data_comparison, scaling=False):
    
    fig_path = '/Users/benjamincollins/University/Master/Red_Cardinal/photometry/apertures/aperture_comparisons/'
    
    # Convert to arrays
    for key in data_comparison:
        data_comparison[key] = np.array(data_comparison[key])

    bands = ['F770W', 'F1800W']
    colors = ['#1f77b4', '#ff7f0e']  # Distinct colors per band
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, band in enumerate(bands):
        mask = data_comparison['Band'] == band
        
        # --- (1) Corrected Flux Scatter ---
        ax = axes[i, 0]
        ax.scatter(data_comparison['Flux_Small_Corrected'][mask],
                   data_comparison['Flux_Big_Corrected'][mask],
                   alpha=0.7, s=30, color=colors[i])
        
        # 1:1 line
        min_flux = min(np.min(data_comparison['Flux_Small_Corrected'][mask]),
                       np.min(data_comparison['Flux_Big_Corrected'][mask]))
        max_flux = max(np.max(data_comparison['Flux_Small_Corrected'][mask]),
                       np.max(data_comparison['Flux_Big_Corrected'][mask]))
        ax.plot([min_flux, max_flux], [min_flux, max_flux], 'k--', alpha=0.8, label='1:1')
        if scaling: ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(f'Corrected Flux small [µJy]')
        ax.set_ylabel(f'Corrected Flux large [µJy]')
        ax.set_title(f'{band} Corrected Flux Comparison')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Calculate and display R² correlation
        corr_diff = data_comparison['Corrected_Flux_Difference'][mask]
        flux_small = data_comparison['Flux_Small_Corrected'][mask]
        frac_diff = corr_diff / flux_small
        #ax.text(0.05, 0.9, f"median = {np.median(frac_diff):.3f}" + "\n" + rf"$\sigma$ = {np.std(frac_diff):.3f}", 
         #       transform=ax.transAxes, fontsize=10,
          #      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        

        # --- (2) Corrected Flux Ratio Histogram ---
        ax = axes[i, 1]

        # Convert masked data to a NumPy array
        ratios = np.array(data_comparison['Corrected_Flux_Ratio'][mask])

        # Define histogram cutoff at 95th percentile
        cutoff = np.percentile(ratios, 95)
        ratios_clipped = ratios[ratios <= cutoff]

        #if i == 1: ratios = ratios[ratios < 3.0]
        
        # Plot histogram
        ax.hist(ratios_clipped, bins=25,
                alpha=0.7, color=colors[i], edgecolor='black',
                range=(0, 3.0))

        # Reference lines
        ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(np.median(ratios), color='darkred', linestyle='-', linewidth=1.5)

        # Add compact statistics
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        num = f'N = {len(ratios_clipped)} (95th pct of {len(ratios)})'
        median_ratio = np.median(ratios)
        
        stats_text = f'μ={mean_ratio:.2f}\nσ={std_ratio:.2f}\nMed={median_ratio:.2f}\n\n{num}'
            
        ax.text(0.65, 0.77, stats_text, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Annotate number of sources
        #ax.text(0.95, 0.95,
        #        f"N = {len(ratios_clipped)} (95th pct of {len(ratios)})",
        #        ha='right', va='top',
        #        transform=ax.transAxes,
        #        fontsize=10,
        #        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Labels and formatting
        ax.set_xlabel('Corrected Flux Ratio (Large/Small)')
        ax.set_ylabel('Number of Sources')
        ax.set_title(f'{band} Corrected Flux Ratio Distribution')
        ax.legend()
        ax.grid(alpha=0.3)


    #plt.suptitle('Aperture Photometry Comparison: Short vs Long λ', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    figname = os.path.join(fig_path, 'new_stats_thesis_v2.png')
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'Saved summary plot to {figname}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for i, band in enumerate(bands):
        mask = data_comparison['Band'] == band
        ax = axes[i]

        sc = ax.scatter(
            data_comparison['Flux_Small_Corrected'][mask],
            data_comparison['Corrected_Flux_Ratio'][mask],
            alpha=0.6, s=30,
            c=data_comparison['Apr_Corr_Big'][mask],
            cmap='viridis'
        )

        # Proper colorbar
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Large Aperture Correction')

        ax.axhline(1.0, color='red', linestyle='--', alpha=0.8, label='Unity')
        if scaling:
            ax.set_xscale('log')
        ax.set_ylim(0,7.5)
        ax.set_xlabel(f'{band} Small Aperture Corrected Flux [µJy]')
        ax.set_ylabel('Flux Ratio (Large/Small)')
        ax.set_title(f'{band} Flux Ratio vs Brightness')
        ax.legend()
        ax.grid(True, alpha=0.3)

    figname = os.path.join(fig_path, 'ratio_vs_brightness.png')
    plt.savefig(figname, bbox_inches="tight", pad_inches=0)
    plt.tight_layout()
    plt.show()
    print(f'Saved scatter plot to {figname}')
    
    

def write_aperture_summary(data_comparison, common_ids, summary_doc_path):
    """
    Write a comprehensive aperture comparison summary to a text file.
    Includes raw ratios, aperture corrections, corrected differences,
    fractional differences, and context relative to measurement errors.
    """

    doc_path = os.path.join('/Users/benjamincollins/University/Master/Red_Cardinal/photometry/apertures/aperture_comparisons', summary_doc_path)
    
    df = pd.DataFrame(data_comparison)
    print("Bands in table:", np.unique(data_comparison['Band']))
    
    bands = ['F770W', 'F1000W', 'F1800W', 'F2100W']

    with open(doc_path, "w") as file:
        file.write("\n" + "="*80 + "\n")
        file.write("COMPREHENSIVE APERTURE COMPARISON SUMMARY\n")
        file.write("="*80 + "\n")

        for band in bands:
            
            band_data = df[df['Band'] == band]

            file.write(f"\n{band} FILTER:\n")
            file.write("-" * 40 + "\n")

            # --- Raw flux ratios ---
            flux_ratio = band_data['Flux_Ratio']
            file.write("Raw Flux Ratios (Large/Small):\n")
            file.write(f"  Median: {np.median(flux_ratio):.3f} ± {np.std(flux_ratio):.3f}\n")
            file.write(f"  Mean:   {np.mean(flux_ratio):.3f}\n")
            file.write(f"  Range:  {np.min(flux_ratio):.3f} – {np.max(flux_ratio):.3f}\n")

            # --- Corrected flux ratios ---
            corr_ratio = band_data['Corrected_Flux_Ratio']
            file.write("\nCorrected Flux Ratios (Large/Small):\n")
            file.write(f"  Median: {np.median(corr_ratio):.3f} ± {np.std(corr_ratio):.3f}\n")
            file.write(f"  Mean:   {np.mean(corr_ratio):.3f}\n")
            file.write(f"  Range:  {np.min(corr_ratio):.3f} – {np.max(corr_ratio):.3f}\n")
            
            # --- Calculate bias reduction ---
            raw_bias = np.median(flux_ratio) - 1.0
            corrected_bias = np.median(corr_ratio) - 1.0
            bias_reduction = (raw_bias - corrected_bias) / raw_bias * 100

            file.write(f"\nBias Reduction:\n")
            file.write(f"  Initial systematic bias: {raw_bias*100:.1f}%\n")
            file.write(f"  Residual systematic bias: {corrected_bias*100:.1f}%\n")
            file.write(f"  Bias reduction achieved: {bias_reduction:.1f}%\n")

            # --- Aperture corrections ---
            file.write("\nAperture Corrections:\n")
            small_corr_med = np.median(band_data['Apr_Corr_Small'])
            big_corr_med   = np.median(band_data['Apr_Corr_Big'])
            small_sigma_corr = np.std(band_data['Apr_Corr_Small'])
            big_sigma_corr = np.std(band_data['Apr_Corr_Big'])
            file.write(f"  Small aperture median: {small_corr_med:.3f}\n")
            file.write(f"  Large aperture median: {big_corr_med:.3f}\n")
            file.write(f"  Difference (Large–Small): {big_corr_med - small_corr_med:.3f}\n")

            # --- Final corrected flux differences ---
            corr_diff = band_data['Corrected_Flux_Difference']
            file.write("\nFinal Corrected Flux Differences (Large – Small) [µJy]:\n")
            file.write(f"  Median: {np.median(corr_diff):.2f} ± {np.std(corr_diff):.2f}\n")
            file.write(f"  Mean:   {np.mean(corr_diff):.2f}\n")
            higher_flux_pct = np.sum(corr_diff > 0) / len(corr_diff) * 100
            file.write(f"  Sources with higher flux in large aperture: {higher_flux_pct:.1f}%\n")

            # --- Fractional differences (preferred) ---
            frac_diff = corr_diff / band_data['Flux_Small_Corrected']  # ΔFlux / Flux_small
            frac_diff_pct = frac_diff * 100
            file.write("\nFractional Differences ((Large – Small)/Small):\n")
            file.write(f"  Median: {np.median(frac_diff_pct):.1f}% ± {np.std(frac_diff_pct):.1f}%\n")
            file.write(f"  Range (5–95th pct): {np.percentile(frac_diff_pct,5):.1f}% – {np.percentile(frac_diff_pct,95):.1f}%\n")

            # --- Compare to uncertainties ---
            flux_small = band_data['Flux_Small_Corrected']
            flux_err_small = np.median(band_data['Flux_Err_Small_Corrected'])
            std_flux_err_small = np.std(band_data['Flux_Err_Small_Corrected'])               

            # Propagated correction-induced uncertainty for each source
            corr_err = np.sqrt((small_corr_med * flux_err_small)**2 +
                            (flux_small * small_sigma_corr)**2)

            # Median difference between large and small aperture corrected fluxes
            median_corr_diff = np.median(band_data['Corrected_Flux_Difference'])

            file.write("\nContext vs. Measurement Errors:\n")
            file.write(f"  Median corrected aperture difference: {median_corr_diff:.2f} µJy\n")
            file.write(f"  Median flux uncertainty: {flux_err_small:.2f} ± {std_flux_err_small:.2f} µJy\n")
            file.write(f"  Median propagated uncertainty incl. correction: {np.median(corr_err):.2f} µJy\n")

            if abs(median_corr_diff) < flux_err_small:
                file.write("  → Aperture differences are smaller than typical measurement errors.\n")
            else:
                file.write("  → Aperture differences are comparable to or larger than typical measurement errors.\n")

            # --- Compact interpretation line ---
            file.write("\nSummary:\n")
            file.write(f"  Corrected photometry converges across apertures, with residuals ≲{np.median(np.abs(frac_diff_pct)):.1f}%.\n")

        # Final count
        file.write(f"\nTotal sources analysed: {len(common_ids)}\n")
        file.write("="*80 + "\n")
        print('Wrote summary document to', doc_path)





def plot_appendix_figure(data_comparison, fig_path, scaling=None):
    """
    Create a compact summary plot for aperture photometry comparison.
    Layout: 4 rows (one per band) × 3 columns
    - Column 1: Corrected flux comparison (scatter)
    - Column 2: Corrected flux ratio distribution (histogram)
    - Column 3: Flux ratio vs brightness (scatter)
    """
    
    # Convert to arrays for easier handling
    for key in data_comparison:
        data_comparison[key] = np.array(data_comparison[key])

    # Create summary figure optimized for A4 appendix
    fig = plt.figure(figsize=(12, 16))  # Good aspect ratio for A4
    
    bands = ['F770W', 'F1000W', 'F1800W', 'F2100W']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # Distinct colors per band
    
    for i, band in enumerate(bands):
        mask = data_comparison['Band'] == band
        band_color = colors[i]
        
        # Column 1: Corrected flux comparison
        ax1 = plt.subplot(4, 3, i*3 + 1)
        plt.scatter(data_comparison['Flux_Small_Corrected'][mask], 
                   data_comparison['Flux_Big_Corrected'][mask], 
                   alpha=0.6, s=20, color=band_color)
        
        # Add 1:1 line
        min_flux = min(np.min(data_comparison['Flux_Small_Corrected'][mask]), 
                      np.min(data_comparison['Flux_Big_Corrected'][mask]))
        max_flux = max(np.max(data_comparison['Flux_Small_Corrected'][mask]), 
                      np.max(data_comparison['Flux_Big_Corrected'][mask]))
        plt.plot([min_flux, max_flux], [min_flux, max_flux], 'k--', alpha=0.7, linewidth=1)
        
        if scaling: plt.loglog()
        plt.xlabel('Small Aperture [µJy]')
        plt.ylabel('Large Aperture [µJy]')
        plt.title(f'{band} Corrected Flux', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Calculate and display R² correlation
        corr_coef = np.corrcoef(data_comparison['Flux_Small_Corrected'][mask], 
                               data_comparison['Flux_Big_Corrected'][mask])[0, 1]
        r_squared = corr_coef**2
        plt.text(0.05, 0.9, f'R² = {r_squared:.3f}', 
                transform=ax1.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Column 2: Corrected flux ratio distribution
        ax2 = plt.subplot(4, 3, i*3 + 2)
        n, bins, patches = plt.hist(data_comparison['Corrected_Flux_Ratio'][mask], 
                                   bins=25, alpha=0.7, color=band_color, 
                                   edgecolor='black', linewidth=0.5, range=(0,4))
        
        # Add vertical lines for key statistics
        median_ratio = np.median(data_comparison['Corrected_Flux_Ratio'][mask])
        plt.axvline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        plt.axvline(median_ratio, color='darkred', linestyle='-', linewidth=1.5)
        
        plt.xlabel('Flux Ratio (Large/Small)')
        plt.ylabel('N Sources')
        plt.title(f'{band} Ratio Distribution', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add compact statistics
        mean_ratio = np.mean(data_comparison['Corrected_Flux_Ratio'][mask])
        std_ratio = np.std(data_comparison['Corrected_Flux_Ratio'][mask])
        remark = '2 outliers > 4.0'
        num = f'N = {len(data_comparison['Corrected_Flux_Ratio'][mask])}'
        
        if i == len(bands) - 2:
            stats_text = f'μ={mean_ratio:.2f}\nσ={std_ratio:.2f}\nMed={median_ratio:.2f}\n\n{num}\n{remark}'
        else:
            stats_text = f'μ={mean_ratio:.2f}\nσ={std_ratio:.2f}\nMed={median_ratio:.2f}\n\n{num}'
            
        plt.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right')
                #bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Column 3: Flux ratio vs brightness
        ax3 = plt.subplot(4, 3, i*3 + 3)
        scatter = plt.scatter(data_comparison['Flux_Small_Corrected'][mask], 
                            data_comparison['Corrected_Flux_Ratio'][mask], 
                            alpha=0.6, s=15, c=data_comparison['Apr_Corr_Small'][mask], 
                            cmap='viridis', vmin=0.8, vmax=3.0)
        
        plt.axhline(1.0, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        plt.axhline(median_ratio, color='darkred', linestyle='-', alpha=0.8, linewidth=1)
        
        if scaling: plt.xscale('log')
        plt.xlabel('Small Aperture Flux [µJy]')
        plt.ylabel('Flux Ratio')
        plt.title(f'{band} Ratio vs Brightness', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 4)
        
        # Add colorbar only for the last band to save space
        if i == len(bands) - 1:
            cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8)
            cbar.set_label('Aperture Correction', fontsize=9)

    # Overall title and layout adjustment
    #plt.suptitle('Aperture Photometry Summary: Small vs Large Aperture Comparison', 
                #fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.35, wspace=0.3)
        
    # Save with high DPI for appendix quality
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    

def analyse_outliers(data_comparison, flags,
                             ratio_col='Corrected_Flux_Ratio', 
                             threshold=3.0, 
                             summary=True):
    """
    Identify outliers where aperture corrections produced unphysical flux ratios.
    
    Parameters
    ----------
    data_comparison : dict, pd.DataFrame, or astropy Table
        Must contain columns: 'ID', 'Band', and ratio_col (default: 'Corrected_Flux_Ratio').
    ratio_col : str
        Column name for corrected flux ratios to check (Large/Small).
    threshold : float
        Absolute deviation from unity considered "bad".
        Example: threshold=3 → flags ratios < 1/3 or > 3.
    summary : bool
        If True, prints summary counts per filter.
    
    Returns
    -------
    outliers : pd.DataFrame
        Subset of rows that are outliers.
    """
    
    from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
            
    # Convert to DataFrame if needed
    if not isinstance(data_comparison, pd.DataFrame):
        data = pd.DataFrame(data_comparison)
    else:
        data = data_comparison.copy()
    
    # Boolean mask: ratios too extreme
    ratios = data[ratio_col].astype(float)
    bad_mask = (ratios < 1/threshold) | (ratios > threshold) | ~np.isfinite(ratios)
    
    outliers = data[bad_mask]
    
    if summary:
        print("=== Outlier Summary ===")
        for band in outliers['Band'].unique():
            count = np.sum(outliers['Band'] == band)
            print(f"{band}: {count} flagged outliers")
        print(f"Total outliers: {len(outliers)} / {len(data)} ({100*len(outliers)/len(data):.1f}%)")
    
    for obj in outliers.to_dict(orient='records'):
        objid = obj['ID']
        band = obj['Band']
        ratio = obj[ratio_col]
        
        if int(objid) in flags.get(band, []):
            print(f"⚠️ {objid} in {band} - known nondetection")
        else:
            print("This counts as a detection (apparently)")
        
        try:
            vis_data = load_vis(f"/Users/benjamincollins/University/Master/Red_Cardinal/photometry/vis_data/{objid}_{band}.h5")
        except:
            print(f"❌ No VIS data found for {objid}")
            continue
        
        img = vis_data["background_subtracted"]

        # Normalisation: auto scale + asinh stretch
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(img)
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
        
        plt.figure(figsize=(4, 4))
        plt.imshow(img, origin="lower", cmap="inferno", norm=norm)
        plt.colorbar(label="Flux")
        plt.title(f"Galaxy {objid} - {band}\nRatio = {ratio:.2f}")
        plt.tight_layout()
        plt.show()

    return outliers



def aperture_flux_at(img, aperture_params):
    # aperture shape (scale only, will move centres + vary theta)
    a = aperture_params['a']
    b = aperture_params['b']
    theta_ref = aperture_params['theta']
    x0, y0 = aperture_params['x_center'], aperture_params['y_center']

    aperture = EllipticalAperture((x0, y0), a, b, theta_ref)

    phot = aperture_photometry(img, aperture, method='exact')
    return phot['aperture_sum'][0]

def empirical_aperture_rms(img, aperture_params, n_random=200):
    """
    Estimate RMS by placing random elliptical apertures on the image.
    
    Parameters
    ----------
    img : 2D array
        Background-subtracted + masked cutout image.
    aperture_params : dict
        Dictionary with keys ['a', 'b', 'theta', 'x_center', 'y_center'].
    n_random : int
        Number of random apertures to place.
    
    Returns
    -------
    rms : float
        Empirical RMS of aperture fluxes.
    """
    ny, nx = img.shape
    aperturesums = []
    attempts = 0
    max_attempts = n_random * 20
    
    # aperture shape (scale only, will move centres + vary theta)
    a = aperture_params['a']
    b = aperture_params['b']
    theta_ref = aperture_params['theta']
    x0, y0 = aperture_params['x_center'], aperture_params['y_center']

    while len(aperturesums) < n_random and attempts < max_attempts:
        attempts += 1
        
        # random centre inside image (avoid edges)
        x = random.uniform(a+2, nx - a - 2)
        y = random.uniform(b+2, ny - b - 2)

        # skip if centre falls on masked pixel
        if np.isnan(img[int(y), int(x)]):       # NaN is the only value not similar to itself! Important!
            continue

        # random angle variation (around reference theta)
        theta = random.uniform(0, 2*np.pi)

        aperture = EllipticalAperture((x, y), a, b, theta)
        phot_table = aperture_photometry(img, aperture, method="exact")
        flux = phot_table['aperture_sum'][0]

        if np.isfinite(flux):
            aperturesums.append(flux)

    if len(aperturesums) < max(10, n_random // 4):
        # fallback: pixel rms scaled to aperture area
        print("⚠️ Too few valid random apertures, using pixel RMS fallback")
        pixrms = np.nanstd(img)
        area = np.pi * a * b
        return pixrms * np.sqrt(area)

    return np.std(aperturesums, ddof=1)


def recompute_empirical_snr(vis_data, n_random=200):
    """
    Compute flux & empirical S/N at source centre using background-subtracted image
    with combined mask applied.
    """
    
    galaxy_id = vis_data["galaxy_id"]
    print(f"Recomputing empirical S/N for {galaxy_id}...")
    
    img = vis_data["background_subtracted"]
    
    bkg_mask = vis_data["segmentation_mask"] | np.isnan(vis_data["original_data"])
    clean_image = np.where(bkg_mask, np.nan, img)
    
    
    combined_mask = vis_data["source_mask"] | bkg_mask
    very_clean_image = np.where(combined_mask, np.nan, img)
    
    # Load aperture used for photometry
    aperture_params = vis_data['aperture_params']
    x_center = aperture_params['x_center']
    y_center = aperture_params['y_center']
    a = aperture_params['a']
    b = aperture_params['b']
    theta = aperture_params['theta']
    
    
    ny, nx = img.shape
    centre = (x_center, y_center)

    flux = aperture_flux_at(clean_image, aperture_params)
    emp_rms = empirical_aperture_rms(very_clean_image, aperture_params = aperture_params, n_random=n_random)
    sn = flux / emp_rms if emp_rms > 0 else 0.0

    return dict(
        objid=vis_data["galaxy_id"],
        flux=flux,
        flux_err=emp_rms,
        sn=sn,
        centre=centre
    )

def stack_cutouts(fits_paths, hdu_index=1, method='median'):
    imgs = []
    for p in fits_paths:
        with fits.open(p) as hdul:
            imgs.append(hdul[hdu_index].data.astype(float))
    arr = np.stack(imgs, axis=0)
    if method == 'median':
        return np.nanmedian(arr, axis=0)
    return np.nanmean(arr, axis=0)
