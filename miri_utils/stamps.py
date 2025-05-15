#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIRI Utils Colour Image Processing Module
=========================================

This module provides utilities for creating colour-composite images from FITS data
obtained from JWST MIRI and NIRCam instruments. It includes functions for resampling,
normalising, and combining image data to create scientifically informative and
visually appealing colour images.

Functions
---------
    - resample_nircam: Resamples NIRCam images to a specified pixel size
    - normalise_image: Applies various stretches to normalise image data
    - preprocess_fits_image: Loads and preprocesses FITS images with background handling
    - make_stamp: Creates RGB composite images from multiple FITS files

Example usage
-------------
    from miri_utils.color_utils import resample_nircam, make_stamp

    # Resample all NIRCam files in directory
    resample_nircam("./data", 1024)

    # Create color composite
    image_dict = {
        'R': ['image_F1800W.fits[0]'],
        'G': ['image_F770W.fits[0]'],
        'B': ['image_F444W.fits[0]']
    }
    make_stamp(image_dict, 10, 0.05, 1.0, 10, 0.05, 1.0, 10, 0.05, 1.0, 
            stretch='asinh', outfile='my_color_image.pdf')

Author: Benjamin P. Collins
Date: May 15, 2025
Version: 1.0
"""

import os
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

def resample_nircam(indir, num_pixels):
    """
    Resamples NIRCam FITS images to a specified square pixel dimension.
    
    This function searches for NIRCam FITS files in the provided directory,
    resamples them to the specified number of pixels (square), and updates
    the WCS information accordingly to maintain astrometric accuracy.
    
    Parameters:
    -----------
    indir : str
        Directory containing NIRCam FITS files to process
    num_pixels : int
        Target image size in pixels (will create num_pixels × num_pixels images)
        
    Returns:
    --------
    None
        Writes resampled images to disk with '_res.fits' suffix
    """
    # Open the FITS files
    fits_files = glob.glob(os.path.join(indir, '*nircam.fits'))
    for file_path in fits_files:
        with fits.open(file_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            wcs = WCS(header)
            
            # Resample the image
            target_shape = (num_pixels, num_pixels)
            ny_old, nx_old = data.shape
            ny_new, nx_new = target_shape
            zoom_y = ny_new / ny_old
            zoom_x = nx_new / nx_old
            resampled_data = zoom(data, (zoom_y, zoom_x), order=1, mode='nearest')
            
            # Update WCS information
            header['NAXIS1'] = nx_new
            header['NAXIS2'] = ny_new
            if 'CDELT1' in header and 'CDELT2' in header:
                header['CDELT1'] /= zoom_x
                header['CDELT2'] /= zoom_y
            if 'CD1_1' in header and 'CD2_2' in header:
                header['CD1_1'] /= zoom_x
                header['CD2_2'] /= zoom_y
                
            # Write new FITS file to the same directory
            out_path = file_path.replace('.fits', '_res.fits')
            fits.writeto(out_path, resampled_data, header, overwrite=True)
            print(f"Saved resampled file to: {out_path}")

def normalise_image(img, stretch='asinh', Q=10, alpha=1, weight=1.0):
    """
    Normalises the input image with optional stretching and channel weighting.
    
    This function applies various stretching functions to enhance image contrast
    and normalize pixel values for optimal visualization. The asinh stretch is
    particularly useful for astronomical data with high dynamic range.
    
    Parameters:
    -----------
    img : 2D numpy array
        Input image data
    stretch : str, optional
        Type of stretch function ('asinh', 'log', or 'linear')
    Q : float, optional
        Controls asinh stretch strength (higher values = more linear at bright end)
    alpha : float, optional
        Controls non-linearity for asinh stretch (scaling factor before stretch)
    weight : float, optional
        Multiplier to boost/dampen this channel's contribution
        
    Returns:
    --------
    numpy.ndarray
        Normalised image scaled between 0 and 1
    """
    # Replace nans, positive and negative infinities with 0.0
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip possible negative fluxes
    img = np.clip(img, 0, None)
    #print(f"Before any scaling: min={np.min(img)}, max={np.max(img)}")
    
    # Subtract the minimum value to shift the range to start from 0
    img_min = np.min(img)
    img -= img_min
    #print(f"After minimum subtraction: min={np.min(img)}, max={np.max(img)}")
    #print(f"After weight scaling: min={np.min(img)}, max={np.max(img)}")
    
    # Determine which scaling to use
    if stretch == 'asinh': # Lupton scaling
        img_scaled = np.arcsinh(alpha * Q * img) / Q
    elif stretch == 'log':
        img_scaled = np.log10(1 + alpha * img)
    elif stretch == 'linear':
        img_scaled = img
    else:
        raise ValueError("Unknown stretch")
    #print(f"After stretch: min={np.min(img_scaled)}, max={np.max(img_scaled)}")
    
    # After stretching the image is normalised to 1
    img_scaled = img_scaled / np.nanmax(img_scaled) if np.nanmax(img_scaled) != 0 else img_scaled
    #print(f"After normalisation: min={np.min(img_scaled)}, max={np.max(img_scaled)}")
    
    return img_scaled

def preprocess_fits_image(filename, ext=0, stretch='asinh', Q=10, alpha=1, weight=1, normalise=False):
    """
    Load, process and normalize a FITS image for display purposes.
    
    This function handles loading FITS data, background estimation,
    stretch application, and normalization to prepare astronomical 
    images for visualization.
    
    Parameters:
    -----------
    filename : str
        FITS filename with optional extension (e.g., 'file.fits[1]')
    ext : int, optional
        FITS extension to use (default: 0)
    stretch : str, optional
        Type of stretch ('asinh' or 'linear')
    Q : float, optional
        Controls asinh stretch strength
    alpha : float, optional
        Controls non-linearity for asinh
    weight : float, optional
        Multiplier to boost/dampen this channel's contribution
    normalise : bool, optional
        Whether to normalize the image to [0,1] range
        
    Returns:
    --------
    numpy.ndarray
        Processed 2D numpy image
    """
    try:
        with fits.open(filename) as hdul:
            img = hdul[ext].data.astype(float)
    except Exception as e:
        raise RuntimeError(f"Could not open {filename}[{ext}]: {e}")
    
    # Remove NaNs and negative values for display purposes
    flat = img.flatten()
    flat = flat[np.isfinite(flat)] # remove NaNs/Infs
    
    # Sort pixel values
    sorted_pixels = np.sort(flat)
    
    # Get the lower X% of the pixels for background estimation
    cutoff_index = int(len(sorted_pixels) * 0.8)
    faint_pixels = sorted_pixels[:cutoff_index]
    
    # Compute robust mean (using two methods - the second is currently used)
    background_mean = np.mean(faint_pixels)  # Statistical background from faint pixels
    background_mean = np.nanmean(img)        # Simple mean ignoring NaNs
    
    # Replace NaN values with background
    img = np.where(np.isnan(img), background_mean, img)
    
    # Shift minimum to zero
    img -= np.min(img)
    
    # Ensure all values are non-negative before stretch
    img[img < 0] = 0.0
    
    # Apply stretch
    if stretch == 'asinh':
        img = np.arcsinh(Q * alpha * img) / Q
    elif stretch == 'linear':
        pass # No stretch applied
    else:
        raise ValueError(f"Unsupported stretch: {stretch}")
    
    # Normalise if requested
    if normalise:
        max_val = np.nanmax(img)
        if max_val > 0:
            img /= max_val
    
    return img

def make_stamp(imagesRGB, Q_r, alpha_r, weight_r, Q_g, alpha_g, weight_g, Q_b, alpha_b, weight_b=1.0, stretch='asinh', outfile='stamp.pdf'):
    """
    Create RGB composite images from multiple FITS files with customizable
    stretch parameters for each color channel.
    
    This function processes multiple FITS images to create both individual grayscale
    representations of each filter and a composite RGB image. It handles cases where
    all three channels (R,G,B) are available or when only R and B are available.
    
    Parameters:
    -----------
    imagesRGB : dict
        Dictionary with keys 'R', 'G', 'B' containing lists of FITS filenames
        for each color channel (e.g., {'R': ['file1.fits[0]'], 'G': ['file2.fits[0]']})
    Q_r, Q_g, Q_b : float
        Q parameter for asinh stretch for R, G, B channels
    alpha_r, alpha_g, alpha_b : float
        Alpha parameter for asinh stretch for R, G, B channels
    weight_r, weight_g, weight_b : float
        Weight multipliers for R, G, B channels
    stretch : str, optional
        Type of stretch to apply ('asinh' or 'linear')
    outfile : str, optional
        Output filename for the RGB composite image
        
    Returns:
    --------
    None
        Saves composite RGB image to outfile and filter panels to outfile_filters.pdf
    """
    # Parameter dictionary to handle values per channel
    params = {
        'R': {'Q': Q_r, 'alpha': alpha_r, 'weight': weight_r},
        'G': {'Q': Q_g, 'alpha': alpha_g, 'weight': weight_g},
        'B': {'Q': Q_b, 'alpha': alpha_b, 'weight': weight_b}
    }
    
    stretched_images = {}
    global_max = 0.0
    
    # --- Section for normalising the images ---
    # Stretch each image and find the global max
    for colour in ['R', 'G', 'B']:
        image_str = imagesRGB[colour][0]
        fname, ext = image_str.split('[')
        ext = int(ext.replace(']', ''))
        colour_params = params[colour]
        
        # Stretch but don't normalise yet
        stretched = preprocess_fits_image(
            fname,
            ext,
            stretch=stretch,
            Q=colour_params['Q'],
            alpha=colour_params['alpha'],
            normalise=False  # Postpone normalization to maintain relative brightness
        )
        stretched_images[colour] = stretched
        max_val = np.nanmax(stretched)
        if max_val > global_max:
            global_max = max_val
    
    # Now normalise all images to global max for proper relative brightness
    norm_images = {}
    for colour in ['B', 'G', 'R']:
        norm_images[colour] = stretched_images[colour] / global_max
        norm_images[colour] = np.clip(norm_images[colour], 0, 1)
    
    # Add scaled images to the dictionary for temporary storage
    temp_files = {}
    for colour in norm_images:
        fname = f'temp_{colour}.fits'
        fits.writeto(fname, norm_images[colour], overwrite=True)
        temp_files[colour] = fname
    
    # Final processing before display
    for colour in norm_images:
        # Replace remaining NaNs just before stacking
        norm_images[colour] = np.nan_to_num(norm_images[colour], nan=0.0)
        norm_images[colour] = np.clip(norm_images[colour], 0, 1)
    
    # --- Section to check whether all filters are available for RGB mapping ---
    if 'fake' in imagesRGB['G'][0]: # only R and B available
        # Create a horizontal panel with 2 grayscale subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # 2 images side-by-side
        
        # Map colour channels to filters for labelling
        channel_labels = {'R': 'F770W', 'B': 'F444W'}
        
        # Legend info
        legend_text = f"R: {os.path.basename(imagesRGB['R'][0]).split('_')[1]}\n" \
                      f"B: {os.path.basename(imagesRGB['B'][0]).split('_')[1]}"
        
        for ax, colour in zip(axes, ['B', 'R']):
            ax.imshow(norm_images[colour], cmap='gray', origin='lower')
            ax.set_title(channel_labels[colour], fontsize=12)
            ax.axis('off')
    else: # R, G and B available
        # Create a horizontal panel with 3 grayscale subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # 3 images side-by-side
        
        # Map colour channels to filters for labelling
        channel_labels = {'R': 'F1800W', 'G': 'F770W', 'B': 'F444W'}
        
        # Legend info
        legend_text = f"R: {os.path.basename(imagesRGB['R'][0]).split('_')[1]}\n" \
                      f"G: {os.path.basename(imagesRGB['G'][0]).split('_')[1]}\n" \
                      f"B: {os.path.basename(imagesRGB['B'][0]).split('_')[1]}"
        
        for ax, colour in zip(axes, ['B', 'G', 'R']):
            ax.imshow(norm_images[colour], cmap='gray', origin='lower')
            ax.set_title(channel_labels[colour], fontsize=12)
            ax.axis('off')
    
    plt.tight_layout()
    greyscale = outfile.replace('.pdf', '_filters.pdf')
    plt.savefig(greyscale, bbox_inches='tight', pad_inches=0.1)
    
    # --- Plot using matplotlib and save RGB composite ---
    fig, ax = plt.subplots(figsize=(6, 6))
    rgb_image = np.stack([norm_images['R'], norm_images['G'], norm_images['B']], axis=-1)
    ax.imshow(rgb_image, origin='lower')
    
    # Add legend with filter information
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax.axis('off') # Hide axes
    
    # Save the RGB image as PDF
    fig.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig) # Clean up the figure
    
    # Remove temporary files
    for temp_file in temp_files.values():
        if os.path.exists(temp_file):
            os.remove(temp_file)
