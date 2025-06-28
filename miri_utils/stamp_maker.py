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
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from collections import defaultdict
from .photometry_tools import load_vis    

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
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0, None)
    img -= np.min(img)
    
    if stretch == 'asinh':
        img = np.arcsinh(alpha * Q * img) / Q
    elif stretch == 'log':
        img = np.log10(1 + alpha * img)
    elif stretch == 'linear':
        pass
    else:
        raise ValueError("Unknown stretch")

    max_val = np.nanmax(img)
    if max_val > 0:
        img /= max_val

    # Apply weight AFTER normalisation to adjust channel brightness
    img *= weight
    img = np.clip(img, 0, 1)

    return img

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
    
    # Background estimation could be improved here
    mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    img = np.where(np.isnan(img), median, img)
    img = np.clip(img, 0, None)

    return normalise_image(img, stretch=stretch, Q=Q, alpha=alpha, weight=weight)




def make_stamps(table_path, cutout_dir, output_dir):
    """
    Create RGB composite images for galaxies with different numbers of available filters.
    Combines image processing and RGB creation into one comprehensive function.
    
    Filter assignment rules:
    - NIRCam filter: F444W (always available)
    - MIRI filters: F770W, F1000W, F1800W, F2100W
    - 1 MIRI Filter: Use F444W as blue and MIRI as red
    - 2 MIRI Filters: Use F444W as blue and MIRI as green and red
    - 3-4 MIRI filters: Only use MIRI filters
    
    Parameters:
    -----------
    table_path : str
        Path to FITS table containing galaxy IDs and filter information
    cutout_dir : str  
        Directory containing FITS cutout files
    output_dir : str
        Directory to save output RGB composite images
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    table = Table.read(table_path, format='fits')

    if 'Filters' not in table.colnames or 'ID' not in table.colnames:
        raise ValueError("FITS table must contain 'Filters' and 'ID' columns.")

    # Define filter categories
    NIRCAM_FILTER = 'F444W'
    MIRI_FILTERS = ['F770W', 'F1000W', 'F1800W', 'F2100W']
    
    # Group galaxies by number of MIRI filters available
    miri_filters_per_count = defaultdict(list)

    vis_dict ={}
    for row in table:
        gid = row['ID']
        filters = [f.strip() for f in row['Filters'].split(',') if f.strip()]
        
        vis_array = []
        for filt in filters:
            vis_file = f'{gid}_{filt}.h5'
            vis_array.append(vis_file)
        
        # Add all the vis_data to the vis_dict for each galaxy ID
        vis_dict[gid] = vis_array
        
        # Count MIRI filters available for this galaxy
        miri_available = [f for f in filters if f in MIRI_FILTERS]
        miri_count = len(miri_available)
        
        miri_filters_per_count[miri_count].append((gid, filters, miri_available))
    
    # Define default stretch parameters for different filters
    default_params = {
        'F444W': {'Q': 12, 'alpha': 0.08, 'weight': 1.6},  # Blue (strong boost)
        'F770W': {'Q': 10, 'alpha': 0.10, 'weight': 0.4},  # Red (heavily suppressed)
        'F1000W': {'Q': 10, 'alpha': 0.08, 'weight': 0.7}, # Optional green/alt
        'F1800W': {'Q': 4,  'alpha': 0.25, 'weight': 0.3}, # MIRI long - suppressed
        'F2100W': {'Q': 3,  'alpha': 0.30, 'weight': 0.2}  # MIRI longest - heavily suppressed
    }
    
     # Default fallback parameters
    default_fallback = {'Q': 8, 'alpha': 0.12, 'weight': 1.0}

    # Get parameters for each channel with blue boost
    def get_channel_params(filter_name, is_red=False, is_green=False, is_blue=False):
        params = default_params.get(filter_name, default_fallback).copy()
        if is_red:
            # Weaken red significantly
            params['weight'] *= 0.3    # Drastically reduce red brightness
            params['Q'] = min(params['Q'], 8)  # Make red flatter (less contrast)
            params['alpha'] *= 1.3     # Make stretch flatter

        elif is_green:
            # Modest green boost
            params['weight'] *= 1.4
            params['Q'] = max(params['Q'], 10)
            params['alpha'] *= 0.9

        elif is_blue:
            # Strong blue boost
            params['weight'] *= 1.7
            params['Q'] = max(params['Q'], 12)
            params['alpha'] *= 0.8

        return params
    
    def find_cutout_file(gid, filter_name, cutout_dir):
        """Find the actual cutout file matching the pattern."""
        pattern = os.path.join(cutout_dir, f'{gid}_{filter_name}_cutout*.fits')
        matches = glob.glob(pattern)
        if matches:
            return matches[0]  # Return first match
        else:
            raise FileNotFoundError(f"No cutout file found for {gid} with filter {filter_name}")

    
    # Main processing loop
    for miri_count in sorted(miri_filters_per_count.keys()):
        print(f"\nProcessing {len(miri_filters_per_count[miri_count])} galaxies with {miri_count} MIRI filter(s):")
        
        for gid, all_filters, miri_available in miri_filters_per_count[miri_count]:
            print(f"  Processing {gid}: MIRI filters = {', '.join(miri_available) if miri_available else 'None'}")
            
            try:
                if miri_count == 0:
                    # No MIRI filters available, skip this galaxy
                    print(f"    ✗ Skipping {gid}: No MIRI filters available")
                    continue
                    
                elif miri_count == 1:
                    # 1 MIRI Filter: Use F444W as blue and MIRI as red
                    b_file = find_cutout_file(gid, NIRCAM_FILTER, cutout_dir)
                    r_file = find_cutout_file(gid, miri_available[0], cutout_dir)
                    
                    imagesRGB = {
                        'R': [r_file + '[0]'], 
                        'G': ['fake_green.fits[0]'], 
                        'B': [b_file + '[0]']
                    }
                    print(f"    Using filters: R={miri_available[0]}, G=fake, B={NIRCAM_FILTER}")
                
                elif miri_count == 2:
                    # 2 MIRI Filters: Use F444W as blue and MIRI as green and red
                    b_file = find_cutout_file(gid, NIRCAM_FILTER, cutout_dir)
                    
                    # Sort MIRI filters by wavelength
                    sorted_miri = miri_available
                    g_file = find_cutout_file(gid, sorted_miri[0], cutout_dir)  # Shorter wavelength
                    r_file = find_cutout_file(gid, sorted_miri[1], cutout_dir)  # Longer wavelength
                    
                    imagesRGB = {
                        'R': [r_file + '[0]'], 
                        'G': [g_file + '[0]'], 
                        'B': [b_file + '[0]']
                    }
                    print(f"    Using filters: R={sorted_miri[1]}, G={sorted_miri[0]}, B={NIRCAM_FILTER}")
                
                elif miri_count >= 3:
                    # 3-4 MIRI filters: Only use MIRI filters
                    # Sort MIRI filters by wavelength
                    sorted_miri = sorted(miri_available, key=lambda x: int(x[1:5]))
                    
                    if miri_count == 3:
                        # Use all 3 MIRI filters
                        b_file = find_cutout_file(gid, sorted_miri[0], cutout_dir)  # Shortest
                        g_file = find_cutout_file(gid, sorted_miri[1], cutout_dir)  # Middle
                        r_file = find_cutout_file(gid, sorted_miri[2], cutout_dir)  # Longest
                        
                        print(f"    Using filters: R={sorted_miri[2]}, G={sorted_miri[1]}, B={sorted_miri[0]}")
                    
                    else:  # miri_count == 4
                        # Use 3 out of 4 MIRI filters with good spacing
                        b_file = find_cutout_file(gid, sorted_miri[0], cutout_dir)  # Shortest
                        g_file = find_cutout_file(gid, sorted_miri[2], cutout_dir)  # Third
                        r_file = find_cutout_file(gid, sorted_miri[3], cutout_dir)  # Longest
                        
                        print(f"    Using filters: R={sorted_miri[3]}, G={sorted_miri[2]}, B={sorted_miri[0]}")
                        print(f"    Skipping: {sorted_miri[1]}")
                    
                    imagesRGB = {
                        'R': [r_file + '[0]'], 
                        'G': [g_file + '[0]'], 
                        'B': [b_file + '[0]']
                    }
                
                # Get parameters for each channel based on the filters being used
                r_filter = os.path.basename(imagesRGB['R'][0]).split('_')[1] if 'fake' not in imagesRGB['R'][0] else 'F2100W'
                g_filter = os.path.basename(imagesRGB['G'][0]).split('_')[1] if 'fake' not in imagesRGB['G'][0] else 'F1000W'  
                b_filter = os.path.basename(imagesRGB['B'][0]).split('_')[1] if 'fake' not in imagesRGB['B'][0] else 'F444W'
                
                rgb_params = {
                    'R': get_channel_params(r_filter, is_red=True),
                    'G': get_channel_params(g_filter, is_green=True), 
                    'B': get_channel_params(b_filter, is_blue=True) # Boost blue channel
                }

                # Create output filename
                outfile = os.path.join(output_dir, f'{gid}_rgb.pdf')

                # Create RGB composite using simplified parameter passing
                create_rgb_plot(
                    imagesRGB=imagesRGB,
                    params=rgb_params,
                    stretch='asinh',
                    outfile=outfile
                )
                
                print(f"    ✓ Created: {outfile}")
                
            except Exception as e:
                print(f"    ✗ Error processing {gid}: {str(e)}")
                continue
    
    print(f"\nRGB composite creation complete. Output saved to: {output_dir}")
    
    
    
def create_rgb_plot(imagesRGB, params, stretch, outfile):
    """
    Function to create RGB composite from processed images.
    
    Args:
        imagesRGB: Image data
        params: Dict with 'r', 'g', 'b' keys, each containing Q, alpha, weight
        stretch: Stretch method
        outfile: Output file path
    """
    print("Creating RGB image now...")
    
    stretched_images = {}
    global_max = 0.0
    
    # --- Section for normalising the images ---
    # Stretch each image and find the global max
    for colour in ['R', 'G', 'B']:
        image_str = imagesRGB[colour][0]
        
        if 'fake' in image_str:
            stretched_images[colour] = None  # We'll fill it later
            continue
        
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
        
    
    # Now fill in any missing (fake) channel using shape from a valid one
    ref_shape = None
    for colour in ['R', 'B']:
        if stretched_images[colour] is not None:
            ref_shape = stretched_images[colour].shape
            break
    if ref_shape is None:
        raise ValueError("No valid image to infer shape from for fake channel.")

    for colour in ['R', 'G', 'B']:
        if stretched_images[colour] is None:
            stretched_images[colour] = np.zeros(ref_shape)
        
    norm_images = {c: np.clip(stretched_images[c] / global_max, 0, 1) for c in ['B', 'G', 'R']}
    
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