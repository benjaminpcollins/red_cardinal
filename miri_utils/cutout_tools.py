#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIRI Utils Astronomical Image Cutout Generator
==============================================

This script creates cutout images from astronomical FITS files based on catalogue coordinates.
It extracts regions of interest around specified celestial coordinates and preserves all
data extensions in the original FITS files. The script also generates preview PNG images
for quick visual inspection of the cutouts.

Dependencies:
    - astropy: For FITS file handling, WCS transformations, and coordinate operations
    - matplotlib: For generating preview images
    - numpy: For array operations and numerical calculations

Author: Benjamin P. Collins
Date: May 15, 2025
Version: 2.0
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import warnings

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import Cutout2D

# Suppress common WCS-related warnings that don't affect functionality
warnings.simplefilter("ignore", category=FITSFixedWarning)


def load_cutout(file_path, index=1):
    """Loads a FITS cutout file and extracts the data, header, and WCS."""
    try:
        with fits.open(file_path) as hdu:
            data = hdu[index].data
            header = hdu[index].header
            wcs = WCS(header)
        return data, wcs
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, None


def produce_cutouts(cat, indir, output_dir, survey, x_arcsec, filter, obs="", nan_thresh=0.4, suffix=''):
    """
    Produces cutout images from astronomical FITS files centred on catalogue positions.
    
    This function extracts square regions of specified size around the celestial coordinates
    provided in a catalogue. It processes all FITS files in the input directory that match
    the specified filter and preserves all image extensions in the output files.
    
    Parameters
    ----------
    cat : str
        Path to the FITS catalogue file containing object IDs and coordinates.
        Expected columns: 'id', 'ra', 'dec'.
    
    indir : str
        Directory containing input FITS files to process.
    
    output_dir : str
        Directory where output cutout files will be saved. Created if it doesn't exist.
    
    survey : str
        Name of the survey. Used for naming output files and plot titles.
    
    x_arcsec : float
        Size of the cutout in arcseconds (will be a square with this side length).
    
    filter : str
        Filter name to select FITS files (e.g., 'F770W'). Will be used in filename matching.
    
    obs : str, optional
        Additional identifier for observation, used in output filenames.
    
    nan_thresh : float, optional
        Maximum allowed fraction of NaN values in a cutout (default: 0.4).
        Cutouts with more NaNs than this threshold will be discarded.
    
    suffix : str, optional
        Additional string to append to output filenames.
    
    Returns
    -------
    None
        Files are written to disk at the specified output_dir.
    
    Notes
    -----
    The function assumes MIRI pixel scale of 0.11092 arcsec/pixel for calculating
    cutout size in pixels. Adjust this value if using data from different instruments.
    """

    # Load target catalogue with object IDs and coordinates
    with fits.open(cat) as catalog_hdul:
        cat_data = catalog_hdul[1].data
        ids = cat_data['id']
        ra = cat_data['ra']
        dec = cat_data['dec']  

    # Find all FITS files matching the requested filter
    filter_l = filter.lower()
    fits_files = glob.glob(os.path.join(indir, f"*{filter_l}*.fits"))
    print(f"Found {len(fits_files)} FITS files from the {survey} survey with filter {filter}.")
    print("Processing:")
    for f in fits_files:
        print(f"{f}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Files will be saved to {output_dir}.")
    
    # Initialise counter for successful cutouts
    counts = 0
    total = len(ra)
    suffx = '_' + suffix
    
    # Calculate cutout size in pixels based on MIRI instrument scale
    miri_scale = 0.11092  # arcsec per pixel
    x_pixels = int(np.round(x_arcsec/miri_scale))
    cutout_size = (x_pixels, x_pixels)

    # Process each FITS file
    for fits_file in fits_files:
        with fits.open(fits_file) as hdul:
            # Use extension 1 as the reference for WCS and field coverage check
            ref_data = hdul[1].data
            ref_header = hdul[1].header
            ref_wcs = WCS(ref_header)

            # Process each galaxy from the catalogue
            for i in range(total):
                # Create SkyCoord object for the target position
                target_coord = SkyCoord(ra[i], dec[i], unit=(u.deg, u.deg))

                # Check if the target is within the field of view
                try:
                    x, y = ref_wcs.world_to_pixel(target_coord)
                except Exception:
                    # Skip if coordinate transformation fails
                    continue
                
                # Skip if outside the bounds of the image
                if not (0 <= x < ref_data.shape[1] and 0 <= y < ref_data.shape[0]):
                    continue

                # Initialise multi-extension FITS output file
                cutout_hdul = fits.HDUList()
                cutout_hdul.append(fits.PrimaryHDU(header=hdul[0].header))

                valid_cutout = True
                max_nan_ratio = 0.0

                # Process each extension in the input FITS file
                for ext in range(1, len(hdul)):
                    hdu = hdul[ext]
                    # Skip non-image or 1D extensions
                    if hdu.data is None or hdu.data.ndim != 2:
                        continue  # Skip non-image or 1D extensions

                    # Create cutout from this extension
                    try:
                        # Try using WCS from this extension (preferred method)
                        wcs = WCS(hdu.header)
                        cutout = Cutout2D(hdu.data, target_coord, cutout_size, wcs=wcs, mode="partial")
                        cutout_header = cutout.wcs.to_header()
                    except Exception:
                        # Fallback to pixel coordinates if WCS fails
                        cutout = Cutout2D(hdu.data, (x, y), cutout_size, mode="partial")
                        cutout_header = hdu.header.copy()

                    # Calculate fraction of NaN values in the cutout
                    nan_ratio = np.isnan(cutout.data).sum() / cutout.data.size
                    max_nan_ratio = max(max_nan_ratio, nan_ratio)

                    # Create output HDU with original extension name preserved
                    cutout_hdu = fits.ImageHDU(data=cutout.data, header=cutout_header)
                    if 'EXTNAME' in hdu.header:
                        cutout_hdu.name = hdu.header['EXTNAME']
                    
                    # Add to output file
                    cutout_hdul.append(cutout_hdu)
                
                # Save the cutout if it meets quality criteria (not too many NaNs and has data extensions)
                if max_nan_ratio < nan_thresh and len(cutout_hdul) > 1:
                    # Generate PNG preview from extension 1 data
                    preview_data = cutout_hdul[1].data
                    plt.figure(figsize=(6, 6))
                    plt.imshow(preview_data, origin="lower", cmap="gray")
                    
                    plt.title(filter)
                    png_filename = os.path.join(output_dir, f"{ids[i]}_{filter}_cutout_{survey}{obs}{suffix}.png")
                    plt.savefig(png_filename)
                    plt.close()

                    # Save multi-extension FITS cutout
                    fits_filename = os.path.join(output_dir, f"{ids[i]}_{filter}_cutout_{survey}{obs}{suffix}.fits")
                    cutout_hdul.writeto(fits_filename, overwrite=True)
                    counts += 1

    # Report completion statistics
    print(f"Produced cutouts for {counts} of {total} galaxies in the catalogue.")
