#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIRI Utils Astronomical Image Cutout Generator
==============================================

This script creates cutout images from astronomical FITS files based on catalogue coordinates.
It extracts regions of interest around the specified celestial coordinates and preserves all
data extensions of the original FITS files. The script also generates preview PNG images
for quick visual inspection of the cutouts.

Dependencies:
    - astropy: For FITS file handling, WCS transformations, and coordinate operations
    - matplotlib: For generating preview images
    - numpy: For array operations and numerical calculations

Author: Benjamin P. Collins
Date: Nov 21, 2025
Version: 3.0
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
from scipy.ndimage import rotate

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

def resample_cutout(indir, num_pixels):
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

def produce_cutouts(cat, indir, output_dir, survey, x_arcsec, filter, nan_thresh=0.4, suffix='', preview=False):
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
    
    nan_thresh : float, optional
        Maximum allowed fraction of NaN values in a cutout (default: 0.4).
        Cutouts with more NaNs than this threshold will be discarded.
    
    suffix : str, optional
        Additional string to append to output filenames.
    
    preview: bool, optional
        If True, generates PNG preview images for each cutout. Defaults to False.
    
    Returns
    -------
    None
        Files are written to disk at the specified output_dir.
    
    Notes
    -----
    The function assumes MIRI pixel scale of 0.11092 arcsec/pixel for calculating
    cutout size in pixels. Adjust this value if using data from different instruments.
    """

    # Extract survey name and observation number from the survey parameter
    if '1' in survey:
        survey_name = survey[:-1]
        obs = '1'
    elif '2' in survey:
        survey_name = survey[:-1]
        obs = '2'
    else:
        survey_name = survey
        obs = ''
    
    # Load target catalogue with object IDs and coordinates
    with fits.open(cat) as catalog_hdul:
        cat_data = catalog_hdul[1].data
        ids = cat_data['id']
        ra = cat_data['ra']
        dec = cat_data['dec']  

    # Find all FITS files matching the requested filter
    filter_l = filter.lower()
    fits_files = glob.glob(os.path.join(indir, f"*{filter_l}*.fits"))
    print(f"Found {len(fits_files)} FITS files from the {survey_name} survey with filter {filter}.")
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
            ref_data = hdul[0].data
            ref_header = hdul[0].header
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
                    
                    # Calculate angle of rotation for NE cross
                    angle = calculate_angle(fits_file)  
                    
                    plt.figure(figsize=(6, 6))                   
                    plt.imshow(preview_data, origin="lower", cmap="gray")
                    plt.title(filter)
                    
                    # Draw North/East compass
                    ax = plt.gca()
                    draw_NE_cross(ax, angle_deg=angle, size=50, offset=20)
                    
                    png_filename = os.path.join(output_dir, f"{ids[i]}_{filter}_cutout_{survey_name}{obs}{suffix}.png")
                    plt.savefig(png_filename)
                    plt.close()

                    # Save multi-extension FITS cutout
                    fits_filename = os.path.join(output_dir, f"{ids[i]}_{filter}_cutout_{survey_name}{obs}{suffix}.fits")
                    cutout_hdul.writeto(fits_filename, overwrite=True)
                    counts += 1

    # Report completion statistics
    print(f"Produced cutouts for {counts} of {total} galaxies in the catalogue.")


def draw_NE_cross(ax, angle_deg, size=40, offset=10, colour="white", lw=2):
    """
    Draw a North-East direction cross in the top-right corner of an image.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw.
    angle_deg : float
        Rotation of the image relative to North (in degrees).
    size : float
        Length of the N/E arrows (pixels).
    offset : float
        Distance from the border (pixels).
    colour : str
        Colour of the lines/text.
    lw : float
        Line width.
    """

    angle = np.deg2rad(angle_deg)
    east_angle = angle + np.pi/2

    # Get image dimensions from axis limits
    x_max = ax.get_xlim()[1]
    y_max = ax.get_ylim()[1]

    # Origin of the compass (top-right corner, moved inward by offset)
    x0 = x_max - offset
    y0 = y_max - offset

    # North arrow endpoint
    xN = x0 + size * np.sin(angle)     # sin(angle) because image coords increase upward
    yN = y0 + size * np.cos(angle)

    # East arrow endpoint (90° CCW)
    xE = x0 + size * np.sin(east_angle)
    yE = y0 + size * np.cos(east_angle)

    # Draw arrows
    ax.plot([x0, xN], [y0, yN], colour, lw=lw)
    ax.plot([x0, xE], [y0, yE], colour, lw=lw)

    # Labels
    ax.text(xN, yN, "N", color=colour, fontsize=12, ha="center", va="center")
    ax.text(xE, yE, "E", color=colour, fontsize=12, ha="center", va="center")


def calculate_angle(fits_file):
    """A function that reads in the header of a .fits file and extracts the information
        about the rotation of the image with respect to the N and E directions.

    Args:
        fits_file (string): The .fits file to be rotated in the next steps
    """
    with fits.open(fits_file) as hdul:
        header = hdul[1].header

        # Extract rotation angle from PC matrix
        if 'PC1_1' in header and 'PC2_2' in header:
            cost = header['PC1_1'] 
            sint = header['PC2_1']
            
            # 180 - angle takes care of X-axis flipping in sky coordinates!
            angle = 180 - np.arccos(cost) * 180 / np.pi
            
            # Restore quadrant information since cos is symmetric
            if sint < 0:
                angle = -angle
            
            #print(f"The image {fits_file} is rotated by {angle:.2f} degrees with respect to North")        
        else:
            print("No PC matrix found, assuming no rotation")
            angle = 0
        
    return angle

        
def rotate_cutouts(cutout_dir, output_dir):
    """Function that reads in cutout FITS files and rotates them so that their Y-axis 
        aligns with north

    Args:
        cutout_dir (str): Directory containing the larger cutouts
        output_dir (str): Directory to store the rotated cutouts
    """
    fits_array = glob.glob(os.path.join(cutout_dir, "*.fits"))
    print(f"Found {len(fits_array)} cutout FITS files.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    for fits_file in fits_array:
        
        fname = os.path.basename(fits_file)
        gal_id = fname.split('_')[0]
        filter = fname.split('_')[1]
        survey_obs = fname.split('_')[3]
        
        # Check the survey obs
        if '003' in survey_obs:
            survey = 'primer'
            obs = '003'
        elif '004' in survey_obs:
            survey = 'primer'
            obs = '004'
        elif 'cweb1' in survey_obs:
            survey = 'cweb'
            obs = '1'
        elif 'cweb2' in survey_obs:
            survey = 'cweb'
            obs = '2'
        elif 'cos3d1' in survey_obs:
            survey = 'cos3d'
            obs = '1'
        elif 'cos3d2' in survey_obs:
            survey = 'cos3d'
            obs = '2'
        else:
            print(f"Unknown survey and/or observation number for galaxy {id}:\n")
            print(survey_obs)
        
        angle = calculate_angle(fits_file)
        
        with fits.open(fits_file) as hdul:
            image_data = hdul[1].data
            header = hdul[1].header
            wcs = WCS(header)
        
        crpix_original = np.array([header['CRPIX1'], header['CRPIX2']])
        crval_original = wcs.pixel_to_world(crpix_original[0], crpix_original[1])

        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[cos_t, -sin_t], 
                                    [sin_t, cos_t]])
        
        if 'PC1_1' in header and 'PC2_2' in header:
            pc_matrix = np.array([[header['PC1_1'], header['PC1_2']],
                                [header['PC2_1'], header['PC2_2']]])
            rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            new_pc_matrix = np.dot(rotation_matrix, pc_matrix)
        
            # Update the PC matrix in the header
            header['PC1_1'], header['PC1_2'] = new_pc_matrix[0]
            header['PC2_1'], header['PC2_2'] = new_pc_matrix[1]
        
        # Recompute the scale (CDELT1, CDELT2) based on new PC matrix
        cdelt1_sign = np.sign(header.get('CDELT1', -1))  # Usually negative
        cdelt2_sign = np.sign(header.get('CDELT2', 1))   # Usually positive

        header['CDELT1'] = cdelt1_sign * np.sqrt(header['PC1_1']**2 + header['PC2_1']**2)
        header['CDELT2'] = cdelt2_sign * np.sqrt(header['PC1_2']**2 + header['PC2_2']**2)

        
        new_wcs = WCS(header)
        new_crpix = new_wcs.world_to_pixel(crval_original)

        header['CRPIX1'] = np.round(new_crpix[0])
        header['CRPIX2'] = np.round(new_crpix[1])

        rotated_image = rotate(image_data, -angle, reshape=False, 
                               order=1, cval=np.nan, mode='grid-constant')
        
        def crop_centered_array(array, x_arcsec):
            """Crop a 2D NumPy array around the centre to the desired shape."""
            y, x = array.shape
            startx = x // 2 - x_arcsec // 2
            starty = y // 2 - x_arcsec // 2
            return array[starty:starty + x_arcsec, startx:startx + x_arcsec]
        
        # Crop cutout to 3x3 arcsec
        miri_scale = 0.11092  # arcsec per pixel
        arcsec = 3
        x_arcsec = int(arcsec/miri_scale)
        
        cropped_data = crop_centered_array(rotated_image, x_arcsec)
        
        # Save the rotated FITS file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{gal_id}_{filter}_cutout_{survey}.fits')
        hdu = fits.PrimaryHDU(cropped_data, header=header)
        hdu.writeto(output_file, overwrite=True)
        
        print(f"Rotated image saved to {output_file}")
        
        
        
