#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIRI Utils Rotate FITS Module
=============================

This module provides functions for expanding and rotating FITS files 
from JWST/MIRI observations. The main purpose is to prepare images for 
further processing by correcting for rotation and preventing cropping 
during transformations.

Functions
---------
- calculate_angle(fits_file):
    Reads the FITS header and extracts the rotation angle relative to 
    the N and E directions.

- expand_fits(fits_file, output_folder, angle, write=False):
    Expands the FITS image to prevent cropping when rotated, updates 
    the WCS, and saves the expanded FITS file.

- rotate_exp(fits_file, output_dir, angle):
    Rotates an expanded FITS file, adjusts the WCS to preserve world 
    coordinate alignment, and saves the rotated FITS file.

Requirements
------------
- astropy
- numpy
- scipy

Example usage
-------------
The functions in this module are designed to work together for 
preparing MIRI FITS images:

    from miri_utils.rotate import calculate_angle, expand_fits, rotate_exp
    
    angle = calculate_angle('example.fits')
    expanded_file = expand_fits('example.fits', 'expanded/', angle, write=True)
    rotate_exp(expanded_file, 'rotated/', angle)

Author: Benjamin P. Collins
Date: May 15, 2025
Version: 1.0
"""

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import os
from scipy.ndimage import rotate


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

def expand_fits(fits_file, output_folder, angle, write=False):
    """
    Expands the FITS image to prevent cropping when rotated,
    updates the WCS, and saves the new expanded FITS file.

    Args:
        fits_file (str): Path to the input FITS file.
        output_folder (str): Directory to save the expanded FITS file.
        angle (float): The rotation angle in degrees (used to determine expansion size).

    Returns:
        str: Path to the expanded FITS file.
    """

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    with fits.open(fits_file) as hdul:
        image_data = hdul[1].data
        header = hdul[1].header
        wcs = WCS(header)

        # Original dimensions
        ny, nx = image_data.shape

        # Compute the new bounding box size to avoid cropping
        theta = np.radians(angle)
        cos_t, sin_t = np.abs(np.cos(theta)), np.abs(np.sin(theta))
        new_nx = int(nx * cos_t + ny * sin_t)
        new_ny = int(nx * sin_t + ny * cos_t)

        # Calculate padding needed
        padx = (new_nx - nx) // 2
        pady = (new_ny - ny) // 2

        # Expand the image with NaN values
        expanded_image = np.full((ny + 2 * pady, nx + 2 * padx), np.nan, dtype=np.float32)
        expanded_image[pady:pady+ny, padx:padx+nx] = image_data

        # Adjust WCS: Update reference pixel position
        header['CRPIX1'] += padx
        header['CRPIX2'] += pady
        
        # Generate output filename
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(fits_file))[0] + '_exp.fits')        

        # Save the expanded image
        hdu = fits.PrimaryHDU(expanded_image, header=header)
        
        if write: hdu.writeto(output_file, overwrite=True)
        
        print(f"Saved expanded FITS file to {output_file}")
    
    return output_file


def rotate_exp(fits_file, output_dir, angle):
    """A function that takes in an expanded .fits file and performs the rotation about
    a certain angle specified when calling the function.

    Args:
        expanded_file (string): The expanded .fits file to be rotated
        output_dir (string): The output directory to save the rotated .fits file to
        angle (float): The angle of rotation
    """
    with fits.open(fits_file) as hdul:
        hdul.info()
        image_data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)

        # Get original reference pixel (before rotation)"
        crpix_original = np.array([header['CRPIX1'], header['CRPIX2']])
        print("CRPIX1:", header['CRPIX1'], "CRPIX2:", header['CRPIX2'])
        print("Image shape:", image_data.shape)
        print(crpix_original)
 
        # Get world coordinates of the original reference pixel (before rotation)
        crval_original = wcs.pixel_to_world(crpix_original[0], crpix_original[1])
        
        # Update the WCS: Rotate PC matrix by -angle (to match sky rotation)
        theta = np.radians(-angle)  # Convert to radians (negative for counterclockwise)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        if 'PC1_1' in header and 'PC2_2' in header:
            pc_matrix = np.array([[header['PC1_1'], header['PC1_2']],
                                  [header['PC2_1'], header['PC2_2']]])
            rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            new_pc_matrix = np.dot(rotation_matrix, pc_matrix)
        
        # Update the PC matrix in the header
        header['PC1_1'], header['PC1_2'] = new_pc_matrix[0]
        header['PC2_1'], header['PC2_2'] = new_pc_matrix[1]

        # Recalculate reference pixel position
        new_wcs = WCS(header)
        new_crpix = new_wcs.world_to_pixel(crval_original)

        # Correct for displacement by shiftingimage_data CRPIX back to original sky position
        header['CRPIX1'] = np.round(new_crpix[0])
        header['CRPIX2'] = np.round(new_crpix[1])

        # Perform the actual action of rotation - this is where the magic happens
        rotated_image = rotate(image_data, angle, reshape=False, order=1, cval=np.nan, mode="grid-constant")
        
        # Save the rotated FITS file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(fits_file).replace('_exp.fits', '_rot.fits'))
        hdu = fits.PrimaryHDU(rotated_image, header=header)
        hdu.writeto(output_file, overwrite=True)

        print(f"Rotated image saved to {output_file}")