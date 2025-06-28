import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import sedpy
from prospect.io import read_results as reader
from prospect.models import model_setup
from prospect import prospect_args
from prospect.sources import CSPSpecBasis
import h5py

def get_zred(objid):
    """Get redshift for given object ID"""
    global id_to_z
    return id_to_z.get(objid, None)

def load_hst_nircam_phot(objid, phot_table_path):
    """
    Load HST/NIRCam photometry for given object ID
    
    Parameters:
    objid : int - Object ID
    phot_table_path : str - Path to your HST/NIRCam photometry table
    
    Returns:
    dict with photometry data
    """
    # Load your HST/NIRCam photometry table
    phot_table = Table.read(phot_table_path)
    
    # Filter for your object - adjust column name as needed
    obj_row = phot_table[phot_table['ID'] == objid]  # or str(objid) if ID is string
    
    if len(obj_row) == 0:
        print(f"Object {objid} not found in photometry table")
        return None
    
    # Define your filter mapping - update this based on your actual filters
    filter_dict_hst_nircam = {
        'F435W': 'hst_acs_wfc_f435w',
        'F606W': 'hst_acs_wfc_f606w', 
        'F775W': 'hst_acs_wfc_f775w',
        'F814W': 'hst_acs_wfc_f814w',
        'F850LP': 'hst_acs_wfc_f850lp',
        'F105W': 'hst_wfc3_ir_f105w',
        'F125W': 'hst_wfc3_ir_f125w',
        'F140W': 'hst_wfc3_ir_f140w',
        'F160W': 'hst_wfc3_ir_f160w',
        'F090W': 'jwst_f090w',
        'F115W': 'jwst_f115w',
        'F150W': 'jwst_f150w',
        'F200W': 'jwst_f200w',
        'F277W': 'jwst_f277w',
        'F356W': 'jwst_f356w',
        'F410M': 'jwst_f410m',
        'F444W': 'jwst_f444w'
    }
    
    # Load filters
    filter_names = list(filter_dict_hst_nircam.values())
    filters = sedpy.observate.load_filters(filter_names)
    
    # Extract photometry - you'll need to adjust column names to match your table
    fluxes = []
    flux_errors = []
    
    for filt in filter_dict_hst_nircam.keys():
        # Adjust these column names to match your photometry table
        flux_col = f'flux_{filt}'  # or however flux is stored
        err_col = f'flux_err_{filt}'  # or however flux error is stored
        
        if flux_col in obj_row.colnames:
            flux = obj_row[flux_col][0]
            flux_err = obj_row[err_col][0] if err_col in obj_row.colnames else flux * 0.1
            
            # Convert to maggies if needed (assuming input is in µJy)
            fluxes.append(flux / 3631e6)  # µJy to maggies
            flux_errors.append(flux_err / 3631e6)
        else:
            fluxes.append(np.nan)
            flux_errors.append(np.nan)
    
    # Get effective wavelengths
    wave_eff = np.array([f.wave_effective for f in filters])
    
    return {
        'maggies': np.array(fluxes),
        'maggies_err': np.array(flux_errors),
        'wave_eff': wave_eff,
        'filters': filters,
        'filter_names': list(filter_dict_hst_nircam.keys())
    }

def load_miri_phot(objid, miri_phot_path):
    """Load MIRI photometry for given object ID"""
    miri_phot = Table.read(miri_phot_path)
    
    # Filter for your object
    ph = miri_phot[miri_phot['ID'] == str(objid)]
    
    if len(ph) == 0:
        print(f"Object {objid} not found in MIRI photometry")
        return None
    
    # MIRI filter dictionary
    filter_dict_miri = {
        'F770W': 'jwst_f770w',
        'F1800W': 'jwst_f1800w'
    }
    
    # Load the MIRI filters
    miri_filter_names = list(filter_dict_miri.values())
    miri_filters = sedpy.observate.load_filters(miri_filter_names)
    
    # Extract fluxes and errors
    flux = ph['Flux'][0] if len(ph) > 0 else np.nan
    flux_err = ph['Flux_Err'][0] if len(ph) > 0 else np.nan
    
    # Convert to maggies (assuming input is in µJy)
    miri_maggies = np.array(flux) / 3631e6  # µJy to maggies
    miri_maggies_err = np.array(flux_err) / 3631e6
    
    # Get effective wavelengths
    miri_wave_eff = np.array([f.wave_effective for f in miri_filters])
    
    return {
        'maggies': miri_maggies,
        'maggies_err': miri_maggies_err,
        'wave_eff': miri_wave_eff,
        'filters': miri_filters,
        'filter_names': list(filter_dict_miri.keys())
    }

def reconstruct_model_from_h5(h5_file_path, zred):
    """
    Reconstruct the PROSPECTOR model from .h5 file
    
    Parameters:
    h5_file_path : str - Path to PROSPECTOR .h5 output file
    zred : float - Redshift of the object
    
    Returns:
    dict with model spectrum and parameters
    """
    
    # Read the results
    try:
        results, obs, model = reader.results_from(h5_file_path)
    except:
        # If obs is None, we'll reconstruct what we can
        with h5py.File(h5_file_path, 'r') as f:
            results = {}
            results['chain'] = f['sampling/chain'][:]
            results['lnprobability'] = f['sampling/lnprobability'][:]
            
    # Get MAP (Maximum A Posteriori) parameters
    if 'chain' in results and 'lnprobability' in results:
        chains = results['chain']
        log_probabilities = results['lnprobability']
        
        # Flatten if needed
        if len(chains.shape) > 2:
            nwalkers, nsteps, ndim = chains.shape
            chains = chains.reshape(-1, ndim)
            log_probabilities = log_probabilities.flatten()
        
        # Get best-fit parameters
        max_prob_index = np.argmax(log_probabilities)
        map_parameters = chains[max_prob_index]
        
        print(f"Best-fit log probability: {log_probabilities[max_prob_index]:.2f}")
        print(f"MAP parameters shape: {map_parameters.shape}")
        
        # If we have a model, try to generate spectrum
        if model is not None:
            # Set model parameters to MAP values
            model.params.update(map_parameters)
            
            # Generate model spectrum
            wave = np.logspace(np.log10(1000), np.log10(50000), 5000)  # 1000-50000 Å
            try:
                spectrum = model.spectrum(wave, zred=zred)[1]  # [1] is flux
                
                return {
                    'wave': wave,
                    'spectrum': spectrum,
                    'map_parameters': map_parameters,
                    'model': model
                }
            except Exception as e:
                print(f"Error generating model spectrum: {e}")
                return {
                    'map_parameters': map_parameters,
                    'model': model
                }
        else:
            return {
                'map_parameters': map_parameters
            }
    else:
        print("Could not find chain or lnprobability in results")
        return None

def plot_prospector_with_data(objid, h5_file_path, hst_nircam_phot_path, 
                             miri_phot_path, zred):
    """
    Plot PROSPECTOR best-fit model with HST/NIRCam and MIRI photometry
    
    Parameters:
    objid : int - Object ID
    h5_file_path : str - Path to PROSPECTOR .h5 file
    hst_nircam_phot_path : str - Path to HST/NIRCam photometry table
    miri_phot_path : str - Path to MIRI photometry file
    zred : float - Redshift
    """
    
    # Load photometry data
    hst_data = load_hst_nircam_phot(objid, hst_nircam_phot_path)
    miri_data = load_miri_phot(objid, miri_phot_path)
    
    # Reconstruct model
    model_data = reconstruct_model_from_h5(h5_file_path, zred)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot HST/NIRCam photometry if available
    if hst_data is not None:
        rest_wave_hst = hst_data['wave_eff'] / (1 + zred)
        valid_hst = ~np.isnan(hst_data['maggies'])
        
        ax.errorbar(rest_wave_hst[valid_hst], 
                   hst_data['maggies'][valid_hst] * 3631e6,  # Convert to µJy
                   yerr=hst_data['maggies_err'][valid_hst] * 3631e6,
                   fmt='o', color='blue', label='HST/NIRCam Photometry', 
                   alpha=0.7, markersize=6)
    
    # Plot MIRI photometry if available
    if miri_data is not None:
        rest_wave_miri = miri_data['wave_eff'] / (1 + zred)
        valid_miri = ~np.isnan(miri_data['maggies'])
        
        ax.errorbar(rest_wave_miri[valid_miri], 
                   miri_data['maggies'][valid_miri] * 3631e6,  # Convert to µJy
                   yerr=miri_data['maggies_err'][valid_miri] * 3631e6,
                   fmt='s', color='red', label='MIRI Photometry', 
                   alpha=0.7, markersize=8)
    
    # Plot model spectrum if available
    if model_data is not None and 'spectrum' in model_data:
        rest_wave_model = model_data['wave'] / (1 + zred)
        model_flux = model_data['spectrum'] * 3631e6  # Convert to µJy
        
        ax.plot(rest_wave_model, model_flux, 
               color='green', alpha=0.8, linewidth=2, 
               label='PROSPECTOR Best-fit Model')
    
    # Formatting
    ax.set_xlabel('Rest-frame Wavelength (Å)', fontsize=12)
    ax.set_ylabel('Flux (µJy)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.set_title(f'Object {objid}: PROSPECTOR Fit + Photometry (z={zred:.3f})', 
                fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    ax.set_xlim(1000, 30000)  # Rest-frame wavelength range
    
    plt.tight_layout()
    plt.savefig(f'prospector_fit_comparison_{objid}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig, model_data

# Example usage function
def example_usage():
    """
    Example of how to use the plotting function
    """
    
    # Example parameters - adjust these for your data
    objid = 12345  # Your object ID
    h5_file = '/path/to/your/prospector_output_12345.h5'
    hst_nircam_phot = '/path/to/your/hst_nircam_photometry.fits'
    miri_phot = '/home/bpc/University/master/Red_Cardinal/photometry/results/Flux_Aperture_PSFMatched_AperCorr_MIRI_v4.fits'
    zred = 0.5  # Your object's redshift
    
    # Create the plot
    fig, model_data = plot_prospector_with_data(objid, h5_file, hst_nircam_phot, 
                                               miri_phot, zred)
    
    return fig, model_data

# If you want to plot multiple objects
def plot_multiple_objects_fixed(objid_list, h5_dir, hst_nircam_phot_path, 
                               miri_phot_path, zred_dict):
    """
    Plot multiple objects
    
    Parameters:
    objid_list : list - List of object IDs
    h5_dir : str - Directory with .h5 files
    hst_nircam_phot_path : str - Path to HST/NIRCam photometry
    miri_phot_path : str - Path to MIRI photometry
    zred_dict : dict - Dictionary mapping objid to redshift
    """
    
    import os
    
    for objid in objid_list:
        # Find corresponding .h5 file
        h5_files = [f for f in os.listdir(h5_dir) if f'_{objid}_' in f and f.endswith('.h5')]
        
        if h5_files:
            h5_file = os.path.join(h5_dir, h5_files[0])
            zred = zred_dict.get(objid, 0.5)  # Default redshift if not found
            
            print(f"Plotting object {objid} (z={zred:.3f})")
            
            try:
                fig, model_data = plot_prospector_with_data(
                    objid, h5_file, hst_nircam_phot_path, miri_phot_path, zred
                )
                plt.close(fig)  # Close to prevent memory issues
                
            except Exception as e:
                print(f"Error plotting object {objid}: {e}")
        else:
            print(f"No .h5 file found for object {objid}")