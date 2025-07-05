import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import table
import astropy.units as u
import astropy.constants as const
import sedpy
import prospect.io.read_results as reader
from prospect.sources import FastStepBasis
from prospect.models.sedmodel import PolySpecModel
from prospect.models import priors
from prospect.models.templates import TemplateLibrary
from prospect.models import transforms
from astropy.cosmology import WMAP9 as cosmo

# Set paths (adjust these to your system)
dir_new_photometry = '/home/bpc/University/master/Red_Cardinal/photometry/results/'
dirout = "/home/bpc/University/master/Red_Cardinal/prospector/fits_with_miri/"

def get_zred(objid):
    """Get redshift for object ID"""
    dat_zred = np.loadtxt('/home/bpc/University/master/Red_Cardinal/catalogues/redshifts.txt', 
                          dtype=[('galid', int),('zred', '<f8')])
    return abs(dat_zred['zred'][dat_zred['galid'] == objid][0])

def get_MAP(results):
    """Get Maximum A Posteriori parameters"""
    chains = results['chain']
    log_probabilities = results['lnprobability']
    max_prob_index = np.argmax(log_probabilities)
    return chains[max_prob_index]

def build_obs_photometry_only(objid):
    """Build observation dictionary with photometry (no spectroscopy)"""
    
    # Load photometry data
    phot_all = fits.open(dir_new_photometry + 'bluejay_phot_cat_v1.3.fits')[1].data
    ph = phot_all[phot_all['ID'] == str(objid)]
    
    # Filter dictionaries
    filter_dict_3dhst = {
        'F125W': 'wfc3_ir_f125w',
        'F140W': 'wfc3_ir_f140w',
        'F160W': 'wfc3_ir_f160w',
        'F606W': 'acs_wfc_f606w',
        'F814W': 'acs_wfc_f814w'
    }
    
    filter_dict_nircam = { 
        'F090W': 'jwst_f090w',
        'F115W': 'jwst_f115w',
        'F150W': 'jwst_f150w',
        'F200W': 'jwst_f200w',
        'F277W': 'jwst_f277w',
        'F356W': 'jwst_f356w',
        'F410M': 'jwst_f410m',
        'F444W': 'jwst_f444w'
    }
    
    # MIRI filters (not used in original fit)
    filter_dict_miri = {
        'F770W': 'jwst_f770w',
        'F1800W': 'jwst_f1800w'
    }
    
    # Combined filters used in original fit
    filter_code_orig = list(filter_dict_3dhst.keys()) + list(filter_dict_nircam.keys())
    filter_name_orig = list(filter_dict_3dhst.values()) + list(filter_dict_nircam.values())
    
    # All filters including MIRI
    filter_code_all = filter_code_orig + list(filter_dict_miri.keys())
    filter_name_all = filter_name_orig + list(filter_dict_miri.values())
    
    # Create obs dictionary
    obs = {}
    obs['filters'] = sedpy.observate.load_filters(filter_name_orig)  # Original filters for model
    obs['filters_all'] = sedpy.observate.load_filters(filter_name_all)  # All filters for plotting
    obs['filter_code'] = filter_code_orig
    obs['filter_code_all'] = filter_code_all
    
    # Extract fluxes for original filters
    fluxes = np.zeros(len(filter_code_orig))
    fluxes_err = np.zeros(len(filter_code_orig))
    
    for ff, fil in enumerate(filter_code_orig):
        fluxes[ff] = ph[str(fil) + '_flux'][0]
        fluxes_err[ff] = ph[str(fil) + '_flux_err'][0]
    
    # Extract all fluxes including MIRI
    fluxes_all = np.zeros(len(filter_code_all))
    fluxes_err_all = np.zeros(len(filter_code_all))
    
    for ff, fil in enumerate(filter_code_all):
        fluxes_all[ff] = ph[str(fil) + '_flux'][0]
        fluxes_err_all[ff] = ph[str(fil) + '_flux_err'][0]
    
    # Add systematic error
    fluxes_err = np.sqrt(fluxes_err**2 + (0.05*fluxes)**2)
    fluxes_err_all = np.sqrt(fluxes_err_all**2 + (0.05*fluxes_all)**2)
    
    # Convert to maggies
    obs['maggies'] = np.array(fluxes) / 3631
    obs['maggies_unc'] = np.array(fluxes_err) / 3631
    obs['maggies_all'] = np.array(fluxes_all) / 3631
    obs['maggies_unc_all'] = np.array(fluxes_err_all) / 3631
    
    # Masks
    obs['phot_mask'] = np.array([True for f in obs["filters"]])
    obs['phot_mask_all'] = np.array([True for f in obs["filters_all"]])
    
    # Wavelengths
    obs['phot_wave'] = np.array([f.wave_effective for f in obs['filters']])
    obs['phot_wave_all'] = np.array([f.wave_effective for f in obs['filters_all']])
    
    return obs

def build_minimal_model(zred, add_duste=True, add_neb=False, add_agn=False, fit_afe=False):
    """Build a minimal model matching the original setup"""
    
    model_params = {}
    
    # Basic parameters
    model_params['zred'] = {"N": 1, "isfree": True, "init": zred, "units": "redshift",
                           "prior": priors.Normal(mean=zred, sigma=0.005)}
    
    model_params['logzsol'] = {"N": 1, "isfree": True, "init": -0.5, 
                              "units": r"$\log (Z/Z_\odot)$",
                              "prior": priors.TopHat(mini=-2, maxi=0.50)}
    
    if fit_afe:
        model_params['afe'] = {"N": 1, "isfree": True, "init": 0.0,
                              "units": r"$[\alpha/fe]$",
                              "prior": priors.TopHat(mini=-0.2, maxi=0.6)}
    else:
        model_params['afe'] = {"N": 1, "isfree": False, "init": 0.0}
    
    # SFH setup
    tuniv = cosmo.age(zred).value
    agelims_Myr = np.append(np.logspace(np.log10(30.0), np.log10(0.8*tuniv*1000), 12), 
                           [0.9*tuniv*1000, tuniv*1000])
    agelims = np.concatenate(([0.0], np.log10(agelims_Myr*1e6)))
    agebins = np.array([agelims[:-1], agelims[1:]]).T
    nbins = len(agelims) - 1
    
    model_params["logmass"] = {"N": 1, "isfree": True, "init": 10.5,
                              'units': "Solar masses formed",
                              'prior': priors.TopHat(mini=8.5, maxi=13)}
    
    model_params["mass"] = {'N': nbins, 'isfree': False, 'init': (10**10.5)/nbins,
                           'units': "Solar masses formed",
                           'depends_on': transforms.logsfr_ratios_to_masses}
    
    model_params["agebins"] = {'N': nbins, 'isfree': False, 'init': agebins,
                              'units': 'log(yr)'}
    
    model_params["logsfr_ratios"] = {'N': nbins-1, 'isfree': True,
                                    'init': np.full(nbins-1, 0.0),
                                    'units': '',
                                    'prior': priors.StudentT(mean=np.full(nbins-1, 0.0),
                                                           scale=np.full(nbins-1, 0.3), 
                                                           df=np.full(nbins-1, 2))}
    
    # IMF
    model_params['imf_type'] = {'N': 1, 'isfree': False, 'init': 1}
    
    # Dust
    model_params['dust_type'] = {"N": 1, "isfree": False, "init": 4}
    model_params['dust2'] = {"N": 1, "isfree": True, "init": 0.5,
                            "prior": priors.TopHat(mini=0.0, maxi=4.0/1.086)}
    model_params["dust_index"] = {"N": 1, "isfree": True, "init": 0.0,
                                 "prior": priors.ClippedNormal(mini=-1.5, maxi=0.4, mean=0.0, sigma=0.3)}
    
    # Dust1 tied to dust2
    def to_dust1(dust1_fraction=None, dust2=None, **extras):
        return dust1_fraction * dust2
    
    model_params['dust1'] = {"N": 1, "isfree": False, 'depends_on': to_dust1, "init": 0.0}
    model_params['dust1_fraction'] = {'N': 1, 'isfree': True, 'init': 1.0,
                                     'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}
    
    # Add dust emission if requested
    if add_duste:
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_gamma']['isfree'] = True
        model_params['duste_gamma']['init'] = 0.01
        model_params['duste_gamma']['prior'] = priors.TopHat(mini=0.0, maxi=1.0)
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_qpah']['init'] = 3.5
        model_params['duste_qpah']['prior'] = priors.TopHat(mini=0.5, maxi=10.0)
        model_params['duste_umin']['isfree'] = True
        model_params['duste_umin']['init'] = 1.0
        model_params['duste_umin']['prior'] = priors.TopHat(mini=0.1, maxi=25.0)
    
    return PolySpecModel(model_params)

def plot_reconstruction_with_miri(objid, h5_file):
    """Main function to reconstruct and plot PROSPECTOR results with MIRI data"""
    
    # Load PROSPECTOR results
    full_path = os.path.join(dirout, h5_file)
    results, obs_orig, model_orig = reader.results_from(full_path)
    
    # Get redshift and MAP parameters
    zred = get_zred(objid)
    map_params = get_MAP(results)
    
    print(f"Galaxy {objid}: z = {zred:.4f}")
    print(f"MAP parameters shape: {map_params.shape}")
    
    # Build new observations including MIRI
    obs = build_obs_photometry_only(objid)
    
    # Build model matching original setup
    run_params = results.get('run_params', {})
    model = build_minimal_model(zred, 
                               add_duste=run_params.get('add_duste', True),
                               add_neb=run_params.get('add_neb', False),
                               add_agn=run_params.get('add_agn', False),
                               fit_afe=run_params.get('fit_afe', False))
    
    # Build SPS
    sps = FastStepBasis(zcontinuous=1)
    
    # Generate model predictions
    try:
        spec, phot, _ = model.predict(map_params, obs=obs, sps=sps)
        print(f"Model prediction successful")
        print(f"Predicted photometry shape: {phot.shape}")
        print(f"Observed photometry shape: {obs['maggies'].shape}")
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Convert to flux units for plotting
    c = const.c.to(u.AA/u.s).value  # Speed of light in AA/s
    factor = 3631 * 1e-23  # Jy to erg/s/cm2/Hz
    
    # Top panel: Full SED
    wave_all = obs['phot_wave_all']
    flux_obs_all = obs['maggies_all'] * factor * c / wave_all**2 * 1e17  # to 10^-17 erg/s/cm2/AA
    flux_err_all = obs['maggies_unc_all'] * factor * c / wave_all**2 * 1e17
    
    # Model prediction (only for original filters)
    wave_model = obs['phot_wave']
    flux_model = phot * factor * c / wave_model**2 * 1e17
    
    # Identify MIRI points
    miri_mask = np.array(['F770W' in f or 'F1800W' in f for f in obs['filter_code_all']])
    orig_mask = ~miri_mask
    
    ax1.errorbar(wave_all[orig_mask], flux_obs_all[orig_mask], yerr=flux_err_all[orig_mask],
                fmt='o', color='blue', alpha=0.7, label='HST+NIRCam (used in fit)')
    ax1.errorbar(wave_all[miri_mask], flux_obs_all[miri_mask], yerr=flux_err_all[miri_mask],
                fmt='s', color='red', alpha=0.7, label='MIRI (not in fit)')
    ax1.plot(wave_model, flux_model, 'go', markersize=8, alpha=0.8, label='Model prediction')
    
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Flux (10⁻¹⁷ erg/s/cm²/Å)')
    ax1.set_title(f'Galaxy {objid}: PROSPECTOR Fit with Additional MIRI Photometry')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Residuals for fitted data only
    orig_indices = np.where(orig_mask)[0]
    if len(orig_indices) == len(flux_model):
        residuals = (flux_obs_all[orig_mask] - flux_model) / flux_err_all[orig_mask]
        ax2.errorbar(wave_all[orig_mask], residuals, yerr=np.ones_like(residuals),
                    fmt='o', color='blue', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=3, color='red', linestyle=':', alpha=0.5)
        ax2.axhline(y=-3, color='red', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('Residuals (σ)')
        ax2.set_title('Fit Residuals (fitted data only)')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Cannot compute residuals:\nMismatch in data dimensions', 
                transform=ax2.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.show()
    
    # Print some diagnostics
    print(f"\nDiagnostics:")
    print(f"Original filters used in fit: {len(obs['filter_code'])}")
    print(f"Total filters (including MIRI): {len(obs['filter_code_all'])}")
    print(f"MIRI bands: {[f for f in obs['filter_code_all'] if 'F770W' in f or 'F1800W' in f]}")
    
    return obs, model, results

# Example usage:
if __name__ == "__main__":
    # Example: plot results for galaxy with ID 12345 using specific h5 file
    objid = 12345  # Replace with your galaxy ID
    h5_filename = "output_12345_sometag_timestamp_mcmc.h5"  # Replace with your h5 filename
    
    obs, model, results = plot_reconstruction_with_miri(objid, h5_filename)