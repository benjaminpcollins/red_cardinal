import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import table
import astropy.units as u
import astropy.constants as const
from sedpy.observate import load_filters, list_available_filters
import fsps
import traceback

import prospect.io.read_results as reader
from prospect.sources import FastStepBasis
from prospect.models import priors
from prospect.models.templates import TemplateLibrary
from prospect.models import transforms
from astropy.cosmology import WMAP9 as cosmo
from prospect.models.sedmodel import PolySpecModel
from prospect.utils.obsutils import fix_obs
from collections import defaultdict




# This script is designed to work with PROSPECTOR results and MIRI photometry data.

# Set paths (adjust these to your system)
cat_dir = '/Users/benjamincollins/University/master/Red_Cardinal/photometry/catalogues/'

dirout = "/Users/benjamincollins/University/master/Red_Cardinal/prospector/outputs/"

def get_zred(objid):
    """
    Retrieve the redshift value for a given object ID from a specified file.

    Parameters:
    objid (int): The object ID for which the redshift value is to be retrieved.

    Returns:
    float: The absolute value of the redshift corresponding to the given object ID.
    """
    dat_zred = np.loadtxt('/Users/benjamincollins/University/Master/Red_Cardinal/photometry/catalogues/redshifts.txt', 
                          dtype=[('galid', int),('zred', '<f8')])
    return abs(dat_zred['zred'][dat_zred['galid'] == objid][0])

def get_MAP(results, verbose=False):
    """Get Maximum A Posteriori parameters"""
    chains = results['chain']   # Shape: (n_samples, n_parameters)
    log_probabilities = results['lnprobability']  # Shape: (n_samples,)
    
    # Find the index of the maximum log probability
    max_prob_index = np.argmax(log_probabilities)
    
    # Get the MAP parameters
    map_parameters = chains[max_prob_index]

    if verbose:
        print("MAP Parameters:", map_parameters)

    return map_parameters

def build_obs(objid):
    """Build observation dictionary with photometry (no spectroscopy)"""
    
    # Load photometry data
    phot_hst_nircam = fits.open(cat_dir + 'bluejay_phot_cat_v1.4.fits')[1].data
    ph_hst_nircam = phot_hst_nircam[phot_hst_nircam['ID'] == objid]
    
    phot_miri = fits.open(cat_dir + 'Photometry_Table_MIRI.fits')[1].data
    ph_miri = phot_miri[phot_miri['ID'] == objid]
    
    if ph_miri is None or len(ph_miri) == 0:
        print(f"No MIRI data found for object {objid}. Using only HST+NIRCam data.")
        return None
    
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
        'F1000W': 'jwst_f1000w',
        'F1800W': 'jwst_f1800w',
        'F2100W': 'jwst_f2100w'
    }
    
    # Combined filters used in original fit
    filter_code_orig = list(filter_dict_3dhst.keys()) + list(filter_dict_nircam.keys())# + list(filter_dict_miri.keys()))
    filter_name_orig = list(filter_dict_3dhst.values()) + list(filter_dict_nircam.values())# + list(filter_dict_miri.values()))
    
    # All filters including MIRI
    filter_code_all = filter_code_orig + list(filter_dict_miri.keys())
    filter_name_all = filter_name_orig + list(filter_dict_miri.values())
    
    # Create obs dictionary
    obs = {}
    obs['filters'] = load_filters(filter_name_orig)  # Original filters for model
    obs['filters_all'] = load_filters(filter_name_all)  # All filters for plotting
    obs['filter_code'] = filter_code_orig
    obs['filter_code_all'] = filter_code_all
    
    # Extract fluxes for original filters
    fluxes     = np.zeros(len(filter_code_orig))
    fluxes_err = np.zeros(len(filter_code_orig))
    
    for ff, fil in enumerate(filter_code_orig):
        fluxes[ff] = ph_hst_nircam[str(fil) + '_flux'][0]
        fluxes_err[ff] = ph_hst_nircam[str(fil) + '_flux_err'][0]
    
    # Extract MIRI filter fluxes from FITS table arrays
    miri_filters_present = ph_miri['Filters'][0].split(',')  # e.g. ['F770W', 'F1800W']
    miri_flux_array = ph_miri['Flux'][0]         # shape: (n_filters,)
    miri_flux_err_array = ph_miri['Flux_Err'][0] # shape: (n_filters,)

    # Combine all fluxes
    fluxes_all = np.concatenate([fluxes, np.full(len(filter_dict_miri), np.nan)])
    fluxes_err_all = np.concatenate([fluxes_err, np.full(len(filter_dict_miri), np.nan)])
    
    for mfilt, flux, err in zip(miri_filters_present, miri_flux_array, miri_flux_err_array):
        if mfilt in filter_dict_miri:
            idx = filter_code_all.index(mfilt)
            fluxes_all[idx] = flux
            fluxes_err_all[idx] = err
        else:
            print(f"No match for: {repr(mfilt)}")

    print("MIRI fluxes added successfully.")
    
    # Add 5% systematic error in quadrature to all bands
    fluxes_err = np.sqrt(fluxes_err**2 + (0.05*fluxes)**2)
    fluxes_err_all = np.sqrt(fluxes_err_all**2 + (0.05*fluxes_all)**2)
    
    # Convert to maggies
    obs['maggies'] = np.array(fluxes) / 3631
    obs['maggies_unc'] = np.array(fluxes_err) / 3631
    obs['maggies_all'] = np.array(fluxes_all) / 3631
    obs['maggies_unc_all'] = np.array(fluxes_err_all) / 3631
    
    # Make mask, where True means that you want to fit that data point
    obs['phot_mask'] = np.array([True for f in obs["filters"]])
    obs['phot_mask_all'] = np.array([True for f in obs["filters_all"]])
    
    # Array of effective wavelengths for each filter, useful for plotting
    obs['phot_wave'] = np.array([f.wave_effective for f in obs['filters']])
    obs['phot_wave_all'] = np.array([f.wave_effective for f in obs['filters_all']])
    
    # ensure all required keys are present in the obs dictionary
    obs = fix_obs(obs)
    
    return obs


# --------------------
# Set up model
# --------------------

# tie dust1 to dust2, with a prior centered on dust1=dust2
def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2
    
# modify to increase nbins
nbins_sfh = 14
    
def zred_to_agebins(zred=None,agebins=None,**extras):
    tuniv = cosmo.age(zred).value[0]*1e9
    tbinmax = (tuniv*0.9)
    agelims = [0.0,7.4772] + np.linspace(8.0,np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv)]
    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T

def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
    agebins = zred_to_agebins(zred=zred)
    logsfr_ratios = np.clip(logsfr_ratios,-10,10) # numerical issues...
    nbins = agebins.shape[0]
    sratios = 10**logsfr_ratios
    dt = (10**agebins[:,1]-10**agebins[:,0])
    coeffs = np.array([ (1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
    m1 = (10**logmass) / coeffs.sum()
    return m1 * coeffs


def build_model(zred=None, waverange=None, add_duste=True, add_neb=False, add_agn=False, fit_afe=False,
                polyorder=10, 
                **extras):
    """Build a prospect.models.SedModel object

    :param zred: (optional, default: None)
        approximate value for the redshift, which is left as a free parameter.

    :param waverange: (optional, default: None)
         rest-frame wavelength range in angstrom; used to calculate polyorder.

    :returns model:
        An instance of prospect.models.SedModel
    """    
    
    # continuity SFH
    model_params = TemplateLibrary["continuity_sfh"]
    
    model_params = {}
    
    if zred is None:
        raise ValueError('zred must be specified')
    
    # Basic parameters
    model_params['zred'] = {"N": 1, "isfree": True, 
                            "init": zred, 
                            "units": "redshift",
                           "prior": priors.Normal(mean=zred, sigma=0.005)}
    
    model_params['logzsol'] = {"N": 1, "isfree": True, 
                               "init": -0.5, 
                              "units": r"$\log (Z/Z_\odot)$",
                              "prior": priors.TopHat(mini=-2, maxi=0.50)}
    
    if fit_afe:
        model_params['afe'] = {"N": 1, "isfree": True, 
                               "init": 0.0,
                              "units": r"$[\alpha/fe]$",
                              "prior": priors.TopHat(mini=-0.2, maxi=0.6)}
    else:
        model_params['afe'] = {"N": 1, "isfree": False,
                               "init": 0.0,
                               "units": r"$[\alpha/fe]$"}
    
    model_params["logt_wmb_hot"] = dict(N=1, isfree=False, init=10.0)

    # velocity dispersion
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=50.0, maxi=400.0)
    
    # This removes the continuum from the spectroscopy. Highly recommend
    # using when modeling both photometry & spectroscopy
    model_params.update(TemplateLibrary['optimize_speccal'])
    model_params['spec_norm']['isfree'] = False
    model_params['polyorder']['init']   = polyorder
    
    
    # This is a pixel outlier model. It helps to marginalize over
    # poorly modeled noise, such as residual sky lines or
    # even missing absorption lines
    model_params['f_outlier_spec'] = {"N": 1,
                                      "isfree": True,
                                      "init": 0.01,
                                      "prior": priors.TopHat(mini=1e-5, maxi=0.2)}
    
    model_params['nsigma_outlier_spec'] = {"N": 1,
                                          "isfree": False,
                                          "init": 50.0}
    
    model_params['f_outlier_phot'] = {"N": 1,
                                      "isfree": True,
                                      "init": 0.00,
                                      "prior": priors.TopHat(mini=0, maxi=0.5)}
    
    model_params['nsigma_outlier_phot'] = {"N": 1,
                                          "isfree": False,
                                          "init": 50.0}
    
    
    # This is a multiplicative noise inflation term. It inflates the noise in
    # all spectroscopic pixels as necessary to get a good fit.
    model_params['spec_jitter'] = {"N": 1,
                                   "isfree": True,
                                   "init": 1.0,
                                   "prior": priors.TopHat(mini=0.5, maxi=15.0)}

    
    # ----------------------------
    # --- Continuity SFH ----
    # ----------------------------
    # A non-parametric SFH model of mass in fixed time bins with a smoothness prior
    
    tuniv = cosmo.age(zred).value*1e6
    #agelims_Myr = np.append( np.logspace( np.log10(30.0), np.log10(0.95*tuniv*1000), 13), tuniv*1000 )
    agelims_Myr = np.append( np.logspace( np.log10(30.0), np.log10(0.8*tuniv*1000), 12), [0.9*tuniv*1000, tuniv*1000])
    agelims = np.concatenate( ( [0.0], np.log10(agelims_Myr*1e6) ))
    print(agelims)
    agebins = np.array([agelims[:-1], agelims[1:]]).T
    nbins = len(agelims) - 1

    """
    # Verification (optional - can remove in production)
    linear_bins = 10**agebins
    min_spacing_check = np.min(np.diff(linear_bins, axis=1))
    if min_spacing_check < 1e6:
        print(f"Warning: Minimum spacing {min_spacing_check/1e6:.2f} Myr still below 1 Myr at z={zred:.2f}")

    print(f"{'Bin':>3} | {'Start':>12} → {'End':>12} [yr] | Width: [Δt] [yr]")
    print("-" * 55)

    for i, (logt0, logt1) in enumerate(agebins):
        t0 = 10**logt0
        t1 = 10**logt1
        width = t1 - t0
        print(f"{i:>3} | {t0:12.3e} → {t1:12.3e} | Δt = {width: .3e}")
    """
    
    
    # This is the *total*  mass formed, as a variable
    model_params["logmass"] = {"N": 1, "isfree": True, 
                               "init": 10.5,
                               'units': "Solar masses formed",
                               'prior': priors.TopHat(mini=8.5, maxi=13)}
    
    # This will be the mass in each bin.  It depends on other free and fixed
    # parameters.  Its length needs to be modified based on the number of bins
    model_params["mass"] = {'N': nbins, 'isfree': False, 
                            'init': (10**10.5)/nbins,
                            'units': "Solar masses formed",
                            'depends_on': transforms.logsfr_ratios_to_masses}
    
    # This gives the start and stop of each age bin.  It can be adjusted and its
    # length must match the length of "mass"
    model_params["agebins"] = {'N': nbins, 'isfree': False, 
                               'init': agebins,
                               'units': 'log(yr)'}
    
    # This controls the distribution of SFR(t) / SFR(t+dt). It has nbins-1 components.
    model_params["logsfr_ratios"] = {'N': nbins-1, 'isfree': True,
                                    'init': np.full(nbins-1, 0.0),  # constant SFH
                                    'units': '',
                                    'prior': priors.StudentT(mean=np.full(nbins-1, 0.0),
                                                           scale=np.full(nbins-1, 0.3), 
                                                           df=np.full(nbins-1, 2))}
    
    # ------------------------------
    # --- Initial Mass Function  ---
    # ------------------------------

    model_params['imf_type'] = {'N': 1, 'isfree': False,
                             'init': 1, #1 = chabrier
                             'units': "FSPS index",
                             'prior': None}


    # ----------------------------
    # --- Dust Absorption ---
    # ----------------------------
    
    model_params['dust_type'] = {"N": 1, "isfree": False, 
                                 "init": 4,
                                 "units": "FSPS index"}
    
    model_params['dust2'] = {"N": 1, "isfree": True, 
                             "init": 0.5,
                             "units": "dust optical depth at 5500A",
                             "prior": priors.TopHat(mini=0.0, maxi=4.0/1.086)}
    
    model_params["dust_index"] = {"N": 1,
                                 "isfree": True,
                                 "init": 0.0, "units": "power-law multiplication of Calzetti",
                                 "prior": priors.ClippedNormal(mini=-1.5, maxi=0.4, mean=0.0, sigma=0.3)}

    model_params['dust1'] = {"N": 1,
                             "isfree": False,
                             'depends_on': to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars",
                             "prior": None}

    model_params['dust1_fraction'] = {'N': 1,
                                      'isfree': True,
                                      'init': 1.0,
                                      'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}    

    # ----------------------------
    # --- Dust Emission ---
    # ----------------------------    
    
    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_gamma']['isfree'] = True
        model_params['duste_gamma']['init']  = 0.01
        model_params['duste_gamma']['prior'] = priors.TopHat(mini=0.0, maxi=1.0)
        
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_qpah']['init']   = 3.5
        model_params['duste_qpah']['prior']  = priors.TopHat(mini=0.5, maxi=10.0)
        
        model_params['duste_umin']['isfree'] = True
        model_params['duste_umin']['init']   = 1.0
        model_params['duste_umin']['prior']  = priors.TopHat(mini=0.1, maxi=25.0)

    if add_agn:
        # Allow for the presence of an AGN in the mid-infrared
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = True
        model_params['fagn']['prior'] = priors.LogUniform(mini=1e-5, maxi=3.0)
        model_params['agn_tau']['isfree'] = True
        model_params['agn_tau']['prior'] = priors.LogUniform(mini=5.0, maxi=150.)

    if add_neb:
        # Add nebular emission
        model_params.update(TemplateLibrary["nebular"])
        model_params['gas_logu']['isfree'] = True
        model_params['gas_logz']['isfree'] = True
        model_params['nebemlineinspec'] = {'N': 1,
                                           'isfree': False,
                                           'init': False}
        _ = model_params["gas_logz"].pop("depends_on")

        model_params.update(TemplateLibrary['nebular_marginalization'])

        model_params['use_eline_prior'] = {'N': 1, 'is_free': False, 'init': False}
        
        model_params['eline_sigma'] = {"N": 1,
                                    "isfree": True,
                                    "init": 100.0,
                                    "prior": priors.TopHat(mini=30, maxi=400)}
        
        
    # ----------------------------
    # ----------------------------
    
    # Now instantiate the model object using this dictionary of parameter specifications
    model = PolySpecModel(model_params)
    
    return model




def plot_reconstruction_with_miri(objid, output_dir=None):
    """Main function to reconstruct and plot PROSPECTOR results with MIRI data"""
    
    # Load the h5 file for the given objid
    h5_file = glob.glob(os.path.join(dirout, f"output_{objid}*.h5"))[0] 
    
    # Load PROSPECTOR results
    full_path = os.path.join(dirout, h5_file)
    results, _, _ = reader.results_from(full_path)
    
    # Get redshift and MAP parameters
    zred = get_zred(objid)
    print(f"GALAXY {objid}:  z = {zred}")

    # Build new observations including MIRI
    obs = build_obs(objid)
    
    # Build model matching original setup
    
    model = build_model(objid, zred, add_duste=results['run_params']['add_duste'],
                    add_neb=False,  # Set add_neb to False
                    add_agn=results['run_params']['add_agn'],
                    fit_afe=results['run_params']['fit_afe'])
    
    model.params['polyorder'] = 25
    
    # Now we have to exclude the last 3 parameters from the fit
    map_parameters = get_MAP(results)
    map_parameters = map_parameters[:-3]
    
    # Calculate the spectrum based on the Maximum A Posteriori (MAP) parameters
    sps = FastStepBasis(zcontinuous=1)
            
    spec, phot, _ = model.predict(map_parameters, obs=obs, sps=sps)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Constants
    c = const.c.to(u.AA/u.s).value  # Speed of light in AA/s
    factor = 3631e6 # maggies to µJy conversion factor

    spec *= factor    # convert to µJy
    # Redshifted model spectrum
    wave_spec = sps.wavelengths * (1 + zred) * 1e-4  # now in observer-frame [µm]

    # Define the style per instrument
    instrument_styles = {
        'acs':     {'color': 'royalblue',   'marker': 'o', 'edgecolor': 'black', 'label': 'HST ACS'},
        'wfc3':    {'color': 'limegreen',  'marker': 'o', 'edgecolor': 'black', 'label': 'HST WFC3'},
        'nircam':  {'color': 'orange', 'marker': 'p', 'edgecolor': 'black',    'alpha': 0.7, 'label': 'JWST NIRCam'},
        'miri':    {'color': 'firebrick',    'marker': 'p', 'edgecolor': 'black',    'alpha': 0.7, 'label': 'JWST MIRI'}
    }

    """
    # Apply HST correction BEFORE the plotting loop
    hst_flux_correction = 0.74  # 100% - 26% = 74%
    corrected_fluxes = obs['maggies_all'].copy()
    corrected_errors = obs['maggies_unc_all'].copy()

    for i, filt in enumerate(obs['filters_all']):
        fname = filt.name.lower()
        if 'acs' in fname or 'wfc3' in fname:
            # Apply 26% downward correction to flux
            corrected_fluxes[i] *= hst_flux_correction

            # Add 26% relative error in quadrature
            relative_error = 0.26 * corrected_fluxes[i]
            corrected_errors[i] = np.sqrt(corrected_errors[i]**2 + relative_error**2)

    # Update obs dictionary
    obs['maggies_all'] = corrected_fluxes
    obs['maggies_unc_all'] = corrected_errors
    """

    # Loop through each filter and plot
    for i, filt in enumerate(obs['filters_all']):
        name = filt.name.lower()

        if 'acs' in name:
            style = instrument_styles['acs']
        elif 'wfc3' in name:
            style = instrument_styles['wfc3']
        elif 'miri' in name or any(m in name for m in ['f770w', 'f1000w', 'f1800w', 'f2100w']):
            style = instrument_styles['miri']
        elif 'nircam' in name or ('jwst' in name and 'f' in name and 'w' in name):
            style = instrument_styles['nircam']
        else:
            continue  # skip unknown filters

        wave = obs['phot_wave_all'][i] * 1e-4  # convert to µm
        flux = obs['maggies_all'][i] * factor  # µJy
        err  = obs['maggies_unc_all'][i] * factor  # µJy

        ax.errorbar(
            wave, flux, yerr=err,
            fmt=style['marker'],
            color=style['color'],
            markeredgecolor=style.get('edgecolor', 'none'),
            alpha=style.get('alpha', 1.0),
            markersize=8,
            label=style['label'] if style['label'] not in ax.get_legend_handles_labels()[1] else None
        )



    # Model photometry (used in fit)
    #ax1.plot(wave_model, phot, 'k*', markersize=9, label='Model prediction')

    # Full SED
    ax.plot(wave_spec, spec, '-', color='crimson', alpha=0.6, lw=1.5, label='Best-fit model')
    """
    import random

    # Number of posterior samples to use
    nsample = 10
    samples = random.sample(range(results['chain'].shape[0]), nsample)

    # Store predicted spectra
    specs = []

    for i in samples:
        theta = results['chain'][i]
        spec_i, _, _ = model.predict(theta, obs=obs, sps=sps)
        specs.append(spec_i)

    
    # Convert to array: shape (nsample, nwave)
    specs = np.array(specs)

    # Compute 16th and 84th percentile envelope
    spec_lo = np.percentile(specs, 16, axis=0)
    spec_hi = np.percentile(specs, 84, axis=0)
    spec_med = np.percentile(specs, 50, axis=0)  # Optional: median spectrum
    
    ax1.plot(model.wavelengths, spec_med, label='Median Spectrum', color='green')
    ax1.fill_between(model.wavelengths, spec_lo, spec_hi, color='green', alpha=0.3, label='1σ range')
    """
    
    # Compute bounds
    ymin = np.min(spec)
    ymax = np.max(spec)

    # Add 10% margin
    yrange = ymax - ymin
    ymin_plot = max(ymin - 0.1 * yrange, 1e-2)  # prevent negative or log(0)
    ymax_plot = ymax + 0.1 * yrange

    # Set limits
    ax.set_ylim(ymin_plot, ymax_plot)
    
    # Plot formatting
    ax.set_xlabel('Observed Wavelength (µm)')
    ax.set_ylabel('Flux (µJy)')
    ax.set_xlim(0.4, 35)    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{objid}_miri_overlay.png'))
    plt.show()
    
    return obs, model, results
