import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from sedpy.observate import load_filters, list_available_filters
import fsps
import traceback
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor
import h5py

from astropy.table import Table
import prospect.io.read_results as reader
from prospect.sources import FastStepBasis
from prospect.models import priors
from prospect.models.templates import TemplateLibrary
from prospect.models import transforms
from astropy.cosmology import WMAP9 as cosmo
from prospect.models.sedmodel import PolySpecModel
from prospect.utils.obsutils import fix_obs
from collections import defaultdict

from scipy.signal import medfilt
from scipy import interpolate


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

def get_MAP(res, verbose=False):
    """
    Get Maximum A Posteriori parameters
    chains = results['chain']   # Shape: (n_samples, n_parameters)
    log_probabilities = results['lnprobability']  # Shape: (n_samples,)
    
    # Find the index of the maximum log probability
    max_prob_index = np.argmax(log_probabilities)
    
    # Get the MAP parameters
    map_parameters = chains[max_prob_index]

    if verbose:
        print("MAP Parameters:", map_parameters)

    return map_parameters
    """
    #Get the posterior sample with the highest posterior probability.
    
    imax = np.argmax(res['lnprobability'])
    # there must be a more elegant way to deal with differnt shapes
    try:
        i, j = np.unravel_index(imax, res['lnprobability'].shape)
        theta_best = res['chain'][i, j, :].copy()
    except(ValueError):
        theta_best = res['chain'][imax, :].copy()
    return theta_best

def build_obs(objid, obs):
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
    
    
    # Modify existing obs dictionary
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
    #obs['spectrum'] = old_obs['spectrum']  # Use the original spectrum
    #obs['wavelength'] = old_obs['wavelength']  # Use the original wavelength
    #obs['unc'] = old_obs['unc']  # Use the original uncertainties
    #print(obs.keys())
    
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
    agebins = np.array([agelims[:-1], agelims[1:]]).T
    nbins = len(agelims) - 1
    
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


def mask_obs(obs, zred):
    # mask emission lines in the wavelength range

    # import lines to mask
    emi_wavelengths = [3723, 3726, 3729, 3835, 3889, 3970,  
                            4102.89, 4341.69, 4364, 4472, 
                            4621, 4720, 4960.30, 5008.24, 
                            6549.86, 6585.27]

    # mask emission lines
    mask = obs['mask']
    for w_emi in emi_wavelengths:
        linemask = abs((obs['wavelength']/(1+zred) - w_emi)/w_emi) < 800/3e5
        mask = np.logical_and(mask,~linemask)

    # mask bad pixels
    normed_spec = obs['spectrum'][mask]
    kernel = 201
    f_cont_masked = medfilt(normed_spec, kernel)
    f_cont_intepolator = interpolate.interp1d(obs['wavelength'][mask], f_cont_masked, kind='linear', fill_value='extrapolate')
    f_cont = f_cont_intepolator(obs['wavelength'])

    # residuals
    std = np.std(obs['spectrum']-f_cont)
    sigma5_lvl =  np.zeros((len(obs['spectrum']),)) + 5*std
    mask2 = np.full_like(obs['mask'],True)
    for k in range(len(obs['wavelength'])):
        if (obs['mask'][k]) & (np.abs((obs['spectrum']-f_cont)[k]) > sigma5_lvl[k]):
            for j in range(k-2,k+3):
                if j < len(obs['wavelength']):
                    mask2[j] = False

    mask = np.logical_and(mask2, mask)

    return mask



def plot_reconstruction_with_miri(objid, output_dir=None):
    """Main function to reconstruct and plot PROSPECTOR results with MIRI data"""
    
    # Load the h5 file for the given objid
    h5_file = glob.glob(os.path.join(dirout, f"output_{objid}*.h5"))
    
    try:
        h5_file = h5_file[0]
    except IndexError:
        print(f"No PROSPECTOR results found for objid {objid}.")
        return None, None, None

    # Load PROSPECTOR results
    full_path = os.path.join(dirout, h5_file)
    results, obs, model = reader.results_from(full_path)
    
    #print("obs: ", obs)
    # Get redshift and MAP parameters
    zred = get_zred(objid)
    print(f"GALAXY {objid}:  z = {zred}")

    # Build new observations including MIRI
    #obs = build_obs(objid, results['obs'])
    
    # IMPORTANT! mask emission lines and bad pixels
    #obs['mask'] = mask_obs(obs, zred)
    
    #obs = results['obs']
    #print(obs.keys())
    
    # Build model matching original setup
    model = build_model(objid, zred, add_duste=results['run_params']['add_duste'],
                    add_neb=False,  # Set add_neb to False
                    add_agn=results['run_params']['add_agn'],
                    fit_afe=results['run_params']['fit_afe'])
    
    model.params['polyorder'] = 10
        
    # Now we have to exclude the last 3 parameters from the fit
    map_parameters = get_MAP(results)
    map_parameters = map_parameters[:-3]
    
    
    for a, b in zip(model.theta_labels(), map_parameters):
        print(f"{a}: {b}")
    
    
    #params_file = '/Users/benjamincollins/University/master/Red_Cardinal/prospector/params/params_MAP_12717.pkl'
    #params_dict = pkl.load(open(params_file, 'rb'))
    
    # Ensure this gives you the correct order expected by the model
    #param_order = model.theta_labels()
    #print(model.theta_labels())
    
    # Extract the values from the dictionary in this order
    #theta_map = [params_dict[key] for key in param_order]

    #print("\nMAP parameters loaded from file:", theta_map)
    
    #zred_used = map_parameters['zred']
    #print("\nRedshift used for prediction:", zred_used)

    # Calculate the spectrum based on the Maximum A Posteriori (MAP) parameters
    sps = FastStepBasis(zcontinuous=1)
        
    spec, phot, _ = model.predict(map_parameters, obs=obs, sps=sps)
    
    calib_vector = model._speccal

    #get the spectral jitter  for the errors
    #spec_jitter = map_parameters[5]
    #errs = obs['unc']*spec_jitter/calib_vector * 3631 # observational errors
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    factor = 3631e6 # maggies to µJy conversion factor
    
    spec = spec/calib_vector * factor    # convert to µJy
    
    # Redshifted model spectrum
    wave_spec = sps.wavelengths * 1e-4 * (1 + zred)   # convert to µm, redshifted
    
    # Define the style per instrument
    instrument_styles = {
        'acs':     {'color': 'royalblue',   'marker': 'o', 'edgecolor': 'black', 'label': 'HST ACS', 'ms': 10},
        'wfc3':    {'color': 'limegreen',  'marker': 'o', 'edgecolor': 'black', 'label': 'HST WFC3', 'ms': 10},
        'nircam':  {'color': 'orange', 'marker': 'p', 'edgecolor': 'black',    'alpha': 0.7, 'label': 'JWST NIRCam', 'ms': 10},
        'miri':    {'color': 'firebrick',    'marker': 'p', 'edgecolor': 'black',    'alpha': 0.7, 'label': 'JWST MIRI', 'ms': 10}
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
    for i, filt in enumerate(obs['filters']):
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

        wave = obs['phot_wave'][i] * 1e-4  # convert to µm
        flux = obs['maggies'][i] * factor  # µJy
        err  = obs['maggies_unc'][i] * factor  # µJy

        ax.errorbar(
            wave, flux, yerr=err,
            fmt=style['marker'],
            color=style['color'],
            markeredgecolor=style.get('edgecolor', 'none'),
            alpha=style.get('alpha', 1.0),
            markersize=10,
            label=style['label'] if style['label'] not in ax.get_legend_handles_labels()[1] else None
        )

    # Model photometry (used in fit)
    wave_model = np.array([filt.wave_effective for filt in obs['filters']]) * 1e-4  # µm
    phot *= factor  # µJy
    ax.plot(wave_model, phot, 'd', markersize=6, color='black', label='Model photometry')
    
    
    # Observed spectrum
    #obs_file = f'/Users/benjamincollins/University/master/Red_Cardinal/prospector/obs/obs_{objid}.npy'
    #loaded_obs = np.load(obs_file, allow_pickle=True).item()
    
    loaded_obs = obs

    spec_new, _, _ = model.predict(map_parameters, obs=loaded_obs, sps=sps)
    spec_new = spec
    
    calib_vector = model.spec_calibration(obs=loaded_obs, spec=spec_new)
    mask = loaded_obs['mask']
    print("Calibration vector:", calib_vector)
    
    wavelengths = loaded_obs['wavelength'][mask] * 1e-4  # convert to µm
    spectrum  = loaded_obs['spectrum'][mask] / calib_vector[mask]
    
    ax.plot(wavelengths, spectrum, label="Observed Spectrum", color='royalblue', alpha=0.7)
    
    # Full SED
    ax.plot(obs['wavelength']*1e-4, spec, '-', color='crimson', alpha=0.6, lw=1.5, label='Best-fit model')

    """
    # Number of posterior samples to use
    nsample = 200
    samples = random.sample(range(results['chain'].shape[0]), nsample)
    
    # Store predicted spectra
    specs = []
    chain = np.asarray(results['chain'])    # Read as array for easier indexing
    
    for i in samples:
        theta = chain[i]
        spec_i, _, _ = model.predict(theta[:-3], obs=obs, sps=sps)
        specs.append(spec_i)

    # Convert to array: shape (nsample, nwave)
    specs = np.array(specs)

    # Compute 16th and 84th percentile envelope
    spec_lo = np.percentile(specs, 16, axis=0)
    spec_hi = np.percentile(specs, 84, axis=0)
    
    spec_lo *= factor  # Convert to µJy
    spec_hi *= factor  # Convert to µJy
    spec_med = np.percentile(specs, 50, axis=0)  # Optional: median spectrum
    
    ax.fill_between(wave_spec, spec_lo, spec_hi, 
                    color='firebrick', alpha=0.1, label='1σ range')
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
    
    output_dir = '/Users/benjamincollins/University/Master/Red_Cardinal/prospector/fits_plus_miri/'
    
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f'{objid}_miri_overlay_v3.png')
        plt.savefig(fname)
        print(f"Saved plot to {fname}")
    plt.show()
    plt.close()
    
    return obs, model, results

table_path = '/Users/benjamincollins/University/Master/Red_Cardinal/photometry/phot_tables/Photometry_Table_MIRI_v6.fits'

table = Table.read(table_path, format='fits')
galaxy_ids = [str(gid) for gid in table['ID']]    


#if __name__ == "__main__":
#    max_workers = min(6, os.cpu_count())
#    with ProcessPoolExecutor(max_workers=max_workers) as executor:
#        executor.map(plot_reconstruction_with_miri, galaxy_ids)


# To Do:
# - Multiply observed spectrum by the mask
# - Add spectrum after prediction
# - 