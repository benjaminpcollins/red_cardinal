import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from sedpy.observate import load_filters, list_available_filters

import prospect.io.read_results as reader
from prospect.models import priors
from prospect.models.templates import TemplateLibrary
from prospect.models import transforms
from astropy.cosmology import WMAP9 as cosmo
from prospect.models.sedmodel import PolySpecModel, SpecModel
from prospect.utils.obsutils import fix_obs

from scipy.signal import medfilt
from scipy import interpolate

from sedpy.observate import getSED


def get_model_photometry(spec, wave_spec, filters, zred):
    # 1) maggies -> f_nu_obs [erg/s/cm^2/Hz]
    # We assume 'spec' is already the flux you see at Earth.
    fnu_obs = spec * 3631e-23
    
    # 2) Wavelengths -> Observer-frame Angstroms
    lam_obs_A = wave_spec * (1.0 + zred)
    
    # 3) f_nu_obs -> f_lambda_obs [erg/s/cm^2/Å]
    # We use observer-frame wavelengths for the conversion.
    wave_obs_cm = lam_obs_A * 1e-8
    flam_obs = fnu_obs * (const.c.cgs.value / wave_obs_cm**2) / 1e8
    
    # 4) The Key Step: Pass Observer-frame wavelengths to getSED
    # This ensures the filter transmission curves align with the redshifted spectrum.
    phot_new = getSED(lam_obs_A, flam_obs, filterlist=filters, linear_flux=True)

    return phot_new

def update_obs_with_miri(objid, obs, phot_table_miri):
    """Add only VALID MIRI photometry to obs."""
    
    # -------------------------------------------------
    # 1. Load MIRI catalogue row
    # -------------------------------------------------
    with fits.open(phot_table_miri) as hdul:
        data = hdul[1].data
        row = data[data['ID'] == objid]

    if len(row) == 0:
        print(f"No MIRI data for {objid}")
        return obs

    row = row[0]

    filter_dict_miri = {
        'F770W':  'jwst_f770w',
        'F1000W': 'jwst_f1000w',
        'F1800W': 'jwst_f1800w',
        'F2100W': 'jwst_f2100w'
    }

    # -------------------------------------------------
    # 2. Collect VALID MIRI bands
    # -------------------------------------------------
    valid_filters = []
    valid_fluxes  = []
    valid_errors  = []
    valid_codes   = []

    for code, sedpy_name in filter_dict_miri.items():

        flux = row[f"{code}_flux"]
        err  = row[f"{code}_flux_err"]

        if np.isfinite(flux) and np.isfinite(err) and err > 0:
            valid_fluxes.append(flux / 3631)      # Jy → maggies
            valid_errors.append(err / 3631)
            valid_filters.append(sedpy_name)
            valid_codes.append(code)

    if len(valid_fluxes) == 0:
        print("No valid MIRI bands.")
        return obs

    # -------------------------------------------------
    # 3. Extend obs consistently
    # -------------------------------------------------
    obs_new = obs.copy()

    # concatenate maggies
    obs_new['maggies_all'] = np.concatenate([obs['maggies'], valid_fluxes])
    obs_new['maggies_unc_all'] = np.concatenate([obs['maggies_unc'], valid_errors])

    # concatenate filters
    miri_filters = load_filters(valid_filters)

    obs_new['filters_all'] = obs['filters'] + miri_filters
    obs_new['filters_miri'] = miri_filters
    obs_new['filter_code_all'] = obs['filter_code'] + valid_codes

    # wavelengths
    obs_new['phot_wave_all'] = np.array(
        [f.wave_effective for f in obs_new['filters_all']]
    )
    
    # mask
    obs_new['valid_mask_all'] = (
        np.isfinite(obs_new['maggies_all']) &
        np.isfinite(obs_new['maggies_unc_all']) &
        (obs_new['maggies_unc_all'] > 0)
    )

    return obs_new


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



def build_obs(objid, phot_table_miri):
    """Build observation dictionary with photometry (no spectroscopy)"""
    
    # Load photometry data
    phot_hst_nircam = fits.open(bluejay)[1].data
    ph_hst_nircam = phot_hst_nircam[phot_hst_nircam['ID'] == objid]
    
    phot_miri = fits.open(phot_table_miri)[1].data
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
    filter_code_orig = list(filter_dict_3dhst.keys()) + list(filter_dict_nircam.keys())
    filter_name_orig = list(filter_dict_3dhst.values()) + list(filter_dict_nircam.values())
    
    # Filters present in MIRI
    filter_code_miri = list(filter_dict_miri.keys())
    filter_name_miri = list(filter_dict_miri.values())
    
    # All filters including MIRI
    filter_code_all = filter_code_orig + filter_code_miri
    filter_name_all = filter_name_orig + filter_name_miri
    
    # Modify obs dictionary
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

    # Combine all fluxes    
    fluxes_all = np.copy(fluxes)
    fluxes_err_all = np.copy(fluxes_err)
        
    # Dynamically add miri fluxes and errors to the existing arrays
    for ff, fil in enumerate(filter_code_miri):
        fluxes_all = np.append(fluxes_all, ph_miri[str(fil) + '_flux'][0])
        fluxes_err_all = np.append(fluxes_err_all, ph_miri[str(fil) + '_flux_err'][0])
    
    print("MIRI fluxes added successfully.")
    
    # Add 5% systematic error in quadrature to all bands
    fluxes_err = np.sqrt(fluxes_err**2 + (0.05*fluxes)**2)
    fluxes_err_all = np.sqrt(fluxes_err_all**2 + (0.05*fluxes_all)**2)
    
    # Convert to maggies
    obs['maggies'] = np.array(fluxes) / 3631
    obs['maggies_unc'] = np.array(fluxes_err) / 3631
    obs['maggies_all'] = np.array(fluxes_all) / 3631
    obs['maggies_unc_all'] = np.array(fluxes_err_all) / 3631
    
    # Build masks (HST+NIRCam only vs all including MIRI)
    valid = np.isfinite(obs['maggies']) & np.isfinite(obs['maggies_unc']) & \
                        (obs['maggies'] > 0) & (obs['maggies_unc'] > 0)

    valid_all = np.isfinite(obs['maggies_all']) & np.isfinite(obs['maggies_unc_all']) & \
                        (obs['maggies_unc_all'] > 0) # & (obs['maggies_all'] > 0) allow for negative fluxes
    
    obs['valid_mask'] = valid
    obs['valid_mask_all'] = valid_all
    
    # Filter out invalid entries everywhere
    obs['maggies']     = obs['maggies'][valid]
    obs['maggies_unc'] = obs['maggies_unc'][valid]
    obs['phot_wave']   = np.array([f.wave_effective for f, m in zip(obs['filters'], valid) if m])
    obs['filters']     = [f for f, m in zip(obs['filters'], valid) if m]

    for filt, mag in zip(obs['filters_all'], obs['maggies_all']):
        print(filt, mag)
    
    obs['maggies_all']     = obs['maggies_all'][valid_all]
    obs['maggies_unc_all'] = obs['maggies_unc_all'][valid_all]
    obs['phot_wave_all']   = np.array([f.wave_effective for f, m in zip(obs['filters_all'], valid_all) if m])
    obs['filters_all']     = [f for f, m in zip(obs['filters_all'], valid_all) if m]
    
    if len(obs['filters']) == len(obs['filters_all']): print("⚠️ Attention: No valid Miri data found!")
    
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
    """Build a prospect.models.PolySpecModel object

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
    
    tuniv = cosmo.age(zred).value#*1e6
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
        print("Adding dust emission...")
        
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
    else:
        print("Dust emission is turned off.")
        
    if add_agn:
        # Allow for the presence of an AGN in the mid-infrared
        print("Adding AGN emission...")
        
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = True
        model_params['fagn']['prior'] = priors.LogUniform(mini=1e-5, maxi=3.0)
        model_params['agn_tau']['isfree'] = True
        model_params['agn_tau']['prior'] = priors.LogUniform(mini=5.0, maxi=150.)

    if add_neb:
        # Add nebular emission
        print("Adding nebular emission...")
        
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


def rebuild_model(zred=None, waverange=None, add_duste=True, add_neb=False, add_agn=False, fit_afe=False,
                polyorder=10, 
                **extras):
    """
    Reconstructs the PolySpecModel with live functions (callables)
    to avoid pickling/AssertionErrors.
    """

    model_params = {}
    
    if zred is None:
        raise ValueError('zred must be specified to reconstruct the SFH agebins.')
    
    # 1. Basic Parameters
    model_params['zred'] = {"N": 1, "isfree": True, "init": zred, 
                            "prior": priors.Normal(mean=zred, sigma=0.005)}
    model_params['logzsol'] = {"N": 1, "isfree": True, "init": -0.5, 
                               "prior": priors.TopHat(mini=-2, maxi=0.5)}
    model_params['afe'] = {"N": 1, "isfree": False, "init": 0.0}
    model_params["logt_wmb_hot"] = dict(N=1, isfree=False, init=10.0)

    # 2. Spectral Smoothing & Calibration
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=50.0, maxi=400.0)
    
    model_params.update(TemplateLibrary['optimize_speccal'])
    model_params['polyorder']['init'] = polyorder
    model_params['spec_norm']['isfree'] = False
    
    # 3. Outlier Models
    model_params['f_outlier_spec'] = {"N": 1, "isfree": True, "init": 0.01, 
                                      "prior": priors.TopHat(mini=1e-5, maxi=0.2)}
    model_params['f_outlier_phot'] = {"N": 1, "isfree": True, "init": 0.00, 
                                      "prior": priors.TopHat(mini=0, maxi=0.5)}
    model_params['nsigma_outlier_spec'] = {"N": 1, "isfree": False, "init": 50.0}
    model_params['nsigma_outlier_phot'] = {"N": 1, "isfree": False, "init": 50.0}
    model_params['spec_jitter'] = {"N": 1, "isfree": True, "init": 1.0, 
                                   "prior": priors.TopHat(mini=0.5, maxi=15.0)}

    # 4. Continuity SFH (Critical: Agebins must scale with zred)
    tuniv = cosmo.age(zred).value
    agelims_Myr = np.append(np.logspace(np.log10(30.0), np.log10(0.8*tuniv*1000), 12), 
                            [0.9*tuniv*1000, tuniv*1000])
    agelims = np.concatenate(([0.0], np.log10(agelims_Myr*1e6)))
    agebins = np.array([agelims[:-1], agelims[1:]]).T
    nbins = len(agelims) - 1
    
    model_params["logmass"] = {"N": 1, "isfree": True, "init": 10.0, 
                               "prior": priors.TopHat(mini=8.0, maxi=13)}
    model_params["agebins"] = {'N': nbins, 'isfree': False, 'init': agebins}
    model_params["mass"] = {'N': nbins, 'isfree': False, 'init': 1e10/nbins,
                            'depends_on': transforms.logsfr_ratios_to_masses}
    model_params["logsfr_ratios"] = {'N': nbins-1, 'isfree': True, 
                                     'init': np.zeros(nbins-1),
                                     'prior': priors.StudentT(mean=np.zeros(nbins-1), 
                                                              scale=np.full(nbins-1, 0.3), 
                                                              df=np.full(nbins-1, 2))}

    model_params['dust_type'] = {"N": 1, "isfree": False, "init": 4}
    model_params['dust2'] = {"N": 1, "isfree": True, "init": 0.5, 
                             "prior": priors.TopHat(mini=0.0, maxi=4.0)}
    model_params["dust_index"] = {"N": 1, "isfree": True, "init": 0.0, 
                                  "prior": priors.ClippedNormal(mini=-1.5, maxi=0.4, mean=0.0, sigma=0.3)}
    model_params['dust1_fraction'] = {'N': 1, 'isfree': True, 'init': 1.0, 
                                      'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}
    model_params['dust1'] = {"N": 1, "isfree": False, "depends_on": to_dust1, "init": 0.0}

    if add_duste:
        # 5. Dust and Emission
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_gamma']['isfree'] = True
        #model_params['duste_gamma']['init']  = 0.01
        model_params['duste_gamma']['prior'] = priors.TopHat(mini=0.0, maxi=1.0)
        
        model_params['duste_qpah']['isfree'] = True
        #model_params['duste_qpah']['init']   = 3.5
        model_params['duste_qpah']['prior']  = priors.TopHat(mini=0.5, maxi=10.0)
        
        model_params['duste_umin']['isfree'] = True
        #model_params['duste_umin']['init']   = 1.0
        model_params['duste_umin']['prior']  = priors.TopHat(mini=0.1, maxi=25.0)

    if add_neb:
        model_params.update(TemplateLibrary["nebular"])
        model_params['gas_logz']['isfree'] = True
        model_params['gas_logu']['isfree'] = True
        
        # This is likely the missing link for your eline_sigma error
        model_params.update(TemplateLibrary['nebular_marginalization'])
        model_params['eline_sigma']['isfree'] = True 
        model_params['eline_sigma']['prior'] = priors.TopHat(mini=30, maxi=400)

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
    normed_spec = obs['spectrum'][mask].byteswap().newbyteorder()
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
