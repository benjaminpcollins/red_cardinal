import time
import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate
from scipy.signal import medfilt

from astropy.cosmology import WMAP9 as cosmo
from astropy import table
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
import sedpy

import prospect
from prospect.utils.obsutils import fix_obs
from prospect.sources import FastStepBasis
from prospect.models import transforms
from prospect.models.sedmodel import SedModel, PolySpecModel
from prospect.models import priors
from prospect.models.templates import TemplateLibrary
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated
#from prospfig import make_figure
#from prosparams_mask import mask_obs

# set the data paths
cosmos_3dhst_cat = '/Users/letiziabugiani/Desktop/bluejay/photometry/data/cosmos_3dhst.v4.1.cats/Catalog/cosmos_3dhst.v4.1.cat'
# ----------------------------------------------------------------------------
# Set up obs for UltraVISTA photometry
# ----------------------------------------------------------------------------

dir_new_photometry = '/Users/letiziabugiani/Desktop/bluejay/photometry/data/bluejay_phot/v1.3/'

def find_zred(objid):
    dat_zred = np.loadtxt('/Users/letiziabugiani/Desktop/bluejay/spectroscopy/redshifts_v0.1.txt', dtype=[('galid', int),('zred', '<f8')])
    return abs(dat_zred['zred'][dat_zred['galid']==int(objid)][0])

def flambda_to_maggies(wave_AA, flux):

    flux = flux * 1e-20 * u.erg/u.s/u.AA/u.cm**2
    fnu = flux * (wave_AA*u.AA)**2 / const.c
    fnu_Jy = fnu.to(u.Jy)
    fnu_maggies = fnu_Jy / 3631

    return fnu_maggies.value

def build_obs(objid, zred, **extras):
    """Build a dictionary of observational data,
    taking the photometry from the 3DHST catalog

    :param objid:
        The 3DHST ID of the galaxy to be fit.

    :returns obs:
        A dictionary of observational data to use in the fit.
    """

    phot_all     = fits.open(dir_new_photometry + 'bluejay_phot_cat_v1.3.fits')[1].data
    #phot_hst    = fits.open(dir_new_photometry + 'new_HST_photometry/HST_Flux_Aperture_PSFMatched_AppCorr.fits')[1].data
    #phot_nircam = fits.open(dir_new_photometry + 'PRIMER_photometry/Flux_Aperture_PSFMatched_AperCorr.fits')[1].data
    #phot_miri   = fits.open(dir_new_photometry + 'PRIMER_photometry/Flux_Aperture_PSFMatched_AperCorr.fits')[1].data

    ph = phot_all[phot_all['ID'] == str(objid)]

    # HST
    filter_dict_3dhst = {
                        'F125W': 'wfc3_ir_f125w',
                        'F140W': 'wfc3_ir_f140w',
                        'F160W': 'wfc3_ir_f160w',
                        'F606W': 'acs_wfc_f606w',
                        'F814W': 'acs_wfc_f814w'
                        }

    # PRIMER
    filter_dict_nircam = { 
                        'F090W':'jwst_f090w',
                        'F115W':'jwst_f115w',
                        'F150W':'jwst_f150w',
                        'F200W':'jwst_f200w',
                        'F277W':'jwst_f277w',
                        'F356W':'jwst_f356w',
                        'F410M':'jwst_f410m',
                        'F444W':'jwst_f444w'
                        }

    # MIRI
    #filter_dict_miri = {
    #                    'F770W':'jwst_f770w',
    #                    'F1800W':'jwst_f1800w'
    #                    }

    # list filters from the table that you want to use
    filter_code = list(filter_dict_3dhst.keys()) + list(filter_dict_nircam.keys()) #+ list(filter_dict_miri.keys())

    # list corresponding names in the sedpy database
    filter_name = list(filter_dict_3dhst.values()) + list(filter_dict_nircam.values()) #+ list(filter_dict_miri.values())

    # create obs dictionary and load filters
    obs = {}
    obs['filters'] = sedpy.observate.load_filters(filter_name)
    obs['filter_code'] = filter_code

    fluxes     = np.zeros(len(filter_name))
    fluxes_err = np.zeros(len(filter_name))
  
    for ff, fil in enumerate(filter_code):
        fluxes[ff] = ph[str(fil) + '_flux']
        fluxes_err[ff] = ph[str(fil) + '_flux_err']

    #fluxes     = ph[ [f + '_flux' for f in filter_code] ]
    #fluxes_err = ph[ [f + '_flux_err' for f in filter_code] ]

    # add 5% systematic error in quadrature to all bands
    fluxes_err = [ np.sqrt(fluxes_err[i]**2 + (0.05*fluxes[i])**2) for i in range(len(fluxes))]

    # Jy --> maggies
    obs['maggies']     = np.array(fluxes) / 3631
    obs['maggies_unc'] = np.array(fluxes_err) / 3631

    # make mask, where True means that you want to fit that data point
    obs['phot_mask'] = np.array([True for f in obs["filters"]])

    # mask GALEX points
    #obs['phot_mask'][filter_code.index('fuv')] = False
    #obs['phot_mask'][filter_code.index('nuv')] = False

    # mask 24um point
    #obs['phot_mask'][filter_code.index('mips24')] = False

    # check whether there were missing data points (identified by -99 in the catalog)
    #w_missing = np.where(np.array(list(fluxes)) < (-90.0))[0]
    #if len(w_missing) > 0:
    #    obs['maggies'][w_missing] = np.nan
    #    obs['maggies_unc'][w_missing] = np.nan
    #    obs['phot_mask'][w_missing] = False

    # array of effective wavelengths for each filter, useful for plotting
    obs['phot_wave'] = np.array([f.wave_effective for f in obs['filters']])

    # ensure all required keys are present in the obs dictionary
    obs = fix_obs(obs)

    # Set elements related to spectral fitting to None
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['unc'] = None
    obs['mask'] = None

    # convert wavelength from air to vacuum (prospector works in vacuum)
    # wave = air_to_vac(wave)
    
    print('objid:', objid)
    

    ######################################################
    #           Read extracted spectrum data             #
    ######################################################

    dir_spec = '/Users/letiziabugiani/Desktop/bluejay/spectroscopy/extracted_spectra/'
    data_spec = fits.open(dir_spec+'s'+str(objid).zfill(5)+'_spec1d.fits')

    obs['wavelength'] = data_spec[1].data * 10000 # convert microns to AA
    obs['spectrum']   = data_spec[2].data
    # flux is an arbitrary unit
    obs['unc']        = data_spec[3].data
    
    ################################################
    #           Read resolution data               #
    ################################################
    nirspec_g140m = fits.open('/Users/letiziabugiani/Desktop/bluejay/spectroscopy/jwst_nirspec_g140m_disp.fits')
    nirspec_g235m = fits.open('/Users/letiziabugiani/Desktop/bluejay/spectroscopy/jwst_nirspec_g235m_disp.fits') 
    nirspec_g395m = fits.open('/Users/letiziabugiani/Desktop/bluejay/spectroscopy/jwst_nirspec_g395m_disp.fits')
    # wavelengths in these data are in micron

    # combine g140m, g235m, g395m resolution data
    # for overlapped region, take the "lower" resolution for now
    # should ask how the resolution really is for these overlapped regions
    nirspec_wave = np.concatenate((nirspec_g140m[1].data['WAVELENGTH'][nirspec_g140m[1].data['WAVELENGTH']<min(nirspec_g235m[1].data['WAVELENGTH'])],
                               nirspec_g235m[1].data['WAVELENGTH'][nirspec_g235m[1].data['WAVELENGTH']<min(nirspec_g395m[1].data['WAVELENGTH'])],
                               nirspec_g395m[1].data['WAVELENGTH']), axis=None)
    nirspec_R    = np.concatenate((nirspec_g140m[1].data['R'][nirspec_g140m[1].data['WAVELENGTH']<min(nirspec_g235m[1].data['WAVELENGTH'])],
                               nirspec_g235m[1].data['R'][nirspec_g235m[1].data['WAVELENGTH']<min(nirspec_g395m[1].data['WAVELENGTH'])],
                               nirspec_g395m[1].data['R']), axis=None)
    func_nirspec = interpolate.interp1d(nirspec_wave, 2.998e5/(nirspec_R*2.355)) 

    # interpolate nirspec_wave vs. nirspec_sigma_v to get sigma_v for observed wavelength points.
    obs['sigma_v'] = func_nirspec(data_spec[1].data) # wavelength should be in micron  
    

    ################################################
    #               Mask out regions               #
    ################################################
    #obs = mask_obs(obs, zred)
    #obs['mask'] = np.full((len(obs['wavelength']),),True)

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


def build_model(objid, zred=None, waverange=None, add_duste=True, add_neb=False, add_agn=False, fit_afe=False,
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

    tuniv = cosmo.age(zred).value
    #agelims_Myr = np.append( np.logspace( np.log10(30.0), np.log10(0.95*tuniv*1000), 13), tuniv*1000 )
    agelims_Myr = np.append( np.logspace( np.log10(30.0), np.log10(0.8*tuniv*1000), 12), [0.9*tuniv*1000, tuniv*1000])
    agelims = np.concatenate( ( [0.0], np.log10(agelims_Myr*1e6) ))
    agebins = np.array([agelims[:-1], agelims[1:]]).T
    nbins = len(agelims) - 1

    # This is the *total*  mass formed, as a variable
    model_params["logmass"]    = {"N": 1, "isfree": True,
                                  "init": 10.5,
                                  'units': "Solar masses formed",
                                  'prior': priors.TopHat(mini=8.5, maxi=13)}

    # This will be the mass in each bin.  It depends on other free and fixed
    # parameters.  Its length needs to be modified based on the number of bins
    model_params["mass"]       = {'N': nbins, 'isfree': False,
                                  'init': (10**10.5)/nbins,
                                  'units': "Solar masses formed",
                                  'depends_on': transforms.logsfr_ratios_to_masses}

    # This gives the start and stop of each age bin.  It can be adjusted and its
    # length must match the length of "mass"
    model_params["agebins"]    = {'N': nbins, 'isfree': False,
                                  'init': agebins,
                                  'units': 'log(yr)'}

    # This controls the distribution of SFR(t) / SFR(t+dt). It has nbins-1 components.
    model_params["logsfr_ratios"] = {'N': nbins-1, 'isfree': True,
                                     'init': np.full(nbins-1, 0.0),  # constant SFH
                                     'units': '',
                                     'prior':priors.StudentT(mean=np.full(nbins-1, 0.0),
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
                             "units": "optical depth at 5500AA",
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
        
        to_fit = ['[S III] 3722', '[O II] 3726', '[O II] 3729', 'Ba-7 3835', 'Ba-6 3889', \
                  'Ba-5 3970','Ba-delta 4101.76A', 'Ba-gamma 4341', '[O III] 4363', 'He I 4471.49A', \
                    '[C  I] 4621', '[Ne IV] 4720', '[O III] 4959', '[O III] 5007', \
                    '[N II] 6548', '[N II] 6584']

        emi_wavelengths = [3723, 3726, 3729, 3835, 3889, 3970,  
                            4102.89, 4341.69, 4364, 4472, 
                            4621, 4720, 4960.30, 5008.24, 
                            6549.86, 6585.27]

        to_fit_exist = to_fit.copy()
        obs_data = np.load('/Users/letiziabugiani/Desktop/bluejay/spectroscopy/prospector/v0.7/fitting_regions/obs_'+str(objid)+'.npy', allow_pickle=True)
        obs_wave_mask = obs_data.item()['wavelength'][obs_data.item()['mask']]
        obs_spec_mask = obs_data.item()['spectrum'][obs_data.item()['mask']]
        obs_unc_mask = obs_data.item()['unc'][obs_data.item()['mask']]

        for i, w_emi in enumerate(emi_wavelengths):
            wave_emi = obs_wave_mask[abs((obs_wave_mask/(1+zred) - w_emi)/w_emi) < 500/3e5]
            spec_emi = obs_spec_mask[abs((obs_wave_mask/(1+zred) - w_emi)/w_emi) < 500/3e5]
            unc_emi  = obs_unc_mask[abs((obs_wave_mask/(1+zred) - w_emi)/w_emi) < 500/3e5]
            
            print(wave_emi, spec_emi, unc_emi)
            if len(wave_emi) < 5:
                print('remove', str(to_fit[i]), 'due to no pixels, pixel num=', str(len(wave_emi)))
                to_fit_exist.remove(to_fit[i]) 
                # keep in mind that the length of "to_fit_exist" changes with the for loop

            elif len(spec_emi[np.isnan(spec_emi)]) > 0 or len(unc_emi[np.isnan(unc_emi)]) > 0 or len(unc_emi[unc_emi == 0]) > 0 \
                    or len(spec_emi[spec_emi < 0]) > 0 or len(unc_emi[unc_emi <=0]) > 0:
                print('remove', str(to_fit[i]), 'due to BAD pixels')
                to_fit_exist.remove(to_fit[i]) 
                
        print('fitting emission lines:', to_fit_exist)  
        print('=====================================')

        model_params['elines_to_fit']['init'] = to_fit_exist
        #model['eline_delta_zred'] # jitter around! 
    # ----------------------------
    # ----------------------------

    # Now instantiate the model object using this dictionary of parameter specifications
    model = PolySpecModel(model_params)

    return model


# --------------
# SPS Object
# --------------

def build_sps(zred, zcontinuous=1, smooth_instrument=False, obs=None, **extras):
    """
    :param zcontinuous:
        A value of 1 insures that we use interpolation between SSPs to
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """
    sps = FastStepBasis(zcontinuous=zcontinuous)

    if (obs is not None) and (smooth_instrument):
        #from exspect.utils import get_lsf
        print('---- wave-dependent resolution ----')
        wave_obs = obs["wavelength"]
        sigma_v  = obs["sigma_v"]
        speclib  = sps.ssp.libraries[1].decode("utf-8")
        wave, delta_v = get_lsf(wave_obs, sigma_v, speclib=speclib, zred=zred, **extras)
        sps.ssp.params['smooth_lsf'] = True
        sps.ssp.set_lsf(wave, delta_v)

    return sps


def get_lsf(wave_obs, sigma_v, speclib, zred, **extras):
    """This method takes an instrimental resolution curve and returns the
    quadrature difference between the instrumental dispersion and the library
    dispersion, in km/s, as a function of restframe wavelength
    :param wave_obs: ndarray
        Observed frame wavelength (AA)
    :param sigma_v: ndarray
        Instrumental spectral resolution in terms of velocity dispersion (km/s)
    :param speclib: string
        The spectral library.  One of 'miles' or 'c3k_a', returned by
        `sps.ssp.libraries[1]`
    """
    lightspeed = 2.998e5  # km/s
    # filter out some places where sdss reports zero dispersion
    good = sigma_v > 0
    wave_obs, sigma_v = wave_obs[good], sigma_v[good]
    wave_rest = wave_obs / (1 + zred)

    # Get the library velocity resolution function at the corresponding
    # *rest-frame* wavelength
    if speclib == "miles":
        miles_fwhm_aa = 2.54
        sigma_v_lib = lightspeed * miles_fwhm_aa / 2.355 / wave_rest
        # Restrict to regions where MILES is used
        good = (wave_rest > 3525.0) & (wave_rest < 7500)

    elif speclib == "c3k_a":
        R_c3k = 3000
        sigma_v_lib = lightspeed / (R_c3k * 2.355)
        # Restrict to regions where C3K is used
        good = (wave_rest > 2750.0) & (wave_rest < 9100.0)

    elif speclib == "c3k_hr ":
        data_lib = np.loadtxt('/Users/letiziabugiani/miniconda3/envs/prospector/fsps/SPECTRA/C3K/c3k_hr.lambda', 
                                dtype=[('wave_lib', '<f8'), ('sigma_v_lib', '<f8')])
        sigma_v_lib = data_lib['sigma_v_lib'][np.digitize(wave_rest, data_lib['wave_lib'])-1]
        good = (wave_rest > 0)

    else:
        sigma_v_lib = sigma_v
        good = slice(None)
        raise ValueError("speclib of type {} not supported".format(speclib))

    # Get the quadrature difference
    # (Zero and negative values are skipped by FSPS)
    dsv = np.sqrt(np.clip(sigma_v**2 - sigma_v_lib**2, 0, np.inf))

    # return the broadening of the rest-frame library spectra required to match
    # the observed frame instrumental lsf
    return wave_rest[good], dsv[good]
# ------------------
# Noise Model
# ------------------

def build_noise(**extras):
    jitter = Uncorrelated(parnames = ['spec_jitter'])
    spec_noise = NoiseModel(kernels=[jitter],metric_name='unc',weight_by=['unc'])
    return spec_noise, None

# ------------------------------------
# ------------------------------------
# ------------------------------------


if __name__=='__main__':

    # - Parser with default arguments -
    parser = prospect.utils.prospect_args.get_parser()

    # - Add custom arguments -
    parser.add_argument('--objid', type=int, default=0,
                        help="ID of the object to fit")
    parser.add_argument('--zred', type=float, default=None,
                        help="fixed redshift value") 
    parser.add_argument('--output_tag', type=str, default=None,
                        help="output tag name")          
    parser.add_argument('--add_duste', action="store_true", default=True,
                        help="If set, add dust emission to the model.")
    parser.add_argument('--add_neb', action="store_true",default=False,
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_agn', action="store_true", default=False,
                        help="If set, add agn emission to the model.")
    parser.add_argument('--fit_afe', action="store_true", default=False,
                        help="If set, use afe as a free parameter to fit.")

    args = parser.parse_args()
    run_params = vars(args)

    run_params["zred"] = find_zred(run_params["objid"])

    run_params["param_file"] = __file__
    run_params["outfile"] = "output"
    
    # add in dynesty settings
    run_params['dynesty'] = True
    run_params['nested_target_n_effective'] = 1000
    run_params['nested_nlive_batch'] = 1000 
    run_params['nested_walks'] = 32  
    run_params['nested_nlive_init'] = 3000 
    run_params['nested_dlogz_init'] = 0.01
    run_params['nested_maxcall'] = 7000000
    run_params['nested_maxcall_init'] = 7000000
    run_params['nested_sample'] = 'rwalk' # make sure *nested_sample* is specified rather than nested_method.
    run_params['nested_maxbatch'] = None
    run_params['nested_first_update'] = {'min_ncall': 20000, 'min_eff': 7.5}


    print("\nFitting {}".format(run_params["objid"]))
    print("------------------\n")

    # build observations
    obs = build_obs(**run_params)
    np.save('obs/obs_'+str(run_params["objid"]), obs)
    print('obs saved')

    # build sps
    sps = build_sps(zred=run_params['zred'], smooth_instrument=True, obs=obs)
    
    # calculate rest-frame wavelength range 
    run_params[' '] = (np.max(obs['wavelength'][obs['mask']]) - 
        np.min(obs['wavelength'][obs['mask']]))/(1+run_params['zred'])

    # build model
    model = build_model(**run_params)

    # build noise
    noise = build_noise(**run_params)

#####

    # Set up MPI. Note that only model evaluation is parallelizable in dynesty,
    # and many operations (e.g. new point proposal) are still done in serial.
    # This means that single-core fits will always be more efficient for large
    # samples. having a large ratio of (live points / processors) helps efficiency
    # Scaling is: S = K ln(1 + M/K), where M = number of processes and K = number of live points
    # Run as: mpirun -np <number of processors> python demo_mpi_params.py
    try:
        import mpi4py
        from mpi4py import MPI
        from schwimmbad import MPIPool

        mpi4py.rc.threads = False
        mpi4py.rc.recv_mprobe = False

        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        withmpi = comm.Get_size() > 1
    except ImportError:
        print('Failed to start MPI; are mpi4py and schwimmbad installed? Proceeding without MPI.')
        withmpi = False

    # Evaluate SPS over logzsol grid in order to get necessary data in cache/memory
    # for each MPI process. Otherwise, you risk creating a lag between the MPI tasks
    # caching data depending which can slow down the parallelization
    if (withmpi) & ('logzsol' in model.free_params):
        dummy_obs = dict(filters=None, wavelength=None)

        logzsol_prior = model.config_dict["logzsol"]['prior']
        lo, hi = logzsol_prior.range
        logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)

        sps.update(**model.params)  # make sure we are caching the correct IMF / SFH / etc
        for logzsol in logzsol_grid:
            model.params["logzsol"] = np.array([logzsol])
            _ = model.predict(model.theta, obs=dummy_obs, sps=sps)

    # ensure that each processor runs its own version of FSPS
    # this ensures no cross-over memory usage
    from prospect.fitting import lnprobfn
    from functools import partial
    lnprobfn_fixed = partial(lnprobfn, sps=sps)

    if withmpi:
        with MPIPool() as pool:

            # The subprocesses will run up to this point in the code
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            nprocs = pool.size
            output = prospect.fitting.fit_model(obs, model, sps, noise, pool=pool, queue_size=nprocs, lnprobfn=lnprobfn_fixed, **run_params)
    else:
        output = prospect.fitting.fit_model(obs, model, sps, noise, lnprobfn=lnprobfn_fixed, **run_params)

######

    # get unique name for the output file
    hfile = "{0}_{1}_{2}_{3}_mcmc.h5".format(run_params["outfile"], 
                run_params["objid"], 
                run_params["output_tag"],
                int(time.time()))

        # write results to file
    prospect.io.write_results.write_hdf5('output/outputs/'+ hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    # output various figures
    #make_figure('output/outputs/'+ hfile)

    from plot_prospector_v1_final import plot_all
    plot_all(run_params["objid"], hfile, fix_solar=True)

    print("\nFinished\n")