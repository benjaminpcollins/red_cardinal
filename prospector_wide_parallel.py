## code that usese previous prospector output to create a
## spectral fitting on a wider wavelength range than the original fit

# NEW VERSION!!!

import os

# set the environment variable for FSPS to work
sps_home = os.getenv('SPS_HOME', "/Users/letiziabugiani/miniconda3/envs/prospector/fsps")
os.environ['SPS_HOME'] = sps_home

import numpy as np
import pickle
from multiprocessing import Pool
import prospect.io.read_results as reader

import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from astropy.io import fits


# import functions from file
from prosparams_emlinesfit import * 

from scipy.signal import medfilt
from scipy import interpolate


def get_zred(objid):
    """
    Retrieve the redshift value for a given object ID from a specified file.

    Parameters:
    objid (int): The object ID for which the redshift value is to be retrieved.

    Returns:
    float: The absolute value of the redshift corresponding to the given object ID.
    """
    dat_zred = np.loadtxt('/Users/letiziabugiani/Desktop/bluejay/spectroscopy/redshifts_v0.1.txt', dtype=[('galid', int),('zred', '<f8')])
    return abs(dat_zred['zred'][dat_zred['galid'] == objid][0])

def get_MAP(results,verbose=False):
    chains = results['chain']  # Shape: (n_samples, n_parameters)
    log_probabilities = results['lnprobability']  # Shape: (n_samples,)

    # Find the index of the maximum log probability
    max_prob_index = np.argmax(log_probabilities)

    # Get the MAP parameters
    map_parameters = chains[max_prob_index]

    if verbose:
        print("MAP Parameters:", map_parameters)

    return map_parameters

def mask_obs(obs, zred):
    # mask emission lines in the wavelength range

    # import lines to mask
    emi_wavelengths = []
    with open(os.environ['SPS_HOME']+'/data/emlines_info.dat', 'r') as file:
        for line in file:
            float_val, _ = line.strip().split(',')
            emi_wavelengths.append(float(float_val))

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

# ----------------------------------------------------------------------------

# MAIN LOOP TO FIT THE SPECTRA

# iterate trough the list of galaxies

# dir where Prospector output are stored
dirout = "/Users/letiziabugiani/Desktop/bluejay/spectroscopy/prospector/v0.7_allfall_marg/outputs/"

#function to be run in parallel
def process_galaxy(output):

    try:
        objid = int(output.split('_')[1])
    except ValueError:
        return

    # check if the file already exists
    # if os.path.exists("polyfit25/spec_calib_wide/spec_calibrated_" + str(objid) + ".pkl"):
    #     print(f"File for galaxy {objid} already exists")
    #     return
    
    # get the parameters of the previous Prospector run
    results, _ , _ = reader.results_from(dirout+output)

    # we have to change the params_file saved in the memory of the model
    results['run_params']['param_file'] = "prosparams_emlinesfit.py"

    file_path = results['run_params']['param_file']
    with open(file_path, 'r') as file:
        script_as_string = file.read()
    results['paramfile_text'] = script_as_string
        
    # get the redshift of the galaxy
    zred = get_zred(objid)

    print(f"GALAXY {objid}:  z = {zred}")

    # build obserrved spectrum
    obs = build_obs(objid,zred)

    # IMPORTANT! mask emission lines and bad pixels
    obs['mask'] = mask_obs(obs, zred)

    # build the model without nebular emission fits
    model = build_model(objid, zred, add_duste=results['run_params']['add_duste'],
                    add_neb=False,  # Set add_neb to False
                    add_agn=results['run_params']['add_agn'],
                    fit_afe=results['run_params']['fit_afe'])
    
    model.params['polyorder'] = 25
    
    # now we have to exclude the last 3 parameters from the fit
    map_parameters = get_MAP(results)
    map_parameters = map_parameters[:-3]
    
    # Calculate the spectrum based on the Maximum A Posteriori (MAP) parameters
    sps = build_sps(zred=zred,smooth_instrument=True,obs=obs)
    spec, phot, _ = model.predict(map_parameters, obs=obs, sps=sps)

    calib_vector = model._speccal

    # calibrate everything 
    jy = 1e-23 * u.erg / u.Hz / u.cm**2 / u.s

    wave = obs['wavelength']
    cont = spec/calib_vector * 3631 * jy
    cont = cont.to_value(u.erg/u.s/u.cm**2/u.AA,u.equivalencies.spectral_density(wave* u.AA))

    spectrum  = obs['spectrum']/calib_vector * 3631 * jy
    spectrum = spectrum.to_value(u.erg/u.s/u.cm**2/u.AA,u.equivalencies.spectral_density(wave* u.AA))

    #get the spectral jitter  for the errors
    spec_jitter = map_parameters[5]
    errs = obs['unc']*spec_jitter/calib_vector * 3631 * jy # observetional errors
    errs = errs.to_value(u.erg/u.s/u.cm**2/u.AA,u.equivalencies.spectral_density(wave* u.AA))

    sub_spectrum = spectrum - cont

    # Calculate the 16th and 84th percentile values of the parameters' posterior distributions
    theta_16th = np.percentile(results['chain'], 16, axis=0)
    theta_84th = np.percentile(results['chain'], 84, axis=0)

    # Generate the model spectra for the 16th percentile parameters
    spec_16th, _, _ = model.predict(theta_16th[:-3], obs=obs, sps=sps)

    # Generate the model spectra for the 84th percentile parameters
    spec_84th, _, _ = model.predict(theta_84th[:-3], obs=obs, sps=sps)

    #calculate uncertainties on the continuum fits
    cont_errs = np.abs(spec_16th  - spec_84th)/2/calib_vector*3631*jy
    cont_errs = cont_errs.to_value(u.erg / u.AA / u.cm**2 / u.s,u.equivalencies.spectral_density(wave* u.AA))

    # errors on the subtracted spectrum
    tot_errs = np.sqrt(cont_errs**2+errs**2)

    # plot
    # Plot the intrinsic observed SED of the model vs the calibrated spectrum
    plt.figure(figsize=(15, 5))
    plt.plot(obs['wavelength'], spectrum, label="Observed Spectrum",color='black',alpha=0.3)
    plt.plot(obs['wavelength'], cont, label="Continuum fit",c='C2')
    plt.plot(obs['wavelength'], errs, label="errs",c='C0')
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux (erg/cm^2/s/Å)")
    ymax = np.nanmax(cont)*2
    plt.ylim(-.1e-19,ymax)
    plt.xlim(.9e4,5.3e4)
    plt.legend()
    plt.title("calibrated spectrum")
    plt.savefig('polyorder25/plots/'+str(objid)+'_calibrated_spectrum.png')
    plt.close()

    ##########################################################

    #save in fits file   
    # Header for the PrimaryHDU
    path_calib = 'polyorder25/calib_specs/'
    p_hdr = fits.Header()
    p_hdr['COMMENT'] = 'Calibrated spectrum'
    p_hdr['ext_1'] = 'data'
    primary_hdu = fits.PrimaryHDU(header=p_hdr)
    
    hdu1 = Table([wave,spectrum,cont,sub_spectrum,tot_errs,errs],
                 names=('wavelength','observed','continuum','subtracted','errors','obs_errors'))
    hdu1 = fits.BinTableHDU(data=hdu1)
    hdu = fits.HDUList([primary_hdu,hdu1])#,hdu2,hdu3,hdu4,hdu5])
    hdu.writeto(path_calib
                 + str(objid)+'.fits', overwrite=True)
       
    ##########################################################

    #save in a pickle file
    dir_calibrated = "/Users/letiziabugiani/Desktop/bluejay/spectroscopy/prospector/v0.7_allfall_marg/polyorder25/spec_calib_wide/"

    # Create a dictionary to hold the data
    data_dict = {
        'OBSERVED_WAVELENGTH': obs['wavelength'],
        'PHOTOMETRY_WAVELENGTH': obs['phot_wave'],
        'OBSERVED_SPECTRUM': obs['spectrum'],
        'OBSERVED_ERRS': obs['unc'],
        'MASK': obs['mask'],
        'OBSERVED_PHOTOMETRY': obs['maggies'],
        'OBSERVED_PHOTOMETRY_ERRS': obs['maggies_unc'],
        'MODEL_PHOTOMETRY': phot,
        'MODEL_SPEC': spec,
        'MODEL_SPEC_16TH': spec_16th,
        'MODEL_SPEC_84TH': spec_84th,
        'CALIBRATION_VECTOR': calib_vector}

    # Write the dictionary to a .pkl file
    with open(dir_calibrated + "spec_calibrated_" + str(objid) + ".pkl", 'wb') as f:
        pickle.dump(data_dict, f)



if __name__ == "__main__":
    outlist = os.listdir(dirout)
    """for out in outlist:
        print(out.split('_')[1])
    """
    #outlist = outlist[:4]
    with Pool() as pool:
        pool.map(process_galaxy, outlist)

    print("done!")


    
    
