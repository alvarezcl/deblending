# ----------------------------------------------------------------------------
# Filename: Bias_Over_Separation.py
# Author: Luis Alvarez
# This program runs through different separations
# for two overlapping objects, runs a number of trials
# in which two overlapping objects are simultaneously
# fitted, deblended and then fitted, and then fitted 
# to the true individual objects. The information is
# then plotted and triangle plot files are saved
# to appropriate directories. 

# ----------------------------- Import Statements ----------------------------
from __future__ import division
import Library
import galsim
import numpy as np

def run_bias_over_separation(directory,
                             psf,
                             est_centroid,
                             random_pixel,
                             x_axis,y_axis,
                             l_diag,r_diag):

    # ----------------------------- Parameters -----------------------------------
    
    # Which run directory to store information in
    if type(directory) == int:
        directory = str(directory)
    elif type(directory) != str:
        raise ValueError('Directory must be int or str type')
    
    # Galsim function definitions
    func = galsim.Sersic
    
    # Seed
    seed_int_arr = np.array([387645,981234,676293,978676,187736,897376,19377656])
    seed_1 = galsim.BaseDeviate(seed_int_arr[0])
    seed_2 = galsim.BaseDeviate(seed_int_arr[1])
    seed_3 = galsim.BaseDeviate(seed_int_arr[2])
    seed_4 = galsim.BaseDeviate(seed_int_arr[3])
    seed_5 = galsim.BaseDeviate(seed_int_arr[4])
    seed_6 = galsim.BaseDeviate(seed_int_arr[5])
    seed_rng_arr = np.array([seed_1,seed_2,seed_3,seed_4,seed_5,seed_6])
    np.random.seed(seed_int_arr[6])
    
    # Image parameters
    pixel_scale = 0.2
    x_len = y_len = 100
    image_params = [pixel_scale,x_len,y_len]
    
    # Parameters for object a
    flux_a = 25000 # counts
    hlr_a = 1  # arcsec
    e1_a = 0.0 
    e2_a = 0.0
    x0_a = 0
    y0_a = 0
    n_a = 0.5
    obj_a = [flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,n_a]
    
    # Parameters for object b
    flux_b = 25000 # counts
    hlr_b = hlr_a # arcsec
    e1_b = 0.0
    e2_b = 0.0
    x0_b = 0
    y0_b = 0
    n_b = 0.5
    obj_b = [flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,n_b]
    
    # Sampling method
    method = 'fft'
    
    # Use LSST defined sky noise for r-band
    add_noise_flag = True
    texp = 6900 # seconds;
    sbar = 26.8 # sky photons per second per pixel
    sky_level = 0 # For noiseless images 
    if sky_level == 0:
        texp = 0 # To avoid a seg fault, ensure texp = 0 if sky level is 0
    sky_info = [add_noise_flag,texp,sbar,sky_level]
    
    # psf properties
    psf_flag = psf
    beta = 3
    fwhm_psf = 0.7
    psf_info = [psf_flag,beta,fwhm_psf]
    
    # Separations to run through, along the axis specified
    separation = [2.4,2.2,2.0,1.8,1.6,1.4]
    x_sep = x_axis
    y_sep = y_axis
    left_diag = l_diag
    right_diag = r_diag
    
    # Number of trials to use for each separation
    num_trials = 1000
    num_trial_arr = num_trials*np.ones(len(separation),dtype=np.int64)
    min_sep = 1.0
    factor = 1.0
    sec_num_trial = factor*num_trials
    num_trial_arr[num_trial_arr <= min_sep] = sec_num_trial
    
    # Use true values
    use_est_centroid = est_centroid
    
    # Do not randomize about median separation
    randomize = random_pixel
    
    # When to save images for checking and outputting place on terminal
    mod_val = 0.2*num_trials
    
    # Bool for saving triangle plots 
    create_tri_plots = True
    
    # Create the string of information
    info_str = Library.join_info(directory,
                                 separation,
                                 num_trial_arr,
                                 func,
                                 seed_int_arr,
                                 image_params,
                                 obj_a,obj_b,method,
                                 sky_info,
                                 psf_info,
                                 mod_val,use_est_centroid,randomize,
                                 x_sep,y_sep,
                                 left_diag,right_diag,
                                 create_tri_plots)
    
    # Create the read me file containing the information of the run                  
    Library.create_read_me(info_str,directory)
    
    # Run through different separations and obtain the mean
    # values of e1 and e2 for objects a and b for each method:
    # fits to the true objects, simultaneous fits, and fits to the
    # deblended objects
    means, s_means = Library.run_over_separation(separation,
                                                 num_trial_arr,
                                                 func,
                                                 seed_rng_arr,
                                                 image_params,
                                                 obj_a,obj_b,method,
                                                 sky_info,
                                                 psf_info,
                                                 mod_val,use_est_centroid,randomize,
                                                 directory,
                                                 create_tri_plots,
                                                 x_sep=x_sep,y_sep=y_sep,
                                                 right_diag=right_diag,
                                                 left_diag=left_diag)
                                                 
    # Plot the bias information in sub-directory
    fs = 14
    leg_fs = 12
    min_offset = 1.3
    max_offset = 1.3
    Library.create_bias_plot_e(directory,separation,means,s_means,pixel_scale,
                               fs,leg_fs,min_offset,max_offset,psf_flag)
                               
if __name__ == '__main__':

    psf = False
    est_centroid = True
    random_pixel = False
    x_axis = True
    y_axis = False
    l_diag = False
    r_diag = False
    dir_str = 'psf:' + str(psf) + ';true_centroid:' + str(not est_centroid) + ';randomization:' + str(random_pixel)
    run_bias_over_separation(dir_str,
                             psf,
                             est_centroid,
                             random_pixel,
                             x_axis,y_axis,
                             l_diag,r_diag)

    psf = False
    est_centroid = False
    random_pixel = False
    x_axis = True
    y_axis = False
    l_diag = False
    r_diag = False
    dir_str = 'psf:' + str(psf) + ';true_centroid:' + str(not est_centroid) + ';randomization:' + str(random_pixel)
    run_bias_over_separation(dir_str,
                             psf,
                             est_centroid,
                             random_pixel,
                             x_axis,y_axis,
                             l_diag,r_diag)
