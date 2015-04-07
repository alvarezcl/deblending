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
import os
import ipdb
# ----------------------------- Parameters -----------------------------------

# Which run
number_run = str(1)

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
y0_a = 0
n_a = 0.5
obj_a = [flux_a,hlr_a,e1_a,e2_a,0,y0_a,n_a]

# Parameters for object b
flux_b = 25000 # counts
hlr_b = hlr_a # arcsec
e1_b = 0.0
e2_b = 0.0
y0_b = 0
n_b = 0.5
obj_b = [flux_b,hlr_b,e1_b,e2_b,0,y0_b,n_b]

# Sampling method
method = 'fft'

# Use LSST defined sky noise for r-band
add_noise_flag = True
texp = 6900 # seconds;
sbar = 26.8 # sky photons per second per pixel
sky_level = 0 # For noiseless images 
sky_noise = np.sqrt(sky_level)
texp = 0; # For only poisson noise
sky_info = [add_noise_flag,texp,sbar,sky_level]

# psf properties
psf_flag = False
beta = 3
fwhm_psf = 0.6
psf_info = [psf_flag,beta,fwhm_psf]

# Separations to run through, along the x-axis
separation = [2.2,2.0,1.8,1.6]

# Number of trials to use
num_trials = 20
num_trial_arr = num_trials*np.ones(len(separation),dtype=np.int64)
min_sep = 1.0
factor = 0.4
sec_num_trial = factor*num_trials
num_trial_arr[num_trial_arr <= min_sep] = sec_num_trial

# Use true values
use_est_centroid = True

# Do not randomize about median separation
randomize = True

# When to save images for checking
mod_val = 0.5*num_trials

# Create the sub-directories
info_str = Library.join_info(separation,
                             num_trial_arr,
                             func,
                             seed_int_arr,
                             image_params,
                             obj_a,obj_b,method,
                             sky_info,
                             psf_info,
                             mod_val,use_est_centroid,randomize)
                            
Library.create_read_me(info_str,number_run)

means, s_means = Library.run_over_separation(separation,
                                             num_trial_arr,
                                             func,
                                             seed_rng_arr,
                                             image_params,
                                             obj_a,obj_b,method,
                                             sky_info,
                                             psf_info,
                                             mod_val,use_est_centroid,randomize,
                                             number_run)
                                             
# Plot the bias information in sub-directory
                                    
                 