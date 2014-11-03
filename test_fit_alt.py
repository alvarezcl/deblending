
# coding: utf-8

# In[1]:

## This script creates two objects, blends, deblends and then measures
## the shape of the deblended objects and compares to the original.

from __future__ import division
import Library
import galsim
import deblend
import sys
import numpy as np
import lmfit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import triangle
get_ipython().magic(u'matplotlib inline')
import pdb


# In[18]:

# Create a blend, deblend, then estimate ellipticity of deblended objects and true objects.
def de_blend(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,n_a,
             flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,n_b,
             func, seed_1, seed_2, seed_3,
             pixel_scale, x_len, y_len, 
             add_noise_flag, texp, sbar, 
             psf_flag, beta, fwhm_psf,
             method,plot=False):
    
    # Create the objects and combine them
    image_a = Library.create_galaxy(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,galtype_gal=func,sersic_index=n_a,
                                    x_len=x_len,y_len=y_len,scale=pixel_scale,
                                    psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                                    method=method, seed=seed_1)
                                
    image_b = Library.create_galaxy(flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,galtype_gal=func,sersic_index=n_b,
                                    x_len=x_len,y_len=y_len,scale=pixel_scale,
                                    psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                                    method=method, seed=seed_2)
    sky_level = texp*sbar
    sersic_func = func
    
    tot_image = image_a + image_b
    if add_noise_flag == True:
        image_noise = Library.add_noise(tot_image,seed=seed_3,sky_level=sky_level)
        image_c = Library.add_noise(image_a,seed=seed_3,sky_level=sky_level)
        image_d = Library.add_noise(image_b,seed=seed_3,sky_level=sky_level)
        image = image_noise
    else:
        image_noise = tot_image
    
        
    # Deblend the resulting blend
    peak1 = (x0_a,y0_a)
    peak2 = (x0_b,y0_b)
    peaks_pix = [[p1/0.2 for p1 in peak1],
                 [p2/0.2 for p2 in peak2]]
                 
    templates, template_fractions, children = deblend.deblend(tot_image.array, peaks_pix)
    
    # Now we can run the fitter to estimate the shape of the children
    # -----------------------------------------------------------------------    
    # Estimate the parameters of the image.
    
    # Define some seed that's far from true values and insert into
    # lmfit object for galaxy one and two
    lim = 1/np.sqrt(2)
    factor = 0.5
    p0_a = factor*np.array([flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a])
    p0_b = factor*np.array([flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b])
    
    parameters_a = lmfit.Parameters()
    parameters_a.add('flux', value=p0_a[0])
    parameters_a.add('hlr', value=p0_a[1])
    #parameters.add('e1_a', value=p0[2],min=-0.99,max=0.99)
    #parameters.add('mag',value=0.99,max=0.99,vary=True)
    #parameters.add('e2_a',expr='sqrt(mag**2 - e1_a**2)')
    parameters_a.add('e1',value=p0_a[2],min=-lim,max=lim)
    parameters_a.add('e2',value=p0_a[3],min=-lim,max=lim)    
    parameters_a.add('x0',value=p0_a[4])
    parameters_a.add('y0',value=p0_a[5])
    
    parameters_b = lmfit.Parameters()
    parameters_b.add('flux', value=p0_b[0])
    parameters_b.add('hlr', value=p0_b[1])
    parameters_b.add('e1',value=p0_b[2],min=-lim,max=lim)
    parameters_b.add('e2',value=p0_b[3],min=-lim,max=lim)
    #parameters.add('e1_b', value=p0[2],min=-0.99,max=0.99)
    #parameters.add('e2_b',expr='sqrt(mag**2 - e1_b**2)')
    parameters_b.add('x0',value=p0_b[4])
    parameters_b.add('y0',value=p0_b[5])
    
    
    # Extract params that minimize the difference of the data from the model.
    result_a = lmfit.minimize(Library.residual_1_obj, parameters_a, args=(children[0], sky_level, x_len, y_len, pixel_scale, sersic_func, n_a))
    
    result_b = lmfit.minimize(Library.residual_1_obj, parameters_b, args=(children[1], sky_level, x_len, y_len, pixel_scale, sersic_func, n_a))                               
    
    # Plot the data if necessary
    if plot != False:
        gs = gridspec.GridSpec(2,8)                                   
        fig = plt.figure(figsize=(15,11))
        sh = 0.8
        plt.suptitle('True Objects vs Deblended Objects')
        ax1 = fig.add_subplot(gs[0,0:2])
        a = ax1.imshow(image_a.array,interpolation='none',origin='lower'); plt.title('Object A'); plt.colorbar(a,shrink=sh)
        ax2 = fig.add_subplot(gs[1,0:2])
        b = ax2.imshow(image_b.array,interpolation='none',origin='lower'); plt.title('Object B'); plt.colorbar(b,shrink=sh)
        ax3 = fig.add_subplot(gs[0,2:4])
        c = ax3.imshow(children[0],interpolation='none',origin='lower'); plt.title('Child A'); plt.colorbar(c,shrink=sh)
        ax4 = fig.add_subplot(gs[1,2:4])
        d = ax4.imshow(children[1],interpolation='none',origin='lower'); plt.title('Child B'); plt.colorbar(d,shrink=sh)
        ax5 = fig.add_subplot(gs[:,4:])
        e = ax5.imshow(tot_image.array,interpolation='none',origin='lower'); plt.title('Original Blend'); plt.colorbar(e,shrink=sh)
        plt.show()
    
    results_deblend = pd.Series(np.array([result_a.params['flux'].value,
                                          result_a.params['hlr'].value,
                                          result_a.params['e1'].value,
                                          result_a.params['e2'].value,
                                          result_a.params['x0'].value,
                                          result_a.params['y0'].value,
                                          result_b.params['flux'].value,
                                          result_b.params['hlr'].value,
                                          result_b.params['e1'].value,
                                          result_b.params['e2'].value,
                                          result_b.params['x0'].value,
                                          result_b.params['y0'].value]),
                                          index=['flux_a','hlr_a','e1_a','e2_a','x0_a','y0_a',
                                                 'flux_b','hlr_b','e1_b','e2_b','x0_b','y0_b'])
    
    # Now estimate the shape for each true object
    result_a_true = lmfit.minimize(Library.residual_1_obj, parameters_a, args=(image_c.array, sky_level, x_len, y_len, pixel_scale, sersic_func, n_a))
    
    result_b_true = lmfit.minimize(Library.residual_1_obj, parameters_b, args=(image_d.array, sky_level, x_len, y_len, pixel_scale, sersic_func, n_a))

    results_true = pd.Series(np.array([result_a_true.params['flux'].value,
                                       result_a_true.params['hlr'].value,
                                       result_a_true.params['e1'].value,
                                       result_a_true.params['e2'].value,
                                       result_a_true.params['x0'].value,
                                       result_a_true.params['y0'].value,
                                       result_b_true.params['flux'].value,
                                       result_b_true.params['hlr'].value,
                                       result_b_true.params['e1'].value,
                                       result_b_true.params['e2'].value,
                                       result_b_true.params['x0'].value,
                                       result_b_true.params['y0'].value]),
                                       index=['flux_a','hlr_a','e1_a','e2_a','x0_a','y0_a',
                                              'flux_b','hlr_b','e1_b','e2_b','x0_b','y0_b'])
    
    return results_deblend, results_true                    


# In[19]:

# Simple function that runs a loop over the main function above and plots
# the bias on ellipticities.    
def run_sample_batch(num_trials_deblender, num_trials_simult_fitter, run_simult_fit=False):

    # Parameters for object a
    flux_a = 2500 # counts
    hlr_a = 1  # arcsec
    e1_a = 0.0 
    e2_a = 0.0
    x0_a = -0.5  # arcsec
    y0_a = 0   # arcsec
    n_a = 0.5
    
    # Parameters for object b
    flux_b = 2500 # counts
    hlr_b = hlr_a # arcsec
    e1_b = 0.0
    e2_b = 0.0
    x0_b = 0.5   # arcsec
    y0_b = 0   # arcsec
    n_b = 0.5
    
    truth = pd.Series(np.array([flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,
                                flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b]),
                                index=['flux_a','hlr_a','e1_a','e2_a','x0_a','y0_a',
                                       'flux_b','hlr_b','e1_b','e2_b','x0_b','y0_b'])
    
    # Galsim function definitions
    sersic_func = galsim.Sersic
    
    # Set the RNG
    seed_1 = galsim.BaseDeviate(0)
    seed_2 = galsim.BaseDeviate(0)
    seed_3 = galsim.BaseDeviate(0)
    
    # Image properties
    pixel_scale = 1/5     # arcsec / pixel
    x_len = y_len = 100            # pixel
    
    # Use LSST defined sky noise for r-band
    add_noise_flag = True
    texp = 6900 # seconds;
    sbar = 26.8 # sky photons per second per pixel
    sky_level = 0 # For now
    sky_noise = np.sqrt(sky_level)
    texp = 0; # For only poisson noise
    
    # psf properties
    psf_flag = False
    beta = 3
    fwhm_psf = 0.6
    
    # FFT
    method = 'fft'
    
    # Store the results of the fits to the deblended children
    results_deblend = []
    results_true = []
    for i in xrange(0,num_trials_deblender):
        results_deb, results_tr = de_blend(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,n_a,
                                             flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,n_b,
                                             sersic_func, seed_1, seed_2, seed_3, 
                                             pixel_scale, x_len, y_len, 
                                             add_noise_flag, texp, sbar, 
                                             psf_flag, beta, fwhm_psf,
                                             method,plot=False)
        results_deblend.append(results_deb)
        results_true.append(results_tr)
    results_deblend = pd.DataFrame(results_deblend)
    results_true = pd.DataFrame(results_true)

    
    if run_simult_fit == True:
        # Run the simultaneous fitter
        results_sim = []
        for i in xrange(0,num_trials_simult_fitter):
            im_no_noise, im_noise, best_fit, lm_results = Library.run_2_galaxy_full_params_simple(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,n_a,
                                                                                                  flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,n_b,
                                                                                                  psf_flag,beta,fwhm_psf,
                                                                                                  x_len,y_len,pixel_scale,sersic_func,sersic_func,seed_1,seed_2,seed_3,
                                                                                                  add_noise_flag,sky_level)    
            #pdb.set_trace()                                                                                      
            results_sim.append(rearrange_lmfit_2obj(lm_results))

        results_sim = pd.DataFrame(results_sim)
        return results_deblend, results_true, results_sim, truth

    
    return results_deblend, results_true, truth
    
# Input an lmfit minimize object to return the values in sorted order    
def rearrange_lmfit_2obj(result):
    arr = np.array([result.params['flux_a'].value,result.params['hlr_a'].value,result.params['e1_a'].value,result.params['e2_a'].value,result.params['x0_a'].value,result.params['y0_a'].value,
                    result.params['flux_b'].value,result.params['hlr_b'].value,result.params['e1_b'].value,result.params['e2_b'].value,result.params['x0_b'].value,result.params['y0_b'].value])   
    arr = pd.Series(arr,index=['flux_a','hlr_a','e1_a','e2_a','x0_a','y0_a',
                               'flux_b','hlr_b','e1_b','e2_b','x0_b','y0_b'])
    return arr

if __name__ == '__main__':
    # In[25]:
    
    results_deblend, results_true, results_sim, truth = run_sample_batch(12,10,run_simult_fit=True)
    
    
    ### We now show the triangle plot of the results of single fits to the child objects.
    
    # In[27]:
    
    figure_1 = triangle.corner(results_deblend,labels=results_deblend.columns,truths=truth,
                             show_titles=True, title_args={'fontsize':20})
    figure_1.gca().annotate('Results of Single Fits to Deblended Children', xy=(0.5,1.0), xycoords='figure fraction')
    
    
    ### We now show the triangle plot for the results of single fits to the true objects.
    
    # In[29]:
    
    
    figure_2 = triangle.corner(results_true,labels=results_true.columns,truths=truth,
                             show_titles=True, title_args={'fontsize':20})
    figure_2.gca().annotate('Results of Single Fits to True Objects', xy=(0.5,1.0), xycoords='figure fraction')
    
    
    ### We now show the triangle plot of the results of simultaenous fitting
    
    # In[30]:
    
    figure_3 = triangle.corner(results_sim,labels=results_sim.columns,truths=truth,
                               show_titles=True, title_args={'fontsize':20})
    figure_3.gca().annotate('Results of Simultaneous Fits', xy=(0.5,1.0), xycoords='figure fraction')

    
    
