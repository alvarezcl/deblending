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

# <codecell>
# Create a blend, deblend, then estimate ellipticity of deblended objects.
def main(argv,plot=False):
    
    # Parameters for object a
    flux_a = 2500 # counts
    hlr_a = 1  # arcsec
    e1_a = 0.0 
    e2_a = 0.0
    x0_a = -0.2  # arcsec
    y0_a = 0.2   # arcsec
    n_a = 0.5
    
    # Parameters for object b
    flux_b = 2500 # counts
    hlr_b = hlr_a # arcsec
    e1_b = 0.0
    e2_b = 0.0
    x0_b = 0.4   # arcsec
    y0_b = 0   # arcsec
    n_b = 0.5
    
    # Galsim function definitions
    sersic_func = galsim.Sersic
    
    # Set the RNG
    seed_1 = galsim.BaseDeviate(0)
    seed_2 = galsim.BaseDeviate(0)
    
    # Image properties
    pixel_scale = 1/5     # arcsec / pixel
    x_len = y_len = 100            # pixel
    
    # Use LSST defined sky noise for r-band
    add_noise_flag = False
    texp = 6900 # seconds
    sbar = 26.8 # sky photons per second per pixel
    sky_level = texp*sbar
    sky_noise = np.sqrt(sky_level)
    
    # psf properties
    psf_flag = False
    beta = 3
    fwhm_psf = 0.6    
    
    # Create the objects and combine them
    # Use photon shooting for now
    method = 'phot'
    image_a = Library.create_galaxy(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,galtype_gal=sersic_func,sersic_index=n_a,
                                    x_len=x_len,y_len=y_len,scale=pixel_scale,
                                    psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                                    method=method, seed=seed_1)
                                
    image_b = Library.create_galaxy(flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,galtype_gal=sersic_func,sersic_index=n_b,
                                    x_len=x_len,y_len=y_len,scale=pixel_scale,
                                    psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                                    method=method, seed=seed_2)
    
    tot_image = image_a + image_b
        
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
    best_fit_a = Library.create_galaxy(result_a.params['flux'].value,
                                       result_a.params['hlr'].value,
                                       result_a.params['e1'].value,
                                       result_a.params['e2'].value,
                                       result_a.params['x0'].value,
                                       result_a.params['y0'].value,
                                       x_len=x_len,y_len=y_len,scale=pixel_scale,galtype_gal=sersic_func,sersic_index=n_a)
    
    result_b = lmfit.minimize(Library.residual_1_obj, parameters_b, args=(children[1], sky_level, x_len, y_len, pixel_scale, sersic_func, n_a))                               
    best_fit_b = Library.create_galaxy(result_b.params['flux'].value,
                                       result_b.params['hlr'].value,
                                       result_b.params['e1'].value,
                                       result_b.params['e2'].value,
                                       result_b.params['x0'].value,
                                       result_b.params['y0'].value,
                                       x_len=x_len,y_len=y_len,scale=pixel_scale,galtype_gal=sersic_func,sersic_index=n_b)
    
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
    
    residual = pd.Series(np.array([result_a.params['e1'].value - e1_a,
                                   result_a.params['e2'].value - e2_a,
                                   result_b.params['e1'].value - e1_b,
                                   result_b.params['e2'].value - e2_b]),
                                   index=['e1_ar','e2_ar','e1_br','e2_br'])
    
    return residual                    


# <codecell>
# Simple function that runs a loop over the main function above and plots
# the bias on ellipticities.    
def run_sample_batch(num_trials):
    results = []
    for i in xrange(0,num_trials):
        residuals_e = main(0,plot=False)
        results.append(residuals_e.values)
    
    sqr_N = np.sqrt(num_trials)    
    results = pd.DataFrame(results,columns=['e1_ar','e2_ar','e1_br','e2_br'])
    stats = np.array([[np.mean(results['e1_ar']),np.mean(results['e2_ar']),np.mean(results['e1_br']),np.mean(results['e2_br'])],
                      [np.std(results['e1_ar']), np.std(results['e2_ar']), np.std(results['e1_br']), np.std(results['e2_br'])],
                      [np.std(results['e1_ar'])/sqr_N, np.std(results['e2_ar'])/sqr_N, np.std(results['e1_br'])/sqr_N, np.std(results['e2_br'])/sqr_N]])
    stats = pd.DataFrame(stats,columns=['e1_ar','e2_ar','e1_br','e2_br'],index=['mean','sigma_data','sigma_mean'])                  

    # Histogram the data TODO    
    bin_nums = num_trials/20;
    gs = gridspec.GridSpec(2,2)                                   
    fig = plt.figure(figsize=(15,11))
    fs = 17
    plt.suptitle('Histograms of Ellipticity Residuals',fontsize=fs)
    ax1 = fig.add_subplot(gs[0,0])
    a = ax1.hist(results['e1_ar'],bins=bin_nums,histtype=u'step'); plt.title('$e1_a$ Residual'); plt.xlabel('Ellipticity Residual',fontsize=fs); plt.ylabel('Occurrence',fontsize=fs)
    ax2 = fig.add_subplot(gs[1,0])
    b = ax2.hist(results['e2_ar'],bins=bin_nums,histtype=u'step'); plt.title('$e2_a$ Residual'); plt.xlabel('Ellipticity Residual',fontsize=fs); plt.ylabel('Occurrence',fontsize=fs)
    ax3 = fig.add_subplot(gs[0,1])
    c = ax3.hist(results['e1_br'],bins=bin_nums,histtype=u'step'); plt.title('$e1_b$ Residual'); plt.xlabel('Ellipticity Residual',fontsize=fs); plt.ylabel('Occurrence',fontsize=fs)
    ax4 = fig.add_subplot(gs[1,1])
    d = ax4.hist(results['e2_br'],bins=bin_nums,histtype=u'step'); plt.title('$e2_b$ Residual'); plt.xlabel('Ellipticity Residual',fontsize=fs); plt.ylabel('Occurrence',fontsize=fs)        
    
    return results, stats, fig

if __name__ == '__main__':
    main(sys.argv)    