# ----------------------------------------------------------------------------
# Filename: Library.py
# Set of library functions for image processing.

from __future__ import division
import galsim
import numpy as np
import lmfit
import deblend
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import scipy
import triangle
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import ipdb

# Create a galaxy with a sersic profile and optional psf to the image. 
def create_galaxy(flux, hlr, e1, e2, x0, y0, galtype_gal=galsim.Sersic, sersic_index=0.5,
                  psf_flag=False, psf_type=galsim.Moffat, beta=5, size_psf=1, flux_psf=1,
                  x_len=100, y_len=100, scale=0.2, method='fft',seed=None,
                  verbose=False, max_fft_size=1000000, return_obj=False):
                  
    big_fft_params = galsim.GSParams(maximum_fft_size=max_fft_size)
    
    if verbose == True:
        print "\nPostage Stamp is", x_len, "by", y_len, "with\na scale of", scale,"\"/Pixel"    
        
    if galtype_gal is galsim.Sersic:
        assert sersic_index != 0
        if verbose == True:        
            if sersic_index == 0.5:
                print "\nThe object drawn is a gaussian with n = 0.5" 
        gal = galtype_gal(n=sersic_index, half_light_radius=hlr, flux=flux, gsparams=big_fft_params)
        gal = gal.shear(g1=e1, g2=e2)
        gal = gal.shift(x0,y0)
        if return_obj == True:
            return gal
        image = galsim.ImageD(x_len, y_len, scale=scale)
        if psf_flag == True:
            psf_gal = convolve_with_psf(gal, beta=beta, size_psf=size_psf, psf_type=psf_type, flux_psf=flux_psf,
                                        verbose=verbose, max_fft_size=max_fft_size)
            if method == 'fft':
                image = psf_gal.drawImage(image=image, method=method)
            else:
                image = psf_gal.drawImage(image=image, method=method,rng=seed)
            return image
        else:
            if method == 'fft':
                image = gal.drawImage(image=image, method=method)
            else:
                image = gal.drawImage(image=image, method=method,rng=seed)
            return image    
        
    else:
        raise ValueError("Not using a sersic profile for the object.")

# Create a HST galaxy
def create_real_galaxy(flux,x0,y0,
                       beta,fwhm_psf,
                       x_len,y_len,pixel_scale,
                       rng):
    
    directory = '/home/luis/Documents/Research/deblending/cs231a/images'
    file_name = 'real_galaxy_catalog.fits'
    my_rgc = galsim.RealGalaxyCatalog(file_name,dir=directory)
    
    rg = galsim.RealGalaxy(my_rgc, index=None, id=None, random=True, 
                           rng=rng, x_interpolant=None, k_interpolant=None,
                           flux=flux, pad_factor=4, noise_pad_size=0,
                           gsparams=None)
                                    
    rg = rg.shift(x0,y0)
    psf_gal = convolve_with_psf(rg,beta,fwhm_psf)
    # Draw Image
    image = galsim.ImageD(x_len, y_len, scale=pixel_scale)
    gal = psf_gal.drawImage(image=image)
    return gal

# Calculate the flux to SNR mapping for real galaxies
def fluxFromSNR_Real(x0_a, y0_a, x0_b, y0_b,
                     beta,fwhm_psf,
                     x_len,y_len,pixel_scale,
                     texp,sbar,weight,
                     rng,increment,range):
    
    data = {'Flux_tot':[],'SNR':[],'Frac_pix':[]}
    flux_a = 0
    flux_b = 0
    snr_val = 0
    
    while snr_val < range:
        flux_a += increment
        flux_b = flux_a
        data['Flux_tot'].append(flux_a+flux_b)
        # Obtain instantiation
        im_a = create_real_galaxy(flux_a, x0_a, y0_a, 
                                  beta,fwhm_psf,
                                  x_len,y_len,pixel_scale,
                                  rng)  
        
        im_b = create_real_galaxy(flux_b, x0_b, y0_b, 
                                  beta,fwhm_psf,
                                  x_len,y_len,pixel_scale,
                                  rng)  
        
        im = im_a + im_b
        snr_val, mask = calcSNR(im, texp, sbar, weight)
        data['SNR'].append(snr_val)
        pix_count_image = (im.array > 0).sum()
        pix_count_masked_image = (mask > 0).sum()
        fractional_pix_count = pix_count_masked_image/pix_count_image
        print snr_val
        data['Frac_pix'].append(fractional_pix_count)
        
    
    snr_points = np.array(data['SNR']); flux_pts = np.array(data['Flux_tot']) 
    cond = np.logical_and(snr_points > 0, snr_points < 150)
    flux_pts = flux_pts[cond]
    snr_points = snr_points[cond]
    
    SNR_to_flux = interpolate.interp1d(snr_points,flux_pts,kind='cubic')
    
    return data['SNR'], data['Flux_tot'], SNR_to_flux

# Convolve an object with a PSF.
def convolve_with_psf(gal, beta, size_psf, psf_type=galsim.Moffat, flux_psf=1, 
                      verbose=False, max_fft_size=100000):
    big_fft_params = galsim.GSParams(maximum_fft_size=max_fft_size)
    if verbose == True:
        print "Using a psf with beta =", beta,"and size = ", size_psf," \"" 
    psf = psf_type(beta=beta, fwhm=size_psf, flux=flux_psf, gsparams=big_fft_params)
    psf_gal = galsim.Convolve([gal,psf])
    return psf_gal

# Add poisson noise to an image with a sky level argument.        
def add_noise(image, noise_type=galsim.PoissonNoise, seed=None, sky_level=0):
    if noise_type is galsim.PoissonNoise:    
        image.addNoise(noise_type(sky_level=sky_level,rng=seed))
        return image
    else:
        raise ValueError("Not using poisson noise in your image.")
        
# Residual function for fitting one model object to the data.
def residual_1_obj(param, data_image, sky_level, x_len, y_len, pixel_scale, 
                   galtype,n):
                       
    assert galtype != None
    flux = param['flux'].value
    hlr = param['hlr'].value
    e1 = param['e1'].value
    e2 = param['e2'].value
    x0 = param['x0'].value
    y0 = param['y0'].value
    image = create_galaxy(flux,hlr,e1,e2,x0,y0,galtype_gal=galtype,sersic_index=n,
                          x_len=x_len,y_len=y_len,scale=pixel_scale)
    
    return (image-data_image).array.ravel()

# Residual function for fitting two objects to data.    
def residual_func_simple(param, data_image, sky_level, x_len, y_len, pixel_scale, 
                         galtype_a,n_a,galtype_b,n_b):
        
    assert galtype_a != None
    assert galtype_b != None
    flux_a = param['flux_a'].value
    hlr_a = param['hlr_a'].value
    e1_a = param['e1_a'].value
    e2_a = param['e2_a'].value
    x0_a = param['x0_a'].value
    y0_a = param['y0_a'].value

    flux_b = param['flux_b'].value
    hlr_b = param['hlr_b'].value
    e1_b = param['e1_b'].value
    e2_b = param['e2_b'].value
    x0_b = param['x0_b'].value
    y0_b = param['y0_b'].value
    image_a = create_galaxy(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,galtype_gal=galtype_a,sersic_index=n_a,
                            x_len=x_len,y_len=y_len,scale=pixel_scale)
                            
    image_b = create_galaxy(flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,galtype_gal=galtype_b,sersic_index=n_b,
                            x_len=x_len,y_len=y_len,scale=pixel_scale)
                        
    image = image_a + image_b
    
    if sky_level > 10:        
        return (data_image-image).array.ravel()/np.sqrt(sky_level + image.array).ravel()
    else:
        return (data_image-image).array.ravel()
        
# Function definition to return the original data array, best-fit array,
# residual, and correlation matrix with differences and error on e1 and e2.
def run_2_galaxy_full_params_simple(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,n_a,
                                    flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,n_b,
                                    psf_flag,beta,fwhm_psf,
                                    x_len,y_len,pixel_scale,galtype_a,galtype_b,seed_a,seed_b,seed_p,
                                    add_noise_flag,sky_level):

    image_a = create_galaxy(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,galtype_gal=galtype_a,sersic_index=n_a,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf)
                                
    image_b = create_galaxy(flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,galtype_gal=galtype_b,sersic_index=n_b,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf)
    
    image_no_noise = image_a + image_b                        
    image = image_a + image_b
    if add_noise_flag == True:
        image_noise = add_noise(image,seed=seed_p,sky_level=sky_level)
        image = image_noise
    else:
        image_noise = image
    
    # -----------------------------------------------------------------------    
    # Estimate the parameters of the image.
    
    # Define some seed that's far from true values and insert into
    # lmfit object for galaxy one and two
    lim = 1/np.sqrt(2)
    p0 = 1.0*np.array([flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,
                       flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b])
    parameters = lmfit.Parameters()
    parameters.add('flux_a', value=p0[0])
    parameters.add('hlr_a', value=p0[1])
    #parameters.add('e1_a', value=p0[2],min=-0.99,max=0.99)
    #parameters.add('mag',value=0.99,max=0.99,vary=True)
    #parameters.add('e2_a',expr='sqrt(mag**2 - e1_a**2)')
    parameters.add('e1_a',value=p0[2],min=-lim,max=lim)
    parameters.add('e2_a',value=p0[3],min=-lim,max=lim)    
    parameters.add('x0_a',value=p0[4])
    parameters.add('y0_a',value=p0[5])
    
    parameters.add('flux_b', value=p0[6])
    parameters.add('hlr_b', value=p0[7])
    parameters.add('e1_b',value=p0[8],min=-lim,max=lim)
    parameters.add('e2_b',value=p0[9],min=-lim,max=lim)
    #parameters.add('e1_b', value=p0[2],min=-0.99,max=0.99)
    #parameters.add('e2_b',expr='sqrt(mag**2 - e1_b**2)')
    parameters.add('x0_b',value=p0[10])
    parameters.add('y0_b',value=p0[11])
    
    
    # Extract params that minimize the difference of the data from the model.
    result = lmfit.minimize(residual_func_simple, parameters, args=(image, sky_level, x_len, y_len, pixel_scale, galtype_a, n_a, galtype_b, n_b))                                   
                                                                      
    return image_no_noise, image_noise, result

# Convert python file to ipython notebook document format.    
def to_ipynb(infile,outfile):
    # infile; input python file <foo.py>
    # outfile; output python file <foo.ipynb>
    import IPython.nbformat.current as nbf
    nb = nbf.read(open(infile, 'r'), 'py')
    nbf.write(nb, open(outfile, 'w'), 'ipynb')    

# Create a blend, deblend, then estimate ellipticity of deblended objects and true objects.
def deblend_estimate(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,n_a,
                     flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,n_b,
                     truth,
                     func, seed_1, seed_2, seed_3,
                     pixel_scale, x_len, y_len,
                     add_noise_flag, texp, sbar,
                     psf_flag, beta, fwhm_psf,
                     method,
                     factor_init,
                     plot=False):
                         
    sky_level = texp*sbar
    sersic_func = func


    # Create the objects and combine them
    image_a = create_galaxy(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,galtype_gal=func,sersic_index=n_a,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                            method=method, seed=seed_1)

    image_b = create_galaxy(flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,galtype_gal=func,sersic_index=n_b,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                            method=method, seed=seed_2)


    tot_image = image_a + image_b
    if add_noise_flag == True:
        image_noise = add_noise(tot_image,seed=seed_3,sky_level=sky_level)
        image_c = add_noise(image_a,seed=seed_3,sky_level=sky_level)
        image_d = add_noise(image_b,seed=seed_3,sky_level=sky_level)
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
    p0_a = factor_init*np.array([flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a])
    p0_b = factor_init*np.array([flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b])

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
    result_a = lmfit.minimize(residual_1_obj, parameters_a, args=(children[0], sky_level, x_len, y_len, pixel_scale, sersic_func, n_a))

    result_b = lmfit.minimize(residual_1_obj, parameters_b, args=(children[1], sky_level, x_len, y_len, pixel_scale, sersic_func, n_a))

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

    p0_a_t = (truth['flux_a'],truth['hlr_a'],truth['e1_a'],truth['e2_a'],truth['x0_a'],truth['y0_a'])
    p0_b_t = (truth['flux_b'],truth['hlr_b'],truth['e1_b'],truth['e2_b'],truth['x0_b'],truth['y0_b'])

    image_a_t = create_galaxy(flux_a,hlr_a,e1_a,e2_a,p0_a[4],p0_a[5],galtype_gal=func,sersic_index=n_a,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                            method=method, seed=seed_1)

    image_b_t = create_galaxy(flux_b,hlr_b,e1_b,e2_b,p0_b[4],p0_b[5],galtype_gal=func,sersic_index=n_b,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                            method=method, seed=seed_2)

    image_a_t = add_noise(image_a_t,seed=seed_3,sky_level=sky_level)
    image_b_t = add_noise(image_b_t,seed=seed_3,sky_level=sky_level)

    parameters_a = lmfit.Parameters()
    parameters_a.add('flux', value=p0_a_t[0])
    parameters_a.add('hlr', value=p0_a_t[1])
    #parameters.add('e1_a', value=p0[2],min=-0.99,max=0.99)
    #parameters.add('mag',value=0.99,max=0.99,vary=True)
    #parameters.add('e2_a',expr='sqrt(mag**2 - e1_a**2)')
    parameters_a.add('e1',value=p0_a_t[2],min=-lim,max=lim)
    parameters_a.add('e2',value=p0_a_t[3],min=-lim,max=lim)
    parameters_a.add('x0',value=p0_a_t[4])
    parameters_a.add('y0',value=p0_a_t[5])

    parameters_b = lmfit.Parameters()
    parameters_b.add('flux', value=p0_b_t[0])
    parameters_b.add('hlr', value=p0_b_t[1])
    parameters_b.add('e1',value=p0_b_t[2],min=-lim,max=lim)
    parameters_b.add('e2',value=p0_b_t[3],min=-lim,max=lim)
    #parameters.add('e1_b', value=p0[2],min=-0.99,max=0.99)
    #parameters.add('e2_b',expr='sqrt(mag**2 - e1_b**2)')
    parameters_b.add('x0',value=p0_b_t[4])
    parameters_b.add('y0',value=p0_b_t[5])

    # Now estimate the shape for each true object
    result_a_true = lmfit.minimize(residual_1_obj, parameters_a, args=(image_a_t.array, sky_level, x_len, y_len, pixel_scale, sersic_func, n_a))

    result_b_true = lmfit.minimize(residual_1_obj, parameters_b, args=(image_b_t.array, sky_level, x_len, y_len, pixel_scale, sersic_func, n_b))

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

    return results_deblend, results_true, children


# Input an lmfit minimize object to return the values in sorted order
def rearrange_lmfit_2obj(result):
    arr = np.array([result.params['flux_a'].value,result.params['hlr_a'].value,result.params['e1_a'].value,result.params['e2_a'].value,result.params['x0_a'].value,result.params['y0_a'].value,
                    result.params['flux_b'].value,result.params['hlr_b'].value,result.params['e1_b'].value,result.params['e2_b'].value,result.params['x0_b'].value,result.params['y0_b'].value])
    arr = pd.Series(arr,index=['flux_a','hlr_a','e1_a','e2_a','x0_a','y0_a',
                               'flux_b','hlr_b','e1_b','e2_b','x0_b','y0_b'])
    return arr
        
def show_stats(results,runs,value):
    data = pd.DataFrame(np.array([results.mean().values-value,results.std().values,results.std().values/np.sqrt(runs)]),columns=results.columns)
    data.index = [r'$\bar\mu$',r'$\sigma$', r'$\sigma_{\mu}$']
    return data    

# Save the information if results have been statistically independent
def save_data(path,results_deblend,results_true,results_sim):
    # Save files according to type

    dble = path + '/results_deblend.csv'
    with open(dble,'a') as f:
        results_deblend.to_csv(f)
    tru = path + '/results_true.csv'
    with open(tru,'a') as f:
        results_true.to_csv(f)
    sim = path + '/results_sim.csv'
    with open(sim,'a') as f:
        results_sim.to_csv(f)
        
# Calculate the SNR for a given noisy image.        
def calcSNR(im, texp, sbar, weight):
    threshold = weight*np.sqrt(texp*sbar)
    mask = im.array > threshold
    # Now calculate the SNR using the original galsim image
    nu = np.sqrt(1/(sbar*texp))*np.sqrt((mask*im.array**2).sum())
    return nu, mask

# Calculate the function that takes in some SNR and returns total flux.    
def fluxFromSNR(hlr_a,e1_a,e2_a,x0_a,y0_a,n_a,
                hlr_b,e1_b,e2_b,x0_b,y0_b,n_b,
                psf_flag,beta,fwhm_psf,
                x_len,y_len,pixel_scale,
                texp,sbar,increment):
                         
    data = {'Flux_tot':[],'SNR':[],'Frac_pix':[]}
    flux_a = 0
    flux_b = 0
    snr_val = 0
    while snr_val < 30:
        flux_a += increment
        flux_b = flux_a
        data['Flux_tot'].append(flux_a+flux_b)
        # Obtain instantiation
        im_a = create_galaxy(flux_a, hlr_a, e1_a, e2_a, x0_a, y0_a, galtype_gal=galsim.Sersic, sersic_index=n_a,
                             psf_flag=psf_flag, psf_type=galsim.Moffat, beta=beta, size_psf=fwhm_psf, flux_psf=1,
                             x_len=x_len, y_len=y_len, scale=pixel_scale, method='phot',seed=None,
                             verbose=False, max_fft_size=1000000, return_obj=False)
        
        im_b = create_galaxy(flux_b, hlr_b, e1_b, e2_b, x0_b, y0_b, galtype_gal=galsim.Sersic, sersic_index=n_b,
                             psf_flag=psf_flag, psf_type=galsim.Moffat, beta=beta, size_psf=fwhm_psf, flux_psf=1,
                             x_len=x_len, y_len=y_len, scale=pixel_scale, method='phot',seed=None,
                             verbose=False, max_fft_size=1000000, return_obj=False)
        
        im = im_a + im_b
        snr_val, mask = calcSNR(im, texp, sbar, 0.5)
        data['SNR'].append(snr_val)
        pix_count_image = (im.array > 0).sum()
        pix_count_masked_image = (mask > 0).sum()
        fractional_pix_count = pix_count_masked_image/pix_count_image 
        data['Frac_pix'].append(fractional_pix_count)
        
    snr_points = np.array(data['SNR']); flux_pts = np.array(data['Flux_tot']) 
    cond = np.logical_and(snr_points > 0, snr_points < 150)
    flux_pts = flux_pts[cond]
    snr_points = snr_points[cond]
    SNR_to_flux = scipy.interpolate.interp1d(snr_points,flux_pts,kind='cubic')

                            
    return data['SNR'], data['Flux_tot'], SNR_to_flux
    
# Simple function that runs a loop over the main functions and returns 
# key information regarding fits.    
def run_batch(num_trials,
              func,
              seed_1,seed_2,seed_3,
              seed_4,seed_5,seed_6,
              image_params,
              obj_a,obj_b,method,
              sky_info,
              psf_info,
              mod_val,est_centroid,randomize):
    
    assert func == galsim.Sersic, "Not using a sersic profile"
    # Galsim function definitions
    sersic_func = func
    
    # Image properties
    pixel_scale, x_len, y_len = image_params
    
    # Parameters for object a
    flux_a, hlr_a, e1_a, e2_a, x0_a, y0_a, n_a = obj_a 
    
    # Parameters for object b
    flux_b, hlr_b, e1_b, e2_b, x0_b, y0_b, n_b = obj_b
    
    truth = pd.Series(np.array([flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,
                                flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b]),
                                index=['flux_a','hlr_a','e1_a','e2_a','x0_a','y0_a',
                                       'flux_b','hlr_b','e1_b','e2_b','x0_b','y0_b'])
    
    # Use LSST defined sky noise for r-band
    add_noise_flag, texp, sbar, sky_level = sky_info
    
    # psf properties
    psf_flag, beta, fwhm_psf = psf_info

    # Store the results of the fits to the deblended children
    results_deblend = []
    results_true = []
    images = []
    factor_init = 1
    results_sim = []
    x_y_coord = []
    for i in xrange(0,num_trials):
        # First run the simultaneous fitter 
        
        if randomize == True:
            x0_a_r = x0_a + np.random.uniform(-pixel_scale/2,pixel_scale/2)
            x0_b_r = x0_b + np.random.uniform(-pixel_scale/2,pixel_scale/2)
            y0_a_r = y0_a + np.random.uniform(-pixel_scale/2,pixel_scale/2)
            y0_b_r = y0_b + np.random.uniform(-pixel_scale/2,pixel_scale/2)
        else:
            x0_a_r = x0_a
            x0_b_r = x0_b
            y0_a_r = y0_a
            y0_b_r = y0_b
            
        x_y_coord.append(pd.Series([x0_a_r,y0_a_r,x0_b_r,y0_b_r],index=['x0_a_r',
                                                                        'y0_a_r',
                                                                        'x0_b_r',
                                                                        'y0_b_r']))

        im_no_noise, im_noise, lm_results = run_2_galaxy_full_params_simple(flux_a,hlr_a,e1_a,e2_a,x0_a_r,y0_a_r,n_a,
                                                                            flux_b,hlr_b,e1_b,e2_b,x0_b_r,y0_b_r,n_b,
                                                                            psf_flag,beta,fwhm_psf,
                                                                            x_len,y_len,pixel_scale,sersic_func,sersic_func,seed_1,seed_2,seed_3,
                                                                            add_noise_flag,sky_level)                                                                                              
        results_sim.append(rearrange_lmfit_2obj(lm_results))

        if est_centroid == True:
            x0_a_est = np.copy(lm_results.values['x0_a'])
            x0_b_est = np.copy(lm_results.values['x0_b'])
            y0_a_est = np.copy(lm_results.values['y0_a'])
            y0_b_est = np.copy(lm_results.values['y0_b'])            
        else:
            x0_a_est = x0_a
            x0_b_est = x0_b
            y0_a_est = y0_a
            y0_b_est = y0_b        
        
        results_deb, results_tr, children = deblend_estimate(flux_a,hlr_a,e1_a,e2_a,x0_a_est,y0_a_est,n_a,
                                                             flux_b,hlr_b,e1_b,e2_b,x0_b_est,y0_b_est,n_a,
                                                             truth,
                                                             sersic_func, seed_4, seed_5, seed_6,
                                                             pixel_scale, x_len, y_len, 
                                                             add_noise_flag, texp, sbar, 
                                                             psf_flag, beta, fwhm_psf,
                                                             method,
                                                             factor_init)
        results_deblend.append(results_deb)
        results_true.append(results_tr)
        if i%mod_val == 0:
            print i
            images.append([children,i])
    results_deblend = pd.DataFrame(results_deblend)
    results_true = pd.DataFrame(results_true)
    results_sim = pd.DataFrame(results_sim)
    x_y_coord = pd.DataFrame(x_y_coord)
    
    return results_deblend, results_true, results_sim, truth, x_y_coord, images
    
# This function takes in the parameters for overlapping profiles and
# runs over different separation values to plot bias vs separation for
# all three methods.
def run_over_separation(separation,
                        num_trial_arr,
                        func,
                        seed_arr,
                        image_params,
                        obj_a,obj_b,method,
                        sky_info,
                        psf_info,
                        mod_val,est_centroid,randomize,
                        number_run,
                        create_tri_plots,
                        x_sep=True,y_sep=False,
                        right_diag=False,left_diag=False):
                                
    means_e1_a = {}
    means_e2_a = {} 
    means_e1_b = {}
    means_e2_b = {}
    
    s_means_e1_a = {}
    s_means_e2_a = {}
    s_means_e1_b = {}
    s_means_e2_b = {}
    
    for i,sep in enumerate(separation):
        print 'sep = ' + str(sep) + '\"'
        num_trials = num_trial_arr[i]
        if x_sep and not y_sep:
            obj_a[4] = -sep/2
            obj_b[4] = sep/2
        elif not x_sep and y_sep:
            obj_a[5] = -sep/2
            obj_b[5] = sep/2
        elif x_sep and y_sep and right_diag:
            obj_a[4] = -np.cos(np.pi/4)*sep/2
            obj_a[5] = -np.sin(np.pi/4)*sep/2
            obj_b[4] = np.cos(np.pi/4)*sep/2
            obj_b[5] = np.sin(np.pi/4)*sep/2
        elif x_sep and y_sep and left_diag:
            obj_a[4] = -np.cos(np.pi/4)*sep/2
            obj_a[5] = np.sin(np.pi/4)*sep/2
            obj_b[4] = np.cos(np.pi/4)*sep/2
            obj_b[5] = -np.sin(np.pi/4)*sep/2
        
        
        results_deblend, results_true, results_sim, truth, x_y_coord, dbl_im = run_batch(num_trials,
                                                                                         func,
                                                                                         seed_arr[0],seed_arr[1],seed_arr[2],
                                                                                         seed_arr[3],seed_arr[4],seed_arr[5],
                                                                                         image_params,
                                                                                         obj_a,obj_b,method,
                                                                                         sky_info,
                                                                                         psf_info,
                                                                                         mod_val,est_centroid,randomize)
        # Create sub directory to save data from this separation                     
        sub_sub_dir = '/sep:' + str(sep) + ';' + 'num_trials:' + str(num_trials)
        path = number_run + sub_sub_dir
        os.mkdir(path)
        
        # Save data
        save_data(path,results_deblend,results_true,results_sim)        
        
        # Obtain relevant stats
        data_dbl = show_stats(results_deblend,num_trials,truth)
        data_true = show_stats(results_true,num_trials,truth)
        data_simult = show_stats(results_sim,num_trials,truth)
        
         # Save triangle plots
        if create_tri_plots:
            create_triangle_plots(path,sep,num_trials,
                                  results_deblend,data_dbl,
                                  results_true,data_true,
                                  results_sim,data_simult,
                                  truth,
                                  x_y_coord,randomize)
                             
        # Save a random image from the set of deblended images
        save_image(path,results_deblend,dbl_im,np.copy(image_params),truth,sep)
        
        # Obtain the mean values with error on mean values
        index = 0
        means_e1_a[str(sep)] = np.array([data_dbl['e1_a'][index],data_true['e1_a'][index],data_simult['e1_a'][index]])
        means_e2_a[str(sep)] = np.array([data_dbl['e2_a'][index],data_true['e2_a'][index],data_simult['e2_a'][index]])
        means_e1_b[str(sep)] = np.array([data_dbl['e1_b'][index],data_true['e1_b'][index],data_simult['e1_b'][index]])
        means_e2_b[str(sep)] = np.array([data_dbl['e2_b'][index],data_true['e2_b'][index],data_simult['e2_b'][index]])
        index = 2
        s_means_e1_a[str(sep)] = np.array([data_dbl['e1_a'][index],data_true['e1_a'][index],data_simult['e1_a'][index]])
        s_means_e2_a[str(sep)] = np.array([data_dbl['e2_a'][index],data_true['e2_a'][index],data_simult['e2_a'][index]])
        s_means_e1_b[str(sep)] = np.array([data_dbl['e1_b'][index],data_true['e1_b'][index],data_simult['e1_b'][index]])
        s_means_e2_b[str(sep)] = np.array([data_dbl['e2_b'][index],data_true['e2_b'][index],data_simult['e2_b'][index]])
    
    # Convert to DataFrame objects
    index = ['Deblending','True','Simultaneous Fitting']
    means_e1_a = pd.DataFrame(means_e1_a,index=index)
    means_e2_a = pd.DataFrame(means_e2_a,index=index)
    means_e1_b = pd.DataFrame(means_e1_b,index=index)
    means_e2_b = pd.DataFrame(means_e2_b,index=index)
    s_means_e1_a = pd.DataFrame(s_means_e1_a,index=index)
    s_means_e2_a = pd.DataFrame(s_means_e2_a,index=index)
    s_means_e1_b = pd.DataFrame(s_means_e1_b,index=index)
    s_means_e2_b = pd.DataFrame(s_means_e2_b,index=index)
    
    
    means = {'means_e1_a':means_e1_a,'means_e2_a':means_e2_a,
             'means_e1_b':means_e1_b,'means_e2_b':means_e2_b}
             
    s_means = {'s_means_e1_a':s_means_e1_a,'s_means_e2_a':s_means_e2_a,
             's_means_e1_b':s_means_e1_b,'s_means_e2_b':s_means_e2_b}
             
    return means, s_means
    
# Create an information string for identifiers 
def join_info(directory,
              separation,
              num_trial_arr,
              func,
              seed_arr,
              image_params,
              obj_a,obj_b,method,
              sky_info,
              psf_info,
              mod_val,use_est_centroid,randomize,
              x_sep,y_sep,
              left_diag,right_diag,
              create_tri_plots,):
                
    if x_sep and y_sep and not right_diag and not left_diag: assert False, "Choose a diagonal."                                    
    if x_sep and y_sep: assert right_diag != left_diag, "Can't run through both diagonals of the image."                  
                  
    print "\nSaving data and images to the following directory: \"" + directory + '\"\n'                   
                  
    if x_sep and not y_sep:
        direction_str = 'x axis'
        print "Moving objects along " + direction_str + '\n'
    elif not x_sep and y_sep:
        direction_str = 'y axis'
        print "Moving objects along " + direction_str + '\n'
    elif x_sep and y_sep and right_diag:
        direction_str = 'right diagonal'
        print "Moving objects along " + direction_str + '\n'
    elif x_sep and y_sep and left_diag:
        direction_str = 'left diagonal'
        print "Moving objects along " + direction_str + '\n'
        
    if psf_info[0]: 
        print "Convolving objects with PSF\n"
    else:
        print "Not convolving objects with PSF\n"
    if use_est_centroid:
        print "Not using prior on (x,y) for both objects\n"
    else:
        print "Using prior on (x,y) for both objects\n"
    if randomize:
        print "Randomizing centroids a pixel about median separation\n"
    else:
        "Not Randomizing centroids a pixel about median separation\n"
    if create_tri_plots:
        print "Creating triangle plots\n"
    else:
        print "Not creating triangle plots\n"        
        
        
    sub_dir = ('x_y_prior = ' + str(not use_est_centroid) + '\n' +
              'randomized_x_y = ' + str(randomize) + '\n' +
              'sep = ' + str(separation) + ' (arcsec)' + '\n' +
              'num_trial_arr = ' + str(num_trial_arr) + ' Number of trials for each separation' + '\n' + 
              'seed_arr = ' + str(seed_arr) + ' Insert into galsim.BaseDeviate' + '\n' +
              'image_params = ' + str(image_params) + ' (pixel_scale,x_len,y_len) of image' + '\n' + 
              'obj_a_info = ' + str(obj_a) + ' (flux,hlr,e1,e2,x0,y0)' + '\n' +
              'obj_b_info = ' + str(obj_b) + ' (flux,hlr,e1,e2,x0,y0)' + '\n' +
              'sky_info = ' + str(sky_info) + ' (flag,texp,sbar,sky_level)' + '\n'
              'psf_info = ' + str(psf_info) + ' (flag,beta,fwhm)' + '\n' +
              'separation_axis = ' + direction_str)
            
    return sub_dir
    
def create_read_me(info_str,number_run):
    os.mkdir(number_run)
    file = open(number_run + '/Run_Information.txt','w+')
    file.write(info_str)
    file.close()
    
def create_triangle_plots(path,sep,num_trials,
                          results_deblend,data_dbl,
                          results_true,data_true,
                          results_sim,data_sim,
                          truth,
                          x_y_coord,randomize):
    
    max_sigma = pd.Series(np.maximum(np.copy(data_dbl.values[1,:]),
                                     np.copy(data_sim.values[1,:])),
                          index=['flux_a','hlr_a','e1_a','e2_a','x0_a','y0_a',
                                 'flux_b','hlr_b','e1_b','e2_b','x0_b','y0_b']) 
                                     
    extents = create_extents(2.25,max_sigma,truth,randomize)                          
    ipdb.set_trace()
    if randomize == True:
        # Join x_y random true locations 
        true_tri = pd.concat([results_true,x_y_coord],axis=1)
        sim_tri = pd.concat([results_sim,x_y_coord],axis=1)
        dbl_tri = pd.concat([results_deblend,x_y_coord],axis=1)
        extents = extents + [(x_y_coord['x0_a_r'].min(),x_y_coord['x0_a_r'].max()),
                             (x_y_coord['y0_a_r'].min(),x_y_coord['y0_a_r'].max()),
                             (x_y_coord['x0_b_r'].min(),x_y_coord['x0_b_r'].max()),
                             (x_y_coord['y0_b_r'].min(),x_y_coord['y0_b_r'].max())]
        rand_xy = pd.Series([truth['x0_a'],truth['y0_a'],
                             truth['x0_b'],truth['y0_b']],
                             index=['x0_a_r','y0_a_r',
                                    'x0_b_r','y0_b_r'])
        truth = truth.append(rand_xy)
    else: 
        true_tri = results_true
        sim_tri = results_sim
        dbl_tri = results_deblend
    
    print "Saving triangle plots"    
    #fig_tru = triangle.corner(true_tri,extents=extents)
    #plt.suptitle('Triangle Plot for Fits to the True Objects\n for a Separation of ' + str(sep) + '\" and ' + str(num_trials) + ' Trials',fontsize=42)
    #plt.savefig(path + '/true_fit_triangle_plot.png')
    #plt.clf()
    #plt.close()
    
    fig_sim = triangle.corner(sim_tri,labels=sim_tri.columns,truths=truth.values,
                              show_titles=True, title_args={'fontsize':20},extents=extents)
    plt.suptitle('Triangle Plot for Simultaneous Fitting to the Deblended Object\n for a Separation of ' + str(sep) + '\" and ' + str(num_trials) + ' Trials',fontsize=42)
    plt.savefig(path + '/simult_fit_triangle_plot.png')
    plt.clf()
    plt.close()
    
    fig_dbl = triangle.corner(dbl_tri,labels=dbl_tri.columns,truths=truth.values,
                              show_titles=True, title_args={'fontsize':20},extents=extents)
    plt.suptitle('Triangle Plot for Fits to the Deblended Objects\n for a Separation of ' + str(sep) + '\" and ' + str(num_trials) + ' Trials',fontsize=42)
    plt.savefig(path + '/dbl_fit_triangle_plot.png')
    plt.clf()
    plt.close()
    print "Finished saving triangle plots"
    
        
def create_extents(factor,max_sigma,truth,randomize):

    flux_interval_a = max_sigma['flux_a']*factor
    hlr_interval_a = max_sigma['hlr_a']*factor
    e1_interval_a = max_sigma['e1_a']*factor
    e2_interval_a = max_sigma['e2_a']*factor
    x0_interval_a = max_sigma['x0_a']*factor
    y0_interval_a = max_sigma['y0_a']*factor
    
    flux_interval_b = max_sigma['flux_b']*factor
    hlr_interval_b = max_sigma['hlr_b']*factor
    e1_interval_b = max_sigma['e1_b']*factor
    e2_interval_b = max_sigma['e2_b']*factor
    x0_interval_b = max_sigma['x0_b']*factor
    y0_interval_b = max_sigma['y0_b']*factor
    
    extents = [(truth['flux_a']-flux_interval_a,truth['flux_a']+flux_interval_a),
               (truth['hlr_a']-hlr_interval_a,truth['hlr_a']+hlr_interval_a),
               (truth['e1_a']-e1_interval_a,truth['e1_a']+e1_interval_a),
               (truth['e2_a']-e2_interval_a,truth['e2_a']+e2_interval_a),
               (truth['x0_a']-x0_interval_a,truth['x0_a']+x0_interval_a),
               (truth['y0_a']-y0_interval_a,truth['y0_a']+y0_interval_a),
               (truth['flux_b']-flux_interval_b,truth['flux_b']+flux_interval_b),
               (truth['hlr_a']-hlr_interval_b,truth['hlr_b']+hlr_interval_b),
               (truth['e1_b']-e1_interval_b,truth['e1_b']+e1_interval_b),
               (truth['e2_b']-e2_interval_b,truth['e2_b']+e2_interval_b),
               (truth['x0_b']-x0_interval_b,truth['x0_b']+x0_interval_b),
               (truth['y0_b']-y0_interval_b,truth['y0_b']+y0_interval_b)]
               
    return extents


def create_bias_plot_e(path,separation,means,s_means,pixel_scale,
                       fs,leg_fs,min_offset,max_offset,psf_flag):
    
    assert len(separation) >= 2, "Separation array must be larger than 1"
         
    def obtain_min_max_df(df_1,df_2,df_3,df_4):
        max_val = np.max([df_1.T.max().values,df_2.T.max().values,df_3.T.max().values,df_4.T.max().values])
        min_val = np.min([df_1.T.min().values,df_2.T.min().values,df_3.T.min().values,df_4.T.min().values])    
        return max_val, min_val
    
        
    def format_df(df,value_1,value_2,index):
        result = pd.concat([pd.DataFrame(np.array([np.NAN,np.NAN,np.NAN]),columns=[str(value_1)],index=index),
                            df,
                            pd.DataFrame(np.array([np.NAN,np.NAN,np.NAN]),columns=[str(value_2)],index=index)],
                           axis=1)
        return result
           
    print "\nSaving bias vs separation plot\n"   
    
    means_e1_a = means['means_e1_a']
    means_e2_a = means['means_e2_a']
    means_e1_b = means['means_e1_b']
    means_e2_b = means['means_e2_b']
    
    s_means_e1_a = s_means['s_means_e1_a']
    s_means_e2_a = s_means['s_means_e2_a']
    s_means_e1_b = s_means['s_means_e1_b']
    s_means_e2_b = s_means['s_means_e2_b']
    
    x_min = np.min(separation) - pixel_scale
    x_max = np.max(separation) + pixel_scale
    
    max_mean, min_mean = obtain_min_max_df(means_e1_a,means_e2_a,means_e1_b,means_e2_b)
    max_s_mean, min_s_mean = obtain_min_max_df(s_means_e1_a,s_means_e2_a,s_means_e1_b,s_means_e2_b)        
    
    gs = gridspec.GridSpec(20,2)
    fig = plt.figure(figsize=(18,16))
    
    if psf_flag: 
        suptitle = 'Ellipticity Bias for Objects a and b\n vs Separation for PSF Convolved Profiles'
        min_sep = 1.6
    else:
        suptitle = 'Ellipticity Bias for Objects a and b\n vs Separation for Profiles with Only Poisson Noise'
        min_sep = 1.4
    plt.suptitle(suptitle,fontsize=fs+6)

    ax1 = fig.add_subplot(gs[0:8,0])
    title = 'e1 on Object a'
    plt.title('Bias vs Separation For ' + title,fontsize=fs)
    plt.ylim([(min_mean - 2*min_s_mean)*min_offset,(max_mean + max_s_mean)*max_offset])
    plt.xlabel('Separation (arcsec)',fontsize=fs)
    plt.ylabel('Residual',fontsize=fs)
    f_m_e1_a = format_df(means_e1_a,x_min,x_max,means_e1_a.index)
    f_s_m_e1_a = format_df(s_means_e1_a,x_min,x_max,s_means_e1_a.index)
    ax1p = f_m_e1_a.T.plot(ax=ax1,style=['k--o','b--o','g--o'],yerr=f_s_m_e1_a.T,legend=True)
    ax1p.legend(loc='upper center', bbox_to_anchor=(0.5,-0.11),
                prop={'size':leg_fs}, shadow=True, ncol=3, fancybox=True)
    ax1p.axhline(y=0,ls='--',c='k',linestyle='--')
    ax1p.axvline(x=min_sep,c='c',linestyle='--')
    
    ax2 = fig.add_subplot(gs[11:19,0])
    title = 'e1 on Object b'
    plt.title('Bias vs Separation For ' + title,fontsize=fs)
    plt.ylim([(min_mean - 2*min_s_mean)*min_offset,(max_mean + max_s_mean)*max_offset])
    plt.xlabel('Separation (arcsec)',fontsize=fs)
    plt.ylabel('Residual',fontsize=fs)
    f_m_e1_b = format_df(means_e1_b,x_min,x_max,means_e1_b.index)
    f_s_m_e1_b = format_df(s_means_e1_b,x_min,x_max,s_means_e1_b.index)
    ax2p = f_m_e1_b.T.plot(ax=ax2,style=['k--o','b--o','g--o'],yerr=f_s_m_e1_b.T,legend=True)
    ax2p.legend(loc='upper center', bbox_to_anchor=(0.5,-0.11),
                prop={'size':leg_fs}, shadow=True, ncol=3, fancybox=True)
    ax2p.axhline(y=0,ls='--',c='k',linestyle='--')
    ax2p.axvline(x=min_sep,c='c',linestyle='--')
    
    ax3 = fig.add_subplot(gs[0:8,1])
    title = 'e2 on Object a'
    plt.title('Bias vs Separation For ' + title,fontsize=fs)
    plt.ylim([(min_mean - 2*min_s_mean)*min_offset,(max_mean + max_s_mean)*max_offset])
    plt.xlabel('Separation (arcsec)',fontsize=fs)
    plt.ylabel('Residual',fontsize=fs)
    f_m_e2_a = format_df(means_e2_a,x_min,x_max,means_e2_a.index)
    f_s_m_e2_a = format_df(s_means_e2_a,x_min,x_max,means_e2_a.index)
    ax3p = f_m_e2_a.T.plot(ax=ax3,style=['k--o','b--o','g--o'],yerr=f_s_m_e2_a.T,legend=True)
    ax3p.legend(loc='upper center', bbox_to_anchor=(0.5,-0.11),
                prop={'size':leg_fs}, shadow=True, ncol=3, fancybox=True)    
    ax3p.axhline(y=0,ls='--',c='k',linestyle='--')
    ax3p.axvline(x=min_sep,c='c',linestyle='--')
    
    ax4 = fig.add_subplot(gs[11:19,1])
    title = 'e2 on Object b'
    plt.title('Bias vs Separation For ' + title,fontsize=fs)
    plt.ylim([(min_mean - 2*min_s_mean)*min_offset,(max_mean + max_s_mean)*max_offset])
    plt.xlabel('Separation (arcsec)',fontsize=fs)
    plt.ylabel('Residual',fontsize=fs)
    f_m_e2_b = format_df(means_e2_b,x_min,x_max,means_e2_b.index)
    f_s_m_e2_b = format_df(s_means_e2_b,x_min,x_max,s_means_e2_b.index)
    ax4p = f_m_e2_b.T.plot(ax=ax4,style=['k--o','b--o','g--o'],yerr=f_s_m_e2_b.T,legend=True)
    ax4p.legend(loc='upper center', bbox_to_anchor=(0.5,-0.11),
                prop={'size':leg_fs}, shadow=True, ncol=3, fancybox=True)
    ax4p.axhline(y=0,ls='--',c='k',linestyle='--')
    ax4p.axvline(x=min_sep,c='c',linestyle='--')
    
    print "\nBias vs separation plot finished. Program terminated.\n\n"
    plt.savefig(path + '/bias_vs_separation.png')
    plt.clf()
    plt.close()
    
def plot_3d(im,fig,gs,i,sep,fs,cushion):
    x,y = im.array.shape
    domain = np.linspace(1,x,num=x)
    X,Y = np.meshgrid(domain,domain)
    ax = fig.add_subplot(gs[i:i+cushion,0],projection='3d')
    ax.scatter(X,Y,im.array)
    plt.suptitle('Flux Distribution vs Separation',fontsize=fs+5)
    plt.title('\nObject Overlap for Separation of ' + str(sep) + '\"',fontsize=fs)
    
def plot_3d_sep(separation,
                func,
                image_params,
                obj_a,obj_b,method,
                sky_info,
                psf_info,
                fs,factor,space,size):
                
    gs = gridspec.GridSpec((factor+space)*len(separation),1)
    fig = plt.figure(figsize=size)
    
    cushion = np.int((factor*len(separation) - space)/len(separation))
                    
    i = 0
    for sep in separation:
        
                                   
        obj_a[4] = -sep/2
        obj_b[4] = sep/2
        
        # Image properties
        pixel_scale, x_len, y_len = image_params
        
        # Parameters for object a
        flux_a, hlr_a, e1_a, e2_a, x0_a, y0_a, n_a = obj_a 
        
        # Parameters for object b
        flux_b, hlr_b, e1_b, e2_b, x0_b, y0_b, n_b = obj_b
        
        # Use LSST defined sky noise for r-band
        add_noise_flag, texp, sbar, sky_level = sky_info
        
        # psf properties
        psf_flag, beta, fwhm_psf = psf_info
        
        # Create the objects and combine them
        image_a = create_galaxy(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,galtype_gal=func,sersic_index=n_a,
                                x_len=x_len,y_len=y_len,scale=pixel_scale,
                                psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                                method=method)
    
        image_b = create_galaxy(flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,galtype_gal=func,sersic_index=n_b,
                                x_len=x_len,y_len=y_len,scale=pixel_scale,
                                psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                                method=method)
    
        tot_image = image_a + image_b
        
        plot_3d(tot_image,fig,gs,i,sep,fs,cushion)
        
        i += cushion + space
        
    return fig
    
def save_image(path,results_deblend,dbl_im,image_params,truth,sep):
    image_params = pd.Series(image_params,index=['pixel_scale','x_len','y_len'])
 
    # Access a random set of data for the best fit and compare, access a random index and plot the deblended with fits
    index = int(len(dbl_im)*np.random.random(1))
    dbl_obj_im_a = dbl_im[index][0][0]
    dbl_obj_im_b = dbl_im[index][0][1]
    index_val = dbl_im[index][1]
    fit_to_dbl_obj = results_deblend.ix[index_val]
    
    # Create the true images of the unblended profiles
    true_im_a = create_galaxy(truth['flux_a'],truth['hlr_a'],truth['e1_a'],truth['e2_a'],truth['x0_a'],truth['y0_a'],
                              x_len=image_params['x_len'],y_len=image_params['y_len'],scale=image_params['pixel_scale'])
    true_im_b = create_galaxy(truth['flux_b'],truth['hlr_b'],truth['e1_b'],truth['e2_b'],truth['x0_b'],truth['y0_b'],
                              x_len=image_params['x_len'],y_len=image_params['y_len'],scale=image_params['pixel_scale'])
    
    # Create the images of the fits to the deblended objects.x_len=image_params['x_len'],y_len=image_params['y_len'],scale=image_params['pixel_scale'])
    fit_dbl_a = create_galaxy(fit_to_dbl_obj['flux_a'],fit_to_dbl_obj['hlr_a'],fit_to_dbl_obj['e1_a'],fit_to_dbl_obj['e2_a'],fit_to_dbl_obj['x0_a'],fit_to_dbl_obj['y0_a'],
                              x_len=image_params['x_len'],y_len=image_params['y_len'],scale=image_params['pixel_scale'])
    fit_dbl_b = create_galaxy(fit_to_dbl_obj['flux_b'],fit_to_dbl_obj['hlr_b'],fit_to_dbl_obj['e1_b'],fit_to_dbl_obj['e2_b'],fit_to_dbl_obj['x0_b'],fit_to_dbl_obj['y0_b'],
                              x_len=image_params['x_len'],y_len=image_params['y_len'],scale=image_params['pixel_scale'])

    # Plot the fits to the deblended profiles vs the deblended proflies vs the true profiles
    gs = gridspec.GridSpec(7,9)                                   
    fig = plt.figure(figsize=(15,25))
    sh = 0.8
    plt.suptitle('  True Objects, Deblended Objects & Fits to Deblended Objects\n For Separation: '+ str(sep) + '\"',fontsize=45)
    
    # Plot the true blend
    ax = fig.add_subplot(gs[0,1:7])
    z = ax.imshow(true_im_a.array + true_im_b.array,interpolation='none',origin='lower'); plt.title('True Blend'); plt.colorbar(z,shrink=sh)
    
    # Plot the true objects
    ax1 = fig.add_subplot(gs[1,0:4])
    a = ax1.imshow(true_im_a.array,interpolation='none',origin='lower'); plt.title('True Object A'); plt.colorbar(a,shrink=sh)
    ax2 = fig.add_subplot(gs[1,6:10])
    b = ax2.imshow(true_im_b.array,interpolation='none',origin='lower'); plt.title('True Object B'); plt.colorbar(b,shrink=sh)
    
    # Plot the deblended objects
    ax3 = fig.add_subplot(gs[2,0:4])
    c = ax3.imshow(dbl_obj_im_a,interpolation='none',origin='lower'); plt.title('Deblended Object A'); plt.colorbar(c,shrink=sh)
    ax4 = fig.add_subplot(gs[2,6:10])
    d = ax4.imshow(dbl_obj_im_b,interpolation='none',origin='lower'); plt.title('Deblended Object B'); plt.colorbar(d,shrink=sh)
    
    # Plot the residual of the deblended and the true individual objects
    ax5 = fig.add_subplot(gs[3,0:4])
    e = ax5.imshow(dbl_obj_im_a-true_im_a.array,interpolation='none',origin='lower'); plt.title('Residual of Deblended A and True A'); plt.colorbar(e,shrink=sh)
    ax6 = fig.add_subplot(gs[3,6:10])
    f = ax6.imshow(dbl_obj_im_b-true_im_b.array,interpolation='none',origin='lower'); plt.title('Residual of Deblended B and True B'); plt.colorbar(f,shrink=sh)
    
    # Plot the fits to the deblended objects
    ax7 = fig.add_subplot(gs[4,0:4])
    g = ax7.imshow(fit_dbl_a.array,interpolation='none',origin='lower'); plt.title('Fit To Deblended Object A'); plt.colorbar(g,shrink=sh)
    ax8 = fig.add_subplot(gs[4,6:10])
    h = ax8.imshow(fit_dbl_b.array,interpolation='none',origin='lower'); plt.title('Fit To Deblended Object B'); plt.colorbar(h,shrink=sh)
    
    # Plot the residual of the fit to the deblended to the deblended objects
    ax9 = fig.add_subplot(gs[5,0:4])
    i = ax9.imshow(fit_dbl_a.array-dbl_obj_im_a,interpolation='none',origin='lower'); plt.title('Residual Of Fit To Deblended Object A and Deblended A'); plt.colorbar(i,shrink=sh)
    ax10 = fig.add_subplot(gs[5,6:10])
    j = ax10.imshow(fit_dbl_b.array-dbl_obj_im_b,interpolation='none',origin='lower'); plt.title('Residual Of Fit To Deblended Object B and Deblended B'); plt.colorbar(j,shrink=sh)
    
    # Plot the residual of the fit to the deblended to the true unblended object
    ax11 = fig.add_subplot(gs[6,0:4])
    k = ax11.imshow(fit_dbl_a.array-true_im_a.array,interpolation='none',origin='lower'); plt.title('Residual Of Fit To Deblended Object A and True Object A'); plt.colorbar(k,shrink=sh)
    ax12 = fig.add_subplot(gs[6,6:10])
    l = ax12.imshow(fit_dbl_b.array-true_im_b.array,interpolation='none',origin='lower'); plt.title('Residual Of Fit To Deblended Object B and True Object B'); plt.colorbar(l,shrink=sh)

    plt.savefig(path + '/image_of_one_trial.png')
    plt.clf()
    plt.close()
