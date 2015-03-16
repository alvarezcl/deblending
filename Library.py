## Set of library functions

import galsim
import numpy as np
import lmfit
import deblend
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import scipy
from scipy import interpolate

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
                     func, seed_1, seed_2, seed_3,
                     pixel_scale, x_len, y_len,
                     add_noise_flag, texp, sbar,
                     psf_flag, beta, fwhm_psf,
                     method,
                     factor_init,
                     plot=False):

    # Create the objects and combine them
    image_a = create_galaxy(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,galtype_gal=func,sersic_index=n_a,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                            method=method, seed=seed_1)

    image_b = create_galaxy(flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,galtype_gal=func,sersic_index=n_b,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                            method=method, seed=seed_2)

    sky_level = texp*sbar
    sersic_func = func

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

    # Now estimate the shape for each true object
    result_a_true = lmfit.minimize(residual_1_obj, parameters_a, args=(image_c.array, sky_level, x_len, y_len, pixel_scale, sersic_func, n_a))

    result_b_true = lmfit.minimize(residual_1_obj, parameters_b, args=(image_d.array, sky_level, x_len, y_len, pixel_scale, sersic_func, n_a))

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
    
# Input a dataframe object to return the mean, std dev, and error on the mean
def show_stats(results,runs):
    data = pd.DataFrame(np.array([np.mean(results),np.std(results),np.std(results)/np.sqrt(runs)]),columns=results.columns)
    data.index = [r'$\bar\mu$',r'$\sigma$', r'$\sigma_{\mu}$']
    return data

# Save the information if results have been statistically independent
def save_data(results_deblend,results_true,results_sim,truth,seed_int):
    # Save files according to separation of objects
    sep = np.sqrt((truth['x0_b'] - truth['x0_a'])**2 + (truth['y0_b'] - truth['y0_a'])**2)

    if seed_int == 0:
        dble = 'results_deblend_sep_' + str(sep) + '.csv'
        with open(dble,'a') as f:
            if os.stat(dble)[6] == 0:
                # If the file is empty, put the header.
                results_deblend.to_csv(f)
            else:
                # If the file is not empty, do not put the header in.
                results_deblend.to_csv(f,header=False)
        tru = 'results_true_sep_' + str(sep) + '.csv'
        with open(tru,'a') as f:
            if os.stat(tru)[6] == 0:
                # If the file is empty, put the header.
                results_true.to_csv(f)
            else:
                # If the file is not empty, do not put the header in.
                results_true.to_csv(f,header=False)
        sim = 'results_sim_sep_' + str(sep) + '.csv'
        with open(sim,'a') as f:
            if os.stat(sim)[6] == 0:
                # If the file is empty, put the header.
                results_sim.to_csv(f)
            else:
                # If the file is not empty, do not put the header in.
                results_sim.to_csv(f,header=False)

    # Otherwise save each set to it's own file with seed integer represented.
    # Never append the data.
    else:
        dble = 'results_deblend_sep_' + str(sep) + '_seed_' + str(seed_int) + '.csv'
        with open(dble,'w') as f:
            # If the file is empty, put the header.
            results_deblend.to_csv(f)
        tru = 'results_true_sep_' + str(sep) + '_seed_' + str(seed_int**2) + '.csv'
        with open(tru,'w') as f:
            # If the file is empty, put the header.
            results_true.to_csv(f)
        sim = 'results_sim_sep_' + str(sep) + '_seed_' + str(seed_int**3) + '.csv'
        with open(sim,'w') as f:
            # If the file is empty, put the header.
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