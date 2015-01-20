## This script attempts to address the sublteties in the deblender with
## the conservation of flux. 

from __future__ import division
import Library
import galsim
import deblend
import sys
import numpy as np

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
    return tot_image, image_a, image_b, templates, template_fractions, children
    
if __name__ == '__main__':
    main(sys.argv)