## Script to obtain the gini coefficients and M20 statistic

from __future__ import division
import Library
import galsim
import numpy as np
import deblend
import pandas as pd
import sys
import pdb
import math

def main(add_blend=False):
    # First create the image, using a simple gaussian

    flux = 25000
    hlr = 0.5
    e1 = 0
    e2 = 0
    x0 = 0
    y0 = 0
    n = 0.5
    # Set the RNG
    seed_1 = galsim.BaseDeviate(1)
    
    # Image properties
    pixel_scale = 1/5     # arcsec / pixel
    x_len = y_len = 25            # pixel
   
    peak = [x0/pixel_scale,y0/pixel_scale]
    
    # Use LSST defined sky noise for r-band
    add_noise_flag = False
    texp = 6900 # seconds;
    sbar = 26.8 # sky photons per second per pixel
    sky_level = texp*sbar # For now
    sky_noise = np.sqrt(sky_level)
    
    # psf properties
    psf_flag = False
    beta = 3
    fwhm_psf = 0.6
    
    # method for drawing
    method = 'fft'
    
    image = Library.create_galaxy(flux,hlr,e1,e2,x0,y0,sersic_index=n,
                                  x_len=x_len,y_len=y_len,scale=pixel_scale,
                                  psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                                  method=method, seed=seed_1)
                                  
    tot_image = image
        
    # We can add another object as well, mirror the properties of the first object 
    if add_blend == True:    
        x0_2 = -x0
        y0_2 = -y0
        image_2 = Library.create_galaxy(flux,hlr,e1,e2,x0_2,y0_2,sersic_index=n,
                                        x_len=x_len,y_len=y_len,scale=pixel_scale,
                                        psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                                        method=method, seed=seed_1)
        image = image + 5*image_2
        
    if add_noise_flag == True:
        image_noise = Library.add_noise(tot_image,seed=seed_1,sky_level=sky_level)
        
    # Compute the gini coefficient
    gini_c = compute_gini(image.array,1)
    a_rms, a_abs = compute_asym(image.array,peak,1)
    m20, min_peak = find_min_m20(image.array,1)
    return gini_c, a_rms, a_abs, m20, min_peak
    
# Take in an array of pixel values and compute the gini coefficient.    
def compute_gini(matrix,threshold):
    matrix = extract_sub(matrix,threshold)
    # Should I be using this mean? Obtaining the mean of effective ps
    xbar = matrix.mean()
    # Compute the effective n
    n = matrix.shape[0]*matrix.shape[1] 
    c = 1/(np.abs(xbar)*(n**2-n))
    # Sort the matrix array in increasing order
    arr = np.sort(matrix.ravel(),kind='heapsort')
    m = 2*np.linspace(1,n,n) - n - 1    
    return c*sum(m*np.abs(arr))

# Compute the asymmetry coefficient.
def compute_asym(matrix,peak,x_len,y_len):
    rotated_mat = deblend.rotate(matrix,peak)
    a_rms = np.sqrt(sum(sum((matrix-rotated_mat)**2))/(2*sum(sum(matrix**2))))
    a_abs = sum(sum(np.abs(matrix-rotated_mat)))/(2*sum(sum(np.abs(matrix))))
    
    # Create a matrix for corresponding a_abs values at each pixel.
    a_abs_mat = np.zeros(matrix.shape)
    # Create the vector of indices to march through.
    i = np.arange(x_len)
    j = np.arange(y_len)
    for val_i in i:
        for val_j in j:
            rotated_mat = deblend.rotate(matrix,[val_i - int(x_len/2), val_j - int(y_len/2)])
            a_ab = sum(sum(np.abs(matrix-rotated_mat)))/(2*sum(sum(np.abs(matrix))))
            a_abs_mat[val_i,val_j] = a_ab
    # The x,y coordinates live in matrix space so convert to image/pixel space
    # where the center is zero-indexed.
    x,y = np.where(a_abs_mat==np.min(a_abs_mat))
    x = x - int(x_len/2); y = y - int(y_len/2)
    return a_rms, a_abs, a_abs_mat, np.min(a_abs_mat), (x[0],y[0])        

# Compute the asymmetry coefficient.
def compute_asym_alt(matrix,peak,x_len,y_len):
    # matrix = extract_sub(matrix,threshold)
    # Make sure one is rotating about the max in pixels not arcsec with the
    # center of the image zero-indexed.
    rotated_mat = deblend.rotate(matrix,peak)
    a_rms = np.sqrt(sum(sum((matrix-rotated_mat)**2))/(2*sum(sum(matrix**2))))
    a_abs = sum(sum(np.abs(matrix-rotated_mat)))/(2*sum(sum(np.abs(matrix))))

    a_abs_mat = np.zeros(matrix.shape)
    # For each i,j in the matrix, compute a_abs and store
    i = np.linspace(int(-x_len/2 ),int(x_len/2 - 1),x_len)
    j = np.linspace(int(-y_len/2 ),int(y_len/2 - 1),y_len)    
    for i_val in i:
        for j_val in j:
            rotated_mat = deblend.rotate(matrix,[i_val,j_val])
            a_ab = sum(sum(np.abs(matrix-rotated_mat)))/(2*sum(sum(np.abs(matrix))))
            a_abs_mat[int(i_val + x_len/2),int(j_val + y_len/2)] = a_ab
    # The x,y coordinates live in matrix space so convert to image/pixel space.
    x,y = np.where(a_abs_mat==np.min(a_abs_mat))
    x = x - int(x_len/2); y = y - int(y_len/2)
    return a_rms, a_abs, a_abs_mat, np.min(a_abs_mat), (x[0],y[0])

# Compute the m20 statistic.
def compute_mtwenty(matrix, peak):
    m_tot = 0
    
    f_tot = sum(sum(matrix))
    info = {}
    # Obtain the total second moment
    for (i,j), value in np.ndenumerate(matrix):
        m_tot = m_tot + value*((i - peak[0])**2 + (j - peak[1])**2)
        # Store each coordinate and pixel flux value as a dict pairing
        info[(i,j)] = value
    # Obtain the brightest 20 percent    
    # Sort the matrix indices by decreasing
    sorted_arr = sorted(info,key=info.get)
    sorted_arr = sorted_arr[::-1]
    # Run through each index with flux value and calculate the moment
    # up to the flux being 20 percent of the total.
    flux = 0
    m = 0
    
    for k in xrange(0,len(sorted_arr)):
        flux = flux + matrix[sorted_arr[k]]
        if flux > 0.2*f_tot:
            break
        else:
            m = m + matrix[sorted_arr[k]]*((sorted_arr[k][0]-peak[0])**2 + (sorted_arr[k][1] - peak[1])**2)
                
    return np.log10(m/m_tot)

# Run through every pixel and compute m_20 to find the min value.
def find_min_m20(matrix,x_len,y_len):
    #matrix = extract_sub(matrix,threshold)
    m_20_mat = np.zeros(matrix.shape)
    for (i,j), value in np.ndenumerate(matrix):
        if math.isnan(compute_mtwenty(matrix,[i,j])):
            pdb.set_trace()
        m_20_mat[i,j] = compute_mtwenty(matrix,[i,j])
    x,y = np.where(m_20_mat==np.min(m_20_mat))        
    return m_20_mat, np.min(m_20_mat), (int(x[0]-x_len/2),int(y[0]-y_len/2))
        
def extract_sub(matrix,threshold):
    x,y = np.where(matrix>threshold)
    return matrix[x.min():x.max()+1, y.min():y.max()+1]        
        
if __name__ == '__main__':
    main(False)
    
