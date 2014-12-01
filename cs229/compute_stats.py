## Script to obtain the gini coefficients and M20 statistic

from __future__ import division
import Library
import galsim
import numpy as np
import deblend
import mahotas as mh
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
def compute_gini(matrix):
    #matrix = extract_sub(matrix,threshold)
    # Should I be using this mean? Obtaining the mean of effective ps
    xbar = matrix.mean()
    # Compute the effective number of pixels
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
                
    return m/m_tot

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
    
# Delete the trailing zeros in a given matrix
def delrc(arr):
    while True:         # delete leading rows with all zeros
        if np.all(arr[0]==0):
            arr=np.delete(arr,0,axis=0)
        else: break
    while True:         # delete trailing rows with all zeros
        if np.all(arr[-1]==0):
            arr=np.delete(arr,-1,axis=0)
        else: break     
    while True:         # delete leading cols with all zeros
        if np.all(arr[:,0]==0):
            arr=np.delete(arr,0,axis=1)
        else: break     
    while True:         # delete trailing cols with all zeros
        if np.all(arr[:,-1]==0):
            arr=np.delete(arr,-1,axis=1)
        else: break
    return arr

# Extract sub-regions of an image corresponding to peaks
def extract_sub_regions(matrix,percent,sigma_smooth):
    # Make a copy of the matrix
    seg_im = np.copy(matrix)
    # Obtain the flux threshold value
    flux_thresh = percent*matrix.max()
    # Smooth the image
    seg_im = mh.gaussian_filter(np.copy(seg_im),sigma_smooth)
    # Convert to binary image
    seg_im = seg_im>=flux_thresh
    # Use mahotas to find centroids and labels
    labeled, n_centroids = mh.label(seg_im)
    if n_centroids == 0:
        raise ValueError('Your image is empty.')
    # labeled is a 2d array of integers corresponding to regions
    regions = []
    # Create an array of the region values
    region_nums = np.arange(labeled.min()+1,labeled.max()+1)
    # Loop through each region value, create the template, and multiply
    # by the original image to obtain each region, then store.
    for num in region_nums:
        region = np.copy(labeled)
        region[region!=num] = 0
        region[region==num] = 1
        region = matrix*region
        regions.append(region)
    return regions, n_centroids
    
# Compute the multimode statistic    
def compute_mm(matrix,num_percents,min_pix,sigma_smooth):    
    # Run through a representative flux percent
    R_val = []
    regions = []
    n_centroids = []
    percents = np.linspace(0.1,0.95,num_percents)
    for p in percents:
        regions_i, centroid_i = extract_sub_regions(matrix,p,sigma_smooth)
        
        if centroid_i == 0:
            pdb.set_trace()
        
        if centroid_i == 1:
            R_val.append(1.0)
            n_centroids.append(centroid_i)
            regions.append(regions_i)

        if centroid_i == 2:
            size_0 = len(regions_i[0][np.nonzero(regions_i[0])])
            size_1 = len(regions_i[1][np.nonzero(regions_i[1])])
            numerator = max(size_0,size_1)
            denominator = min(size_0,size_1)
            
            if denominator <= min_pix: # Hit a low probability spike 
                R_val.append(1.0)
                n_centroids.append(centroid_i-1)
                if size_1 > size_0:
                    regions.append([regions_i[1]])
                else:
                    regions.append([regions_i[0]])
            else:
                R_val.append(1.0/(1 + numerator/denominator))
                n_centroids.append(centroid_i)
                regions.append(regions_i)
                
        if centroid_i == 3:
            size_0 = len(regions_i[0][np.nonzero(regions_i[0])])
            size_1 = len(regions_i[1][np.nonzero(regions_i[1])])            
            size_2 = len(regions_i[2][np.nonzero(regions_i[2])])
            
            if size_0 <= min_pix: # Hit a low probability spike
                centroid_i = centroid_i - 1
                numerator = max(size_1,size_2)
                denominator = min(size_1,size_2)
                if denominator <= min_pix: # Hit a low probability spike 
                    R_val.append(1.0)
                    n_centroids.append(centroid_i-1)
                    if size_1 > size_2:
                        regions.append([regions_i[1]])
                        continue
                    else:
                        regions.append([regions_i[2]])
                        continue
                else:
                    R_val.append(1.0/(1.0+numerator/denominator))
                    n_centroids.append(centroid_i)
                    regions.append([regions_i[1],regions_i[2]])
                    continue
                    
            if size_1 <= min_pix: # Hit a low probability spike
                centroid_i = centroid_i - 1
                numerator = max(size_0,size_2)
                denominator = min(size_0,size_2)
                if denominator <= min_pix: # Hit a low probability spike 
                    R_val.append(1.0)
                    n_centroids.append(centroid_i-1)
                    if size_2 > size_0:
                        regions.append([regions_i[2]])
                        continue
                    else:
                        regions.append([regions_i[0]])
                        continue
                else:
                    R_val.append(1.0/(1.0+numerator/denominator))
                    n_centroids.append(centroid_i)
                    regions.append([regions_i[0],regions_i[2]])
                    continue
              
            if size_2 <= min_pix: # Hit a low probability spike
                centroid_i = centroid_i - 1
                numerator = max(size_0,size_1)
                denominator = min(size_0,size_1)
                if denominator <= min_pix: # Hit a low probability spike 
                    R_val.append(1.0)
                    n_centroids.append(centroid_i-1)
                    if size_1 > size_0:
                        regions.append([regions_i[1]])
                        continue
                    else:
                        regions.append([regions_i[0]])
                        continue
                else:
                    R_val.append(1.0/(1.0+numerator/denominator))
                    n_centroids.append(centroid_i)
                    regions.append([regions_i[0],regions_i[1]])
                    continue
                    
            if size_0 > min_pix and size_1 > min_pix and size_2 > min_pix:
                numerator = max(size_0,size_1,size_2)
                denominator = min(size_0,size_1,size_2)
                R_val.append(1.0/(1.0+numerator/denominator))
                n_centroids.append(centroid_i)
                regions.append(regions_i)
                continue

    # The mm stat is the minimum value observed
    mm = np.min(R_val)
    index = np.where(R_val==mm)[0]
    # Assure that index is a scalar
    if len(index) != 1:
        index = index[len(index)-1]
    if index > len(percents)-1:
        pdb.set_trace()
    prcentile = percents[index]
    region = regions[index]
    n_centroid = n_centroids[index]
    return mm, prcentile, region, n_centroid

def compute_intensity_ratio(regions):
    if len(regions) == 1:
        return 0
    elif len(regions) == 2:
        assert max(sum(sum(regions[0])),sum(sum(regions[1]))) != 0
        return min(sum(sum(regions[0])),sum(sum(regions[1])))/max(sum(sum(regions[0])),sum(sum(regions[1])))
    else:
        numerator = min(sum(sum(regions[0])),sum(sum(regions[1])),sum(sum(regions[2])))
        denominator = max(sum(sum(regions[0])),sum(sum(regions[1])),sum(sum(regions[2])))
        assert denominator != 0
        return numerator/denominator
        
if __name__ == '__main__':
    pass
    
