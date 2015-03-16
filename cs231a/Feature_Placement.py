# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:24:11 2015

@author: luis
"""

from __future__ import division
import galsim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import astropy
import astroML as aml
from astroML.fourier import FT_continuous, sinegauss, sinegauss_FT
import Library
import mahotas as mh
from mahotas import features
import SimpleCV
import sklearn
from sklearn.decomposition import RandomizedPCA
import skimage
from skimage.feature import hog
import compute_stats as cs
from mpl_toolkits.mplot3d import Axes3D
import mpld3
mpld3.enable_notebook()
import ipdb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Filter out noisy peaks
def filter_mini_peaks(im,limit):
    labeled, n_nuclei = mh.label(im)
    #pdb.set_trace()
    tot_region = np.zeros(im.shape)
    for i in xrange(1,n_nuclei+2):
        labeled_i = np.copy(labeled)
        labeled_i[labeled_i != i] = 0
        if (np.sum(labeled_i) < limit):
            labeled_i[labeled_i == i] = 0
        reg_i = labeled_i*np.copy(im)
        tot_region = tot_region + reg_i
    return tot_region
    
# Segmentation for Peak Finding
def find_peaks(im,interval):
    ranges = np.linspace(im.min(),im.max(),interval)
    tot_nuclei = []
    labels_arr = []
    matrix = im
    for val in ranges:
        matrix[matrix < val] = 0
        new_matrix = filter_mini_peaks(np.copy(matrix),2)
        labels_i, n_nuclei_i = mh.label(new_matrix)
        tot_nuclei.append(n_nuclei_i)
        labels_arr.append(labels_i)
    return pd.Series(tot_nuclei), pd.Panel(labels_arr)
    
def plot_samples(filtered_images):
    # Plot Images 
    fig = plt.figure(figsize=(15,11))
    gs = gridspec.GridSpec(4,4)
    count = 0
    fs = 10
    for i in xrange(0,4):
        for j in xrange(0,4):
            if (count > len(filtered_images)-1):
                continue
            else:
                ax = fig.add_subplot(gs[i,j])
                a = ax.imshow(filtered_images[count],interpolation='bilinear',origin='lower',cmap=plt.get_cmap(name='hot'))
                plt.xlabel('Pixels',fontsize=fs)
                cbar = plt.colorbar(a,shrink=0.7)
                cbar.ax.tick_params(labelsize=10)
                if j == 0: plt.ylabel('Pixels',fontsize=fs)
            count = count + 1
    plt.suptitle('Gaussian Filtered Images',fontsize=fs+15)
    plt.show()    
    
    
if __name__ == '__main__':

    # Parameters    
    texp = 6900 # seconds
    sbar = 26.8 # sky photons per second per pixel
    sky_level = texp*sbar  
    sky_noise = np.sqrt(sky_level)

    # Which data set to use
    train = True
    test = False
    
    assert train != True and test != True, "Can't read in both data sets."

    # Save the data    
    save = True
    
    # Read in the images
    if train == True:
        filtered_im = pd.read_pickle('training_data/noisy_images')
        labels = pd.read_pickle('training_data/labels')
        snrs = pd.read_pickle('training_data/snr')

    if test == True:
        filtered_im = pd.read_pickle('test_data/noisy_images')
        labels = pd.read_pickle('test_data/labels')
        snrs = pd.read_pickle('test_data/snr')

    # Number of images to consider (up to 4000 for each)
    num_images = 10
    sample = filtered_im.iloc[0:num_images]
    
    # Array to store informatino
    feature_matrix = []
    
    # Count if HOG returns null values
    count_bad = 0
    
    # Iterate through each image, calculating features and storing them
    for key, df in sample.iteritems():
        
        # Access the label
        label = labels[key]
        # Acess the SNR
        snr = snrs[key]
        if snr < 40: continue # SNR threshold
        
        # Obtain the image
        base_image = df.values
        
        # Floor negative pixels. 
        base_image[base_image < 0] = 0
        
        # Floor values under the sky noise
        base_image[base_image < sky_noise] = 0
        
        # Filter out noisy peaks
        threshold = 500
        image = filter_mini_peaks(np.copy(base_image),threshold)
        if np.all(image==0): continue 
        
        # Asym
        x_len = image.shape[0]
        y_len = image.shape[1]
        a_rms, a_abs, a_abs_mat, a_est, peak = cs.compute_asym(image,[0,0],x_len,y_len)
        
        # M20
        m20, m20mat, peak = cs.find_min_m20(image,x_len,y_len)
        
        # Gini coefficient
        gini = cs.compute_gini(image)
        
        # Find peaks
        peaks, regions = find_peaks(np.copy(image),1000)
        max_peaks = np.max(peaks)
        
        # Hog
        fd, hog_image = hog(image, orientations=6, pixels_per_cell=(10, 10),
                        cells_per_block=(1,1), visualise=True, normalise=True)
        if np.all(fd==0): count_bad += 1; print key
        
        # Local Binary Patterns
        radius = 1
        lbp_feats = features.lbp(image,radius,8*radius)
        
        # Eigenfaces
        n_components = x_len/6
        pca = RandomizedPCA(n_components=n_components, whiten=True).fit(image)
        train_pca = pca.transform(image)
        pca_feats = train_pca.ravel()
        
        # Create feature vector 
        feature_matrix.append([label,snr,a_est,m20,gini,max_peaks]+fd.tolist()+lbp_feats.tolist()+pca_feats.tolist())
    
    
    # Create the column information for the DataFrame    
    fd_len = len(fd)
    lbp_len = len(lbp_feats)
    pca_len = len(pca_feats)    
    scalars = ['label','snr','asym','m20','gini','peaks']
    fd_str = ['fd' for i in xrange(0,fd_len)]
    lbp_str = ['lbp' for i in xrange(0,lbp_len)]
    pca_str = ['pca' for i in xrange(0,pca_len)]
    tot_str = scalars + fd_str + lbp_str + pca_str
    
    # Create DataFrame object    
    feature_matrix_df = pd.DataFrame(feature_matrix,columns=tot_str)
    
    # Save the data if necessary
    if save == True:
        if train == True:
            feature_matrix_df.to_pickle('feature_data/training_features')
        elif test == True:
            feature_matrix_df.to_pickle('feature_data/testing_features')
            
