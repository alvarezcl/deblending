## Script to create training and test set of images
import galsim
import pandas as pd
import numpy as np
import random
import Library

def catalog(num):

    # Image Parameters
    x_len = y_len = 50 # pixel
    pixel_scale = 0.2  # "/pixel
    method = 'phot'
    seed_a = galsim.BaseDeviate(1); seed_b = galsim.BaseDeviate(2)
    seed_c = galsim.BaseDeviate(3); seed_d = galsim.BaseDeviate(4)        
        
    # Object Parameters
    low_flux = 100000; high_flux = 10*low_flux    # Counts
    low_hlr = 5*pixel_scale; high_hlr = 3*low_hlr # Arcsec
    low_e1 = -0.5; high_e1 = -low_e1
    low_e2 = -0.5; high_e2 = -low_e2
    low_x0 = -x_len/8; high_x0 = x_len/8          # Arcsec
    low_y0 = -y_len/8; high_y0 = y_len/8          # Arcsec
        
    # Store each image in a dict to convert to a pandas panel and
    # store each image label in a dict as well.
    images = {}
    labels = {}
    for i in xrange(0,num):
        # In considering the parameters for each image, we allow a range of 4
        # objects in one image.
        flux_a = random.randint(low_flux,high_flux)
        hlr_a = random.uniform(low_hlr,high_hlr)
        e1_a = random.uniform(low_e1,high_e1)
        e2_a = random.uniform(low_e2,high_e2)
        x0_a = random.uniform(low_x0,high_x0)
        y0_a = random.uniform(low_y0,high_y0)
        # Image
        image_a = Library.create_galaxy(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,
                                        x_len=x_len,y_len=y_len,method=method,
                                        seed=seed_a)
        
        # Now for any other objects, we make sure each object has centroid
        # coordinates within the hlr of the primary object.
        flux_b = random.randint(low_flux,high_flux)
        hlr_b = random.uniform(low_hlr,high_hlr)
        e1_b = random.uniform(low_e1,high_e1)
        e2_b = random.uniform(low_e2,high_e2)
        x0_b = x0_a + hlr_a*np.cos(np.arctan2(y0_a,x0_a))*random.uniform(-1,1)
        y0_b = y0_a + hlr_a*np.sin(np.arctan2(y0_a,x0_a))*random.uniform(-1,1)
        # Image
        image_b = Library.create_galaxy(flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,
                                        x_len=x_len,y_len=y_len,method=method,
                                        seed=seed_b)
        
        flux_c = random.randint(low_flux,high_flux)
        hlr_c = random.uniform(low_hlr,high_hlr)
        e1_c = random.uniform(low_e1,high_e1)
        e2_c = random.uniform(low_e2,high_e2)
        x0_c = x0_a + hlr_a*np.cos(np.arctan2(y0_a,x0_a))*random.uniform(-1,1)
        y0_c = y0_a + hlr_a*np.sin(np.arctan2(y0_a,x0_a))*random.uniform(-1,1)
        # Image        
        image_c = Library.create_galaxy(flux_c,hlr_c,e1_c,e2_c,x0_c,y0_c,
                                        x_len=x_len,y_len=y_len,method=method,
                                        seed=seed_c)
        
        flux_d = random.randint(low_flux,high_flux)
        hlr_d = random.uniform(low_hlr,high_hlr)
        e1_d = random.uniform(low_e1,high_e1)
        e2_d = random.uniform(low_e2,high_e2)
        x0_d = x0_a + hlr_a*np.cos(np.arctan2(y0_a,x0_a))*random.uniform(-1,1)
        y0_d = y0_a + hlr_a*np.sin(np.arctan2(y0_a,x0_a))*random.uniform(-1,1)
        # Image        
        image_d = Library.create_galaxy(flux_d,hlr_d,e1_d,e2_d,x0_d,y0_d,
                                        x_len=x_len,y_len=y_len,method=method,
                                        seed=seed_d)

        # Sample a random number between zero and one.
        sample_num = random.uniform(0,1)
        # This will be a single object
        if sample_num < 0.5:
            image = image_a
            images[str(i)] = image
            labels[str(i)] = 0
        # This will be two overlapping objects
        elif sample_num >= 0.5 and sample_num < 0.75:
            image = image_a + image_b
            images[str(i)] = image
            labels[str(i)] = 1
        # This will be three overlapping objects    
        elif sample_num >= 0.75 and sample_num < 0.9:
            image = image_a + image_b + image_c
            images[str(i)] = image
            labels[str(i)] = 1
        # This will be four overlapping objects    
        elif sample_num >= 0.9 and sample_num <= 1.0:
            image = image_a + image_b + image_c + image_d
            images[str(i)] = image
            labels[str(i)] = 1
            
    return images, labels