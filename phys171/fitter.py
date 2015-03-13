# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:42:42 2015

@author: luis
"""

## Fit the correlation function data for the exponent
## First smooth the data

from __future__ import division
import lmfit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
    
def model(beta,data):
    x = np.linspace(1,len(data),len(data))
    return 1/(x**(beta))

# Take the difference of the data and the model for one galaxy
def resid(param, data):
    beta = param['beta'].value
    true = model(beta,data)
    true[0] = 1
    return (true-data)
    
def correlation_fit(data):
    parameters = lmfit.Parameters()
    parameters.add('beta', value=(1/4))
    # Extract params that minimize the difference of the data from the true.
    result = lmfit.minimize(resid, parameters, args=[data])
    return model(result.values['beta'],data)
    

if __name__ == '__main__':
    data = pd.read_csv('data/two_point_met1600_2000.csv')
    crit_temp_data = np.abs(data['2.26'])
    refined_data = data[0:33]
    crit_temp_refined = crit_temp_data[0:33].values
    parameters = lmfit.Parameters()
    parameters.add('beta', value=(1/4))
    # Extract params that minimize the difference of the data from the true.
    result = lmfit.minimize(resid, parameters, args=[crit_temp_refined])
    
    domain = np.array(xrange(0,len(crit_temp_refined)))
    fit = model(result.values['beta'],crit_temp_refined)
    plt.plot(domain,crit_temp_refined,'--o',
             domain,fit,'--o',
             domain,model(1/4,crit_temp_refined),'--o')
             
    plt.legend(['Data','Fit','True'])
    
    plt.figure()
    cor_leng = np.array([0.02,0.0015,0.017,fit[len(fit)-1],refined_data['2.33'].values[32],refined_data['2.4'].values[32]])
    plt.plot([2.06,2.13,2.20,2.26,2.33,2.4],-40/np.log(cor_leng),'--o')
    