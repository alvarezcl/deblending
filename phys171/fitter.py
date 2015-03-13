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
    return 1/(np.array(xrange(0,len(data)))**(beta))

# Take the difference of the data and the model for one galaxy
def resid(param, data):
    beta = param['beta'].value
    true = model(beta,data)
    true[0] = 1
    return (true-data)
    

if __name__ == '__main__':
    data = pd.read_csv('data/two_point_met1600_2000.csv')
    crit_temp_data = data['2.26']
    crit_temp_refined = crit_temp_data[0:38].values
    parameters = lmfit.Parameters()
    parameters.add('beta', value=(1/4))
    # Extract params that minimize the difference of the data from the true.
    result = lmfit.minimize(resid, parameters, args=[crit_temp_refined])
    
    domain = np.array(xrange(0,len(crit_temp_refined)))
    plt.plot(domain,crit_temp_refined,'--o',
             domain,model(result.values['beta'],crit_temp_refined),'--o',
             domain,model(1/4,crit_temp_refined),'--o')
             
    plt.legend(['Data','Fit','True'])