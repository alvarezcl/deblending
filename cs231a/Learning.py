# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:27:20 2015

@author: luis
"""
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import cross_validation
import pandas as pd
import compute_stats as cs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as Gridspec
import numpy as np

# Read in feature data
train_data = pd.read_pickle('feature_data/training_features')
# Obtain the labels
train_labels = train_data['label']
# Obtain the SNR
train_snr = train_data['snr']
# Reformate training data on just observables
train_data = train_data.drop(['label'],1)
train_data = train_data.drop(['snr'],1)

# Models
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data, 
 #                                                                    train_labels, 
  #                                                                   test_size=0.4, 
   #                                                                  random_state=0)


clf = svm.SVC(kernel='linear', C=1)
scores_lin_svm = cross_validation.cross_val_score(clf, train_data, train_labels, cv=10)
lin_svm_score = np.mean(scores_lin_svm)

ridge = Ridge(alpha=1.0)
log_reg = LogisticRegression(C=1)
svm_proj = svm.SVC(C=1)
ran_forest = RandomForestClassifier(n_estimators=45)

# Estimate


