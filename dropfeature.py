# -*- coding=utf-8 -*-
#!/usr/bin/python2.7

##################################################
#update_dt: 2018-08-06
#author：梁云
##################################################

import numpy as np 
import pandas as pd
from sklearn import feature_selection
import ks

def drop_feature(X_train,y_train,X_test = '',coverage_threshold = 0.1, 
                 ks_threshold = 0.05):
    
    if not coverage_threshold and not ks_threshold:
        return(X_train, X_test)
    
    sample_num = len(X_train)
    for col in X_train.columns:
        
        nan_index = pd.isnull(X_train[col])
        x_col = X_train[col][~nan_index]
        y_col = y_train[~nan_index]
        
        coverage_ratio = len(x_col)/float(sample_num)
        class_num = len(set(x_col))
        ks_value = max(ks.ks_analysis(x_col.values,y_col.values)['ks_value']) if class_num > 2 else 1
        if any([class_num <2, coverage_ratio < coverage_threshold,
                ks_value < ks_threshold]) :
            X_train = X_train.drop([col],axis = 1)
            if len(X_test): X_test = X_test.drop([col],axis = 1) 
                
    return(X_train,X_test)
           