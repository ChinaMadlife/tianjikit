# -*- coding=utf-8 -*-
#!/usr/bin/python2.7

##################################################
#update_dt: 2018-08-06
#author：梁云
##################################################
'''
有以下标准化策略
'MinMax':缩放到[0,1]之间
'Standard':缩放至0均值，1标准差
'MaxAbs':缩放至[-1,1]之间，无偏移
'Robust':鲁棒性缩放
'''
import numpy as np 
import pandas as pd
from sklearn import preprocessing


def scale_feature(X_train,X_test = '',method = 'MinMax'):
    
    
    method_dict = {'MinMax':preprocessing.MinMaxScaler,
                   'Standard':preprocessing.StandardScaler,
                   'MaxAbs':preprocessing.MaxAbsScaler,
                   'Robust':preprocessing.RobustScaler}
    
    scaler = method_dict[method]()
    scaler.fit(X_train)
    
    X_train_array = scaler.transform(X_train)
    if len(X_test):X_test_array = scaler.transform(X_test)
    
    X_train_new = pd.DataFrame(X_train_array,columns = X_train.columns)
    if len(X_test):X_test_new = pd.DataFrame(X_test_array,columns = X_test.columns)
        
    if not len(X_test):X_test_new = ''
        
    return(X_train_new,X_test_new)
