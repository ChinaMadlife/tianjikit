# -*- coding=utf-8 -*-
#!/usr/bin/python2.7

##################################################
#update_dt: 2018-08-06
#author：梁云
##################################################
'''
有以下填充策略
'infer':使用逾期率与缺失样本相当的分组的特征的平均值推断
'mean':平均值填充
'median':中位数填充
'most':众数填充
'-1': -1填充
'0': 0填充
所有含有缺失值的列都会增加一列新的特征 '*_isnan'来指示该列是否缺失
'''

import numpy as np 
import pandas as pd
from scipy import stats
from collections import Counter
import ks
import re

def most_value(valuelist):
    cer = Counter(valuelist)
    dfcnt = pd.DataFrame(cer.most_common(),columns = ['value','cnt'])
    most = dfcnt.loc[dfcnt['cnt'] == dfcnt['cnt'].max(),:]['value'].mean() if len(dfcnt) else np.nan
    return most

def fill_nan(X_train,y_train,X_test = '',method = 'infer'):
    
    X_train_new,X_test_new = pd.DataFrame(),pd.DataFrame()
    method_dict = {'mean':np.mean,'median':np.median,'most':most_value}
    
    if method in ['0','-1']:
        for col in X_train.columns:
            nan_index = pd.isnull(X_train[col])
            X_train_new[col] = X_train[col].fillna(value = int(method))
            if len(X_test):X_test_new[col] = X_test[col].fillna(value = int(method))
            if any(nan_index):
                X_train_new[col + '_isnan'] = nan_index.astype('int').values
                if len(X_test):X_test_new[col + '_isnan'] = pd.isnull(X_test[col]).astype('int').values
                
    elif method in ['mean','median','most']:
        for col in X_train.columns:
            nan_index = pd.isnull(X_train[col])
            
            #根据不同的填充策略计算相应的填充值
            func =  method_dict[method]
            fill_value = func(X_train[col].dropna().values)
            X_train_new[col] = X_train[col].fillna(value = fill_value)
            if len(X_test):X_test_new[col] = X_test[col].fillna(value = fill_value)
            if any(nan_index):
                X_train_new[col + '_isnan'] = nan_index.astype('int').values
                if len(X_test):X_test_new[col + '_isnan'] = pd.isnull(X_test[col]).astype('int').values
                
    elif method == 'infer':
        for col in X_train.columns:
            nan_index = pd.isnull(X_train[col])
            x_col = X_train[col][~nan_index]
            y_col = y_train[~nan_index]
            
            if not any(nan_index):
                X_train_new[col] = X_train[col]
                if len(X_test):X_test_new[col] = X_test[col]
                continue
            
            nan_overdue_ratio = (lambda x:sum(x)/float(len(x)))(y_train[nan_index].values)
            dfks = ks.ks_analysis(X_train[col].values,y_train.values)
            
            # 找到逾期率最接近缺失样本逾期率的分组
            g = np.abs(dfks['overdue_ratio'].values - nan_overdue_ratio).argmin()
            
            # 寻找到对应的取值范围
            str_interval = dfks['feature_interval'][g]
            
            p,q = [float(x) for x in re.sub(r'\[|\]|\)','',str_interval).split(',')]
            if ')' in str_interval:
                l = [x for x in x_col if x>=p and x<q]
            else:
                l = [x for x in x_col if x>=p and x<=q]
            
            # 计算该范围内特征的平均值
            fill_value = np.mean(l)            
            
            X_train_new[col] = X_train[col].fillna(value = fill_value)
            if len(X_test):X_test_new[col] = X_test[col].fillna(value = fill_value)
            if any(nan_index):
                X_train_new[col + '_isnan'] = nan_index.astype('int').values
                if len(X_test):X_test_new[col + '_isnan'] = pd.isnull(X_test[col]).astype('int').values
        
        
    # 如果有的X_test列有缺失值，但对应X_train不缺失，使用中位数填充策略
    for col in X_test_new.columns:
        nan_index = pd.isnull(X_test_new[col])
        if not any(nan_index):continue
        fill_value = np.median(X_train_new[col].values)
        X_test_new[col] = X_test_new[col].fillna(value = fill_value)
    return(X_train_new,X_test_new)