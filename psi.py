# -*- coding:utf-8 -*-
#!/usr/bin/python2.7
###########################################################
#update_dt:2018-05-20
#author:liangyun
#usage:计算psi稳定性指标，对于离散特征依然适用
###########################################################

import sys,math
import numpy as np
import pandas as pd


def __get_hist(data,cut_points):
    # 给定数据和分隔点计算数据的区间分布频率
    if len(cut_points) == 1:
        frequencies = [1.0]
    elif len(cut_points) == 2:
        frequencies = [sum([1.0 for x in data if x < sum(cut_points)/2.0 ])/len(data),
                       sum([1.0 for x in data if x >= sum(cut_points)/2.0 ])/len(data)]
    elif len(cut_points) == 3:
        frequencies = [sum([1.0 for x in data if x < cut_points[1] ])/len(data),
                       sum([1.0 for x in data if x >= cut_points[1] ])/len(data)]
    else:
        cnts = [len([x for x in data if x>=cut_points[i-1] and x<cut_points[i]]) 
              for i in range(2,len(cut_points)-2)]
        allcnts = [len([x for x in data if x<cut_points[1]])] + cnts +\
                [len([x for x in data if x>=cut_points[-2]])]
        frequencies = [float(x)/len(data) for x in allcnts]
    return(frequencies)

def __psi(train_frequencies,test_frequencies):

    trainf = [x if x>0 else 1e-10 for x in train_frequencies] # 修改异常值
    testf = [x if x>0 else 1e-10 for x in test_frequencies] # 修改异常值

    psi_value = sum([(testf[i] - trainf[i])*math.log(testf[i]/trainf[i]) 
               for i in range(len(testf))])
    
    return(psi_value)

def psi_analysis(train_data,test_data,parts = 10):
    
    dftrain = pd.DataFrame(train_data,columns = ['x'])
    dftest  = pd.DataFrame(test_data,columns = ['x'])

    # 处理nan值逻辑
    dftrain_nan = dftrain.loc[np.isnan(dftrain['x']),:]
    train_nan_cnt = len(dftrain_nan)
    train_notnan_ratio = 1- train_nan_cnt/float(len(dftrain)) 

    dftest_nan = dftest.loc[np.isnan(dftest['x']),:]
    test_nan_cnt = len(dftest_nan)
    test_notnan_ratio = 1- test_nan_cnt/float(len(dftest)) 

    # 初始化返回值
    dfpsi = pd.DataFrame([np.nan],columns = ['psi'])
    dfpsi['is_stable'] = [0]
    dfpsi['train_class_num'] = [len(dftrain.drop_duplicates())]
    dfpsi['test_class_num'] = [len(dftest.drop_duplicates())]
    dfpsi['train_value_num'] = [len(dftrain)]
    dfpsi['test_value_num'] = [len(dftest)]

    #若数据不足或仅有一类数据，返回 psi = nan
    if len(dftrain) <10 or len(dftest) <5 or len(dftrain.drop_duplicates())<2:
        return(dfpsi)

    dftrain = dftrain.dropna()
    dftest = dftest.dropna()

    dftrain.index,dftest.index = dftrain['x'],dftest['x']
    dftrain = dftrain.sort_index()
    dftest = dftest.sort_index()
    dftrain.index = range(len(dftrain))
    dftest.index = range(len(dftest))

    #以下代码计算psi指标
    indexmax = len(dftrain)-1
    quantile_points = [dftrain['x'][int(np.ceil(float(indexmax*i)/parts))]
                      for i in range(0,parts+1)]
    cut_points = list(pd.Series(quantile_points).drop_duplicates().values)

    train_frequencies = __get_hist(list(dftrain['x'].values),cut_points)
    test_frequencies = __get_hist(list(dftest['x'].values),cut_points)

    # 考虑nan值的处理
    if train_nan_cnt or test_nan_cnt:
        train_frequencies = [x*train_notnan_ratio for x in train_frequencies]
        test_frequencies = [x*test_notnan_ratio for x in test_frequencies]
        train_frequencies.append(1-train_notnan_ratio)
        test_frequencies.append(1-test_notnan_ratio)

    psi_value = __psi(train_frequencies,test_frequencies)

    dfpsi.loc[0,'psi'] = psi_value 
    dfpsi.loc[0,'is_stable'] = 1 if psi_value <0.2 else 0 
    return(dfpsi)