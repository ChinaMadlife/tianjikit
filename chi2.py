#coding=utf-8
#!/usr/bin/python2.7

##################################################
#update_dt: 2018-05-25
#author：梁云
##################################################

import numpy as np
import pandas as pd

from sklearn import feature_selection

def chi2_analysis(data,label):

    df = pd.DataFrame(data,columns = ['feature'])
    df['label'] = label
    df = df.dropna()

    TP = len(df[(df['label']>=0.5) & (df['feature']>=0.5)])
    FP = len(df[(df['label']<0.5) & (df['feature']>=0.5)])
    TN = len(df[(df['label']<0.5) & (df['feature']<0.5)])
    FN = len(df[(df['label']>=0.5) & (df['feature']<0.5)])

    TPR = float(TP)/(TP+FN) if (TP+FN)>=1 else np.nan
    FPR = float(FP)/(FP+TN) if (FP+TN)>=1 else np.nan

    overdue_ratio_0 = float(FN)/(FN+TN) if (FN+TN)>=1 else np.nan
    overdue_ratio_1 = float(TP)/(TP+FP) if (TP+FP)>=1 else np.nan

    precision = TP/float(TP+FP) if (TP+FP)>=1 else np.nan
    accuracy = (TP+TN)/float(TP+FN+FP+TN) if (TP+FN+FP+TN)>=1 else np.nan

    chi2, chi2_pvalue = feature_selection.chi2(
                        df['feature'].values.reshape(-1,1), df['label']) if len(df)>=5 else (np.nan,np.nan)

    colnames = ['TP','FP','TN','FN','TPR','FPR',
                'overdue_ratio_0','overdue_ratio_1',
               'precision','accuracy',
                'chi2', 'chi2_pvalue']

    dfchi2 = pd.DataFrame(np.array([TP,FP,TN,FN,TPR,FPR,\
             overdue_ratio_0,overdue_ratio_1,precision,accuracy,\
             chi2,chi2_pvalue]).reshape(1,-1),columns = colnames)
    
    return(dfchi2)