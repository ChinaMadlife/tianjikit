# -*- coding:utf-8 -*-
#!/usr/bin/python2.7
###########################################################
#update_dt:2018-05-20
#author:liangyun
#usage:输入dftrain,dftest,对全部特征进行basic_analysis、
#ks_analysis、psi_analysis、chi2_analysis
###########################################################

import sys,math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
import basic,ks,chi2,psi,iv

class AnalysisFeatures(object):
    """
    Examples
    --------
    # 多特征分析示范
    import numpy as np
    import pandas as pd
    from tianjikit.analysisfeatures import AnalysisFeatures

    # 构造dftrain 训练集特征数据
    dftrain = pd.DataFrame()
    dftrain['phone'] = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12']
    dftrain['loan_dt'] = ['2018-01-01']*12
    dftrain['label'] = [0,1,1,0,1,0,0,0,0,0,1,0]
    dftrain['feature1'] = [1,0,1,0,1,0,1,0,1,0,1,1]
    dftrain['feature2'] = [1.0,2,3,4,5,6,7,8,9,10,11,12]


    # 构造dftest测试集特征
    dftest = pd.DataFrame()
    dftest['phone'] = ['y1','y2','y3','y4','y5','y6','y7','y8','y9','y10']
    dftest['loan_dt'] = ['2018-02-01']*10
    dftest['label'] = [1,0,0,1,0,0,0,1,0,0]
    dftest['feature1'] = [1,0,0,1,0,0,1,0,1,0]
    dftest['feature2'] = [10.0,9,8,7,6,5,4,3,2,1]

    AFS = AnalysisFeatures(dftrain,dftest)

    #特征基本分析
    dfBasic = AFS.BasicAnalysis()

    #特征稳定性分析
    dfPsi = AFS.PsiAnalysis()

    #特征ks分析
    dfKs = AFS.KsAnalysis()

    #特征iv分析
    dfIv = AFS.IvAnalysis()

    #特征chi2分析
    dfChi2 = AFS.Chi2Analysis()
    """
    
    def __init__(self,dftrain,dftest = pd.DataFrame()):
        
        # 若dftest未赋值，根据dftrain进行basic_analysis,否则合并
        
        dfdata = pd.concat([dftrain,dftest],ignore_index = True) \
                 if len(dftest)> 0 else dftrain
        
        # 移除非特征列
        for col in ['phone','id_card','name','loan_dt']: 
            if col in dfdata.columns:dfdata.drop(columns = col,inplace = True)
        features = list(dfdata.columns.values)
        try: features.remove('label').remove('loan_dt')
        except: pass
        
        self.__dftrain = dftrain
        self.__dftest = dftest
        self.__dfdata = dfdata
        self.__features = features
        
    def BasicAnalysis(self):
        
        #调用basic_analysis对全部特征进行基本分析
        print('start BasicAnalysis...')
        print('[total|done|todo]')
        
        dfBasic = pd.DataFrame()
        features_num = len(self.__features)
        for i,col in enumerate(self.__features):
            dfcol = basic.basic_analysis(self.__dfdata[col].values,
                    self.__dfdata['label'].values)
            dfcol.insert(0,'feature_name',[col])
            dfBasic = pd.concat([dfBasic,dfcol],ignore_index = True)
            if np.mod(i+1,100) == 0 : 
                print('[{}|{}|{}]'.format(features_num,i+1,features_num-i-1))
        print('[{}|{}|{}]'.format(features_num,i+1,features_num-i-1))
        return(dfBasic)
    
    def PsiAnalysis(self):
        
        #调用psi_analysis对全部特征进行稳定性分析
        print('start PsiAnalysis...')
        print('[total|done|todo]')
        
        dfPsi = pd.DataFrame()
        features_num = len(self.__features)
        if len(self.__dftest)<1:return(dfPsi.copy())
        for i,col in enumerate(self.__features):
            dfcol = psi.psi_analysis(self.__dftrain[col].values,
                                    self.__dftest[col].values)
            dfcol.insert(0,'feature_name',[col])
            dfPsi = pd.concat([dfPsi,dfcol],ignore_index = True)
            if np.mod(i+1,100) == 0 : 
                print('[{}|{}|{}]'.format(features_num,i+1,features_num-i-1))
        print('[{}|{}|{}]'.format(features_num,i+1,features_num-i-1))
        
        return(dfPsi)
    
    def KsAnalysis(self):
        
        #调用ks_analysis对全部特征的有效性进行ks分析
        print('start KsAnalysis...')
        print('[total|done|todo]')
        
        dfKs = pd.DataFrame()
        features_num = len(self.__features)
        for i,col in enumerate(self.__features):
            try:
                dfcol = ks.ks_analysis(self.__dfdata[col].values,
                       self.__dfdata['label'].values)
                ks_value = max(dfcol['ks_value'])
                dfcol.index = [[col]*len(dfcol),[ks_value]*len(dfcol),range(len(dfcol))]
                dfKs = pd.concat([dfKs,dfcol])
                dfKs.index.names = ['feature','ks','seq']
            except:
                pass
            if np.mod(i+1,100) == 0 : 
                print('[{}|{}|{}]'.format(features_num,i+1,features_num-i-1))
        print('[{}|{}|{}]'.format(features_num,i+1,features_num-i-1))
        
        return(dfKs)
    
    
    def IvAnalysis(self):
        #调用iv_analysis对全部特征的有效性进行ks分析
        print('start IvAnalysis...')
        print('[total|done|todo]')
        
        dfIv = pd.DataFrame()
        features_num = len(self.__features)
        for i,col in enumerate(self.__features):
            try:
                dfcol = iv.iv_analysis(self.__dfdata[col].values,
                       self.__dfdata['label'].values)
                iv_value = np.mean(dfcol['iv_value'])
                dfcol.index = [[col]*len(dfcol),[iv_value]*len(dfcol),range(len(dfcol))]
                dfIv = pd.concat([dfIv,dfcol])
                dfIv.index.names = ['feature','iv','seq']
            except:
                pass
            if np.mod(i+1,100) == 0 : 
                print('[{}|{}|{}]'.format(features_num,i+1,features_num-i-1))
        print('[{}|{}|{}]'.format(features_num,i+1,features_num-i-1))
        
        return(dfIv)
    
    
    def Chi2Analysis(self):
        
        #调用chi2_analysis对class_num<=5的特征的有效性进行卡方检验
        print('start Chi2Analysis...')
        print('[total|done|todo]')
        
        dfChi2 = pd.DataFrame()
        features_num = len(self.__features)
        
        for i,col in enumerate(self.__features):
            dfc = self.__dfdata[[col,'label']].dropna().copy()
            if len(dfc[[col]].drop_duplicates())>5:continue
            dfc[col] = dfc[col].astype('f4')
            featuredata = preprocessing.minmax_scale(dfc[col].values)
            label = dfc['label'].values
            try:
                dfcol = chi2.chi2_analysis(featuredata,label)
                dfcol.index = [col]
                dfChi2 = pd.concat([dfChi2,dfcol])
            except:
                pass
            if np.mod(i+1,100) == 0 : 
                print('[{}|{}|{}]'.format(features_num,i+1,features_num-i-1))
        print('[{}|{}|{}]'.format(features_num,i+1,features_num-i-1))
     
        return(dfChi2)