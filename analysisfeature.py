# -*- coding:utf-8 -*-
#!/usr/bin/python2.7
###########################################################
#update_dt:2018-05-20
#author:liangyun
#usage:单特征分析，用类接口对相关功能进行了包装
###########################################################

import basic,ks,chi2,psi,outliers,iv

class AnalysisFeature(object):
    """
    Examples
    --------
    import numpy as np
    import pandas as pd
    from tianjikit.analysisfeature import AnalysisFeature

    # 准备数据
    data = [1.0,2,3,4,5,6,4,3,2,1,2,9,10,100,np.nan,0,7,8,10,6]
    label = [0,1,1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,1,1]
    assert len(data)==len(label)

    af = AnalysisFeature()
    # 离群值分析
    dfoutliers = af.outliers_analysis(data,alpha = 2)

    # 去除离群值
    data_clean = af.drop_outliers(data,data,alpha = 2)

    # 基本分析
    dfbasic = af.basic_analysis(data,label)

    # psi稳定性分析
    test_data = [10,9,5,3,4,3,2,1,6,7,5,np.nan,10,100]
    dfpsi = af.psi_analysis(data,test_data)

    # ks有效性分析,主要对连续特征，对离散特征也可分析
    dfks = af.ks_analysis(data,label)

    # iv有效性分析，主要针对离散特征，对连续特征也适用
    dfiv = af.iv_analysis(data,label)

    # 卡方及召回率等分析，主要针对离散特征
    dfchi2 = af.chi2_analysis(data,label)
    
    """
    
    def basic_analysis(self,data,label):
        """
        #------覆盖率------------------------#
        'not_nan_ratio',  #非空比例，通常覆盖率coverage即是它
        'not_zero_ratio', #非零比例
        'not_outlier_ratio', #非离群值比例
        #------统计值------------------------#
        'class_num', #数据类别数目
        'value_num', #非空数据数目
        'min', #最小值
        'mean',#均值
        'med', #中位数
        'most', #众数
        'max', #最大值
        #------有效性----------------------#
        'ks(continous feature)', #ks统计量，适合连续特征
        'ks_pvalue', #ks统计量的p值
        'chi2(discrete feature)', #chi2统计量，适合离散特征
        'chi2_pvalue', #chi2统计量的p值
        't(for mean)', #均值t检验
        't_pvalue' ,#均值t检验的p值
        'z(for coverage)',#覆盖率z检验，coverage指 not_zero_ratio
        'z_pvalue', #覆盖率z检验的p值
        'iv' #iv检验值
        """
        return(basic.basic_analysis(data,label))
    
    def ks_analysis(self,data,label):
        """
        'feature_interval',#区间
        'order_num', #订单数量
        'order_ratio', #订单占比
        'overdue_num', #逾期订单数量
        'overdue_ratio', #逾期订单占比
        'normal_num', #正常订单数量
        'normal_ratio', #正常订单占比
        'overdue_cum_ratio', #累计逾期订单比例
        'normal_cum_ratio', #累计正常订单比例
        'ks_value' #ks统计值
        """
        return(ks.ks_analysis(data,label))
    
    def chi2_analysis(self,data,label):
        """
        'TP', #feature为1的逾期样本数量
        'FP', #feature为1的正常样本数量
        'TN', #feature为0的正常样本数量
        'FN', #feature为0的逾期的样本数量
        'TPR', #TP/(TP+FN),逾期样本中feature取1比例
        'FPR',#FP/(FP+TN),正常样本中feature取1比例
        'overdue_ratio_0',# feature为0样本的逾期率
        'overdue_ratio_1',# feature为1样本的逾期率
        'precision',#精度
        'accuracy',#准确度
        'chi2', #卡方统计量
        'chi2_pvalue'; #卡方统计量的p值
        """
        return(chi2.chi2_analysis(data,label))
    
    def psi_analysis(self,train_data,test_data,parts = 10):
        """
        'psi', #psi指标，仅当 train_data和 test_data 有效数据数量 >10时才取值，否则为 nan值
        'is_stable', #是否稳定，psi<0.2判定为稳定
        'train_class_num', # train_data中数据类别数目
        'test_class_num' , # test_data中数据类别数目
        'train_value_num', #train_data中有效数据数目
        'test_value_num';#test_data中有效数据数目
        """
        return(psi.psi_analysis(train_data,test_data,parts))
    
    def iv_analysis(self,data,label,parts = 10):
        """
        'feature_interval',#区间
        'order_num', #订单数量
        'order_ratio', #订单占比
        'overdue_num', #逾期订单数量
        'overdue_ratio', #逾期订单比例
        'overdue_interval_ratio', #区间逾期订单占总逾期订单比例
        'normal_num', #正常订单数量
        'normal_ratio', #正常订单占比
        'normal_interval_ratio', #区间正常订单占总正常订单比例
        'iv_value'; #iv检验值，列重复
        """
        return(iv.iv_analysis(data,label,parts))
    
    def outliers_analysis(self,data,alpha = 1.5):
        """
        'med', #中位数
        'seg_25', #1/4分位数
        'seg_75', #3/4分位数
        'up_limit',  #离群值判定上边界
        'low_limit', #离群值判定下边界
        'up_ratio',  #超上边界离群值比例
        'low_ratio';  #超下边界离群值比例
        """
        return(outliers.outliers_analysis(data,alpha))
    
    def drop_outliers(self,X_train,X_test,alpha = 1.5):
        return(outliers.drop_outliers(X_train,X_test,alpha))
    
    

