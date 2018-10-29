# -*- coding=utf-8 -*-
#!/usr/bin/python2.7

##################################################
#update_dt: 2018-08-23
#author：梁云
##################################################
from __future__ import print_function
import numpy as np
import pandas as pd
from prettydf import pretty_dataframe

def ks_analysis(data,label,parts = 10):
    
    colnames = ['feature_interval',#区间
            'order_num', #订单数量
            'order_ratio', #订单占比
            'overdue_num', #逾期订单数量
            'overdue_ratio', #逾期订单占比
            'normal_num', #正常订单数量
            'normal_ratio', #正常订单占比
            'overdue_cum_ratio', #累计逾期订单比例
            'normal_cum_ratio', #累计正常订单比例
            'ks_value' #ks统计值
           ]

    df = pd.DataFrame(data,columns = ['feature'])
    df['label'] = label

    # 处理nan值逻辑
    dfnan = df.loc[np.isnan(df['feature']),:]
    nan_cnt = len(dfnan)
    notnan_ratio = 1- nan_cnt/float(len(df)) 
    if nan_cnt:
        dicnan = {'feature_interval':'[nan,nan]',
                  'order_num':nan_cnt,
                  'order_ratio':round(1-notnan_ratio,2),
                  'overdue_num':len(dfnan.query('label>=0.5')),
                  'overdue_ratio':round(float(len(dfnan.query('label>=0.5')))/len(dfnan),5),
                  'normal_num':len(dfnan.query('label<0.5')),
                  'normal_ratio':round(float(len(dfnan.query('label<0.5')))/len(dfnan),5),
                  'overdue_cum_ratio':np.nan,
                  'normal_cum_ratio':np.nan,
                  'ks_value':np.nan }
        
    df = df.dropna()
    
    # 处理特征全为nan值逻辑：
    if not len(df):
        dfks = pd.DataFrame(columns = colnames)
        dfks.loc[0] = dicnan
        return(dfks)
    
    # 对非缺失值寻找分割点
    df.index = df['feature']
    df = df.sort_index()
    df.index = range(len(df))

    indexmax = len(df)-1
    quantile_points = [df['feature'][int(np.ceil(float(indexmax*i)/parts))]
                      for i in range(parts+1)]

    cut_points = list(pd.Series(quantile_points).drop_duplicates().values)
    
    # 处理特征只有1种非nan取值的异常情况
    if len(cut_points) == 1:
        dfks = pd.DataFrame()
        dfks['feature_interval'] = ['[{},{}]'.format(cut_points[0],cut_points[0])]
        dfks['order_num'] = [len(df)]
        dfks['order_ratio'] = [1*notnan_ratio]
        dfks['overdue_num'] = [len(df.query('label>=0.5'))]
        dfks['overdue_ratio'] = [float(len(df.query('label>=0.5')))/len(df)]
        dfks['normal_num'] = [len(df.query('label<0.5'))]
        dfks['normal_ratio'] = [float(len(df.query('label<0.5')))/len(df)]
        dfks['overdue_cum_ratio'] = [1]
        dfks['normal_cum_ratio'] = [1]
        dfks['ks_value'] = [np.nan]
        
        if nan_cnt:
            dfks.loc[1] = dicnan
        return(dfks)
    
    # 处理特征只有2种非nan取值的情况
    if len(cut_points) == 2:
        cut_points = [cut_points[0],sum(cut_points)/2.0,cut_points[1]]
    points_num = len(cut_points)
    
    
    # 处理非nan数据逻辑
    Ldf = [0]*(points_num-1)
    for i in range(0,points_num-2):
        Ldf[i] = df.loc[(df['feature']>=cut_points[i]) \
                 & (df['feature']<cut_points[i+1]),:]    
    Ldf[points_num-2] = df.loc[(df['feature']>=cut_points[points_num-2]) \
                        & (df['feature']<=cut_points[points_num-1]),:]
    
    dfks = pd.DataFrame(np.zeros((points_num -1,10)),columns = colnames)

    total_overdue = len(df[df['label']>=0.5])
    total_normal = len(df[df['label']<0.5])

    for i in range(0,points_num-1):
        dfks.loc[i,'feature_interval'] = '[{},{})'.format(np.round(cut_points[i],5),np.round(cut_points[i+1],5))
        dfks.loc[i,'order_num'] = len(Ldf[i])
        dfks.loc[i,'order_ratio'] = round(len(Ldf[i])*notnan_ratio/float(len(df)),2)
        dfks.loc[i,'overdue_num'] = len(Ldf[i].query('label>=0.5'))
        dfks.loc[i,'overdue_ratio'] = round(float(dfks.loc[i,'overdue_num'])/len(Ldf[i]),5)
        dfks.loc[i,'normal_num'] = len(Ldf[i].query('label<0.5'))
        dfks.loc[i,'normal_ratio'] = round(float(dfks.loc[i,'normal_num'])/len(Ldf[i]),2)
        dfks.loc[i,'overdue_cum_ratio'] = round(sum(dfks['overdue_num'])/float(total_overdue) if total_overdue else np.nan,5)
        dfks.loc[i,'normal_cum_ratio'] = round(sum(dfks['normal_num'])/float(total_normal) if total_normal else np.nan,5)
        dfks.loc[i,'ks_value'] = round(np.abs(dfks.loc[i,'overdue_cum_ratio']\
                                 -dfks.loc[i,'normal_cum_ratio']),5)

    dfks.loc[points_num-2,'feature_interval'] = \
         '[{},{}]'.format(np.round(cut_points[points_num-2],5),np.round(cut_points[points_num-1],5))
    
    # 在最后一行增加nan值逻辑
    if nan_cnt:
        dfks.loc[points_num - 1] = dicnan

    # 更改dfks各整数列数据类型
    for col in ['order_num','overdue_num','normal_num']:
        dfks[col] = dfks[col].astype('i4')
    
    return(dfks)


# 格式化ks矩阵字符串
def print_ks(data,label):
    
    dfks = ks_analysis(data,label)
    dfks = dfks.drop(['normal_num','normal_ratio','overdue_cum_ratio','normal_cum_ratio'],axis = 1)
    #cols = [u'评分区间',u'订单数量',u'订单占比',u'逾期数量',u'逾期占比',u'正常数量',
           #u'正常占比',u'累计逾期',u'累计正常',u'ks取值']
    #cols = [u'评分区间',u'订单数量',u'订单占比',u'逾期数量',u'逾期占比',u'ks取值']
    #dfks.columns = cols
    
    output = pretty_dataframe(dfks)
    return(output)

    