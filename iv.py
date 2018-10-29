#coding=utf-8
#!/usr/bin/python2.7

##################################################
#update_dt: 2018-05-25
#author：梁云
#usage: 计算特征iv有效性指标,对连续特征和离散特征都适用。
##################################################

import numpy as np
import pandas as pd
import math


def __iv(overdue,normal):
    overduer = [x if x>0 else 1e-10 for x in overdue] # 修改异常值
    normalr = [x if x>0 else 1e-10 for x in normal] # 修改异常值
    iv_value = sum([(overduer[i] - normalr[i])*math.log(float(overduer[i])/normalr[i]) 
               for i in range(len(normalr))])
    return(iv_value)

def iv_analysis(data,label,parts = 10):
    
    colnames = ['feature_interval',#区间
        'order_num', #订单数量
        'order_ratio', #订单占比
        'overdue_num', #逾期订单数量
        'overdue_ratio', #区间逾期订单比例
        'overdue_interval_ratio', #区间逾期订单占总逾期订单比例
        'normal_num', #正常订单数量
        'normal_ratio', #正常订单占比
        'normal_interval_ratio', #区间正常订单占总正常订单比例
        'iv_value' #iv检验值，列重复
       ]

    df = pd.DataFrame(data,columns = ['feature'])
    df['label'] = label

    total_overdue = len(df[df['label']>=0.5])
    total_normal = len(df[df['label']<0.5])

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
                  'overdue_interval_ratio':round(len(dfnan.query('label>=0.5'))/float(total_overdue),5) if total_overdue else np.nan,
                  'normal_num':len(dfnan.query('label<0.5')),
                  'normal_ratio':round(float(len(dfnan.query('label<0.5')))/len(dfnan),5),
                  'normal_interval_ratio':round(len(dfnan.query('label<0.5'))/float(total_normal),5) if total_normal else np.nan,
                  'iv_value':np.nan }

    df = df.dropna()

    # 处理特征全为nan值逻辑：
    if not len(df):
        dfiv = pd.DataFrame(columns = colnames)
        dfiv.loc[0] = dicnan
        return(dfiv)

    df.index = df['feature']
    df = df.sort_index()
    df.index = range(len(df))

    # 对非nan值计算分割点
    indexmax = len(df)-1
    quantile_points = [df['feature'][int(np.ceil(float(indexmax*i)/parts))]
                      for i in range(parts+1)]

    cut_points = list(pd.Series(quantile_points).drop_duplicates().values)

    # 处理特征只有1种取值的异常情况
    if len(cut_points) == 1:
        dfiv = pd.DataFrame()
        dfiv['feature_interval'] = ['[{},{}]'.format(cut_points[0],cut_points[0])]
        dfiv['order_num'] = [len(df)]
        dfiv['order_ratio'] = [1*notnan_ratio]
        dfiv['overdue_num'] = [len(df[df['label'] >= 0.5 ])]
        dfiv['overdue_ratio'] = [float(len(df[df['label'] >= 0.5 ]))/len(df)]
        dfiv['overdue_interval_ratio'] = [round(len(df[df['label'] >= 0.5 ])/float(total_overdue),5)]
        dfiv['normal_num'] = [len(df[df['label'] < 0.5 ])]
        dfiv['normal_ratio'] = [len(df[df['label'] < 0.5 ]) /len(df)]
        dfiv['normal_interval_ratio'] = [round(len(df[df['label'] < 0.5 ])/float(total_normal),5)]
        dfiv['iv_value'] = [np.nan]

        if nan_cnt:
            dfiv.loc[1] = dicnan
            iv_value = __iv(dfiv['overdue_interval_ratio'],dfiv['normal_interval_ratio'])
            dfiv['iv_value'] = [iv_value] * len(dfiv)
            
        return(dfiv)

    # 处理特征只有2种取值的情况   
    if len(cut_points) == 2:
        cut_points = [cut_points[0],sum(cut_points)/2.0,cut_points[1]]


    # 处理特征非nan值逻辑
    points_num = len(cut_points)
    Ldf = [0]*(points_num-1)
    for i in range(0,points_num-2):
        Ldf[i] = df.loc[(df['feature']>=cut_points[i]) \
                 & (df['feature']<cut_points[i+1]),:]    
    Ldf[points_num-2] = df.loc[(df['feature']>=cut_points[points_num-2]) \
                        & (df['feature']<=cut_points[points_num-1]),:]

    dfiv = pd.DataFrame(np.zeros((points_num -1,10)),
                        columns = colnames)

    for i in range(0,points_num-1):
        dfiv.loc[i,'feature_interval'] = '[{},{})'.format(cut_points[i],cut_points[i+1])
        dfiv.loc[i,'order_num'] = len(Ldf[i])
        dfiv.loc[i,'order_ratio'] = len(Ldf[i])*notnan_ratio/float(len(df))
        dfiv.loc[i,'overdue_num'] = len(Ldf[i][Ldf[i]['label'] >= 0.5 ])
        dfiv.loc[i,'overdue_ratio'] = float(dfiv.loc[i,'overdue_num'])/len(Ldf[i])
        dfiv.loc[i,'overdue_interval_ratio'] = float(dfiv.loc[i,'overdue_num'])/total_overdue if total_overdue else np.nan
        dfiv.loc[i,'normal_num'] = len(Ldf[i][Ldf[i]['label'] < 0.5 ])
        dfiv.loc[i,'normal_ratio'] = float(dfiv.loc[i,'normal_num'])/len(Ldf[i])
        dfiv.loc[i,'normal_interval_ratio'] = float(dfiv.loc[i,'normal_num'])/total_normal if total_normal else np.nan

    dfiv.loc[points_num-2,'feature_interval'] = \
         '[{},{}]'.format(cut_points[points_num-2],cut_points[points_num-1])

    # 在最后一行增加nan值逻辑
    if nan_cnt:
        dfiv.loc[points_num - 1] = dicnan

    iv_value = __iv(dfiv['overdue_interval_ratio'],dfiv['normal_interval_ratio'])

    dfiv['iv_value'] = [iv_value] * len(dfiv) 
    
     # 更改dfiv各整数列数据类型
    for col in ['order_num','overdue_num','normal_num']:
        dfiv[col] = dfiv[col].astype('i4')
   
    return(dfiv)

    