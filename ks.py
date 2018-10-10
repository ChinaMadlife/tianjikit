# -*- coding=utf-8 -*-
#!/usr/bin/python2.7

##################################################
#update_dt: 2018-08-23
#author：梁云
##################################################

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from io import StringIO

def ks_analysis(data,label,parts = 10):

    df = pd.DataFrame(data,columns = ['feature'])
    df['label'] = label
    df = df.dropna()
    df.index = df['feature']
    df = df.sort_index()
    df.index = range(len(df))

    indexmax = len(df)-1
    quantile_points = [df['feature'][int(np.ceil(float(indexmax*i)/parts))]
                      for i in range(parts+1)]
    
    cut_points = list(pd.Series(quantile_points).drop_duplicates().values)
    if len(cut_points) == 2:
        cut_points = [cut_points[0],sum(cut_points)/2.0,cut_points[1]]
    points_num = len(cut_points)

    Ldf = [0]*(points_num-1)
    for i in range(0,points_num-2):
        Ldf[i] = df.loc[(df['feature']>=cut_points[i]) \
                 & (df['feature']<cut_points[i+1]),:]    
    Ldf[points_num-2] = df.loc[(df['feature']>=cut_points[points_num-2]) \
                        & (df['feature']<=cut_points[points_num-1]),:]

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

    dfks = pd.DataFrame(np.zeros((points_num -1,10)),columns = colnames)

    total_overdue = len(df[df['label']>=0.5])
    total_normal = len(df[df['label']<0.5])

    for i in range(0,points_num-1):
        dfks.loc[i,'feature_interval'] = '[{},{})'.format(np.round(cut_points[i],5),np.round(cut_points[i+1],5))
        dfks.loc[i,'order_num'] = len(Ldf[i])
        dfks.loc[i,'order_ratio'] = round(len(Ldf[i])/float(len(df)),2)
        dfks.loc[i,'overdue_num'] = len(Ldf[i][Ldf[i]['label'] >= 0.5 ])
        dfks.loc[i,'overdue_ratio'] = round(float(dfks.loc[i,'overdue_num'])/len(Ldf[i]),5)
        dfks.loc[i,'normal_num'] = len(Ldf[i][Ldf[i]['label'] < 0.5 ])
        dfks.loc[i,'normal_ratio'] = round(float(dfks.loc[i,'normal_num'])/len(Ldf[i]),2)
        dfks.loc[i,'overdue_cum_ratio'] = round(sum(dfks['overdue_num'])/float(total_overdue) if total_overdue else np.nan,5)
        dfks.loc[i,'normal_cum_ratio'] = round(sum(dfks['normal_num'])/float(total_normal) if total_normal else np.nan,5)
        dfks.loc[i,'ks_value'] = round(np.abs(dfks.loc[i,'overdue_cum_ratio']\
                                 -dfks.loc[i,'normal_cum_ratio']),5)

    dfks.loc[points_num-2,'feature_interval'] = \
         '[{},{}]'.format(np.round(cut_points[points_num-2],5),np.round(cut_points[points_num-1],5))
    
    # 更改dfks各整数列数据类型
    for col in ['order_num','overdue_num','normal_num']:
        dfks[col] = dfks[col].astype('i4')
    
    return(dfks)


def _PrettyDataFrame(df):
    table = PrettyTable([''] + list(df.columns))
    for row in df.itertuples():
        table.add_row(row)
    return table

# 打印格式化ks矩阵
def print_ks(data,label):
    
    dfks = ks_analysis(data,label)
    dfks = dfks.drop(['normal_num','normal_ratio','overdue_cum_ratio','normal_cum_ratio'],axis = 1)
    #cols = [u'评分区间',u'订单数量',u'订单占比',u'逾期数量',u'逾期占比',u'正常数量',
           #u'正常占比',u'累计逾期',u'累计正常',u'ks取值']
    #cols = [u'评分区间',u'订单数量',u'订单占比',u'逾期数量',u'逾期占比',u'ks取值']
    #dfks.columns = cols
    
    output = _PrettyDataFrame(dfks)
    print(output)
    return(dfks)

    