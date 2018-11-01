# -*- coding:utf-8 -*-
#!/usr/bin/python2.7
from __future__ import print_function

#================================================================================
# You can change the code here below! 可以改动以下配置代码修改超参优化目标和范围。
#================================================================================
# 一，配置优化目标条件

score_func = 'ks'  #优化评估指标，可以为 'ks'或'auc'
score_gap_limit  = 0.05  #可接受train和validate最大评分差值gap

#--------------------------------------------------------------------------------
# 二，配置超参数初始值

init_params = {}
# booster参数
init_params['learning_rate'] = 0.1        # 学习率，初始值为 0.1，通常越小越好。
init_params['n_estimators'] = 50          # 加法模型树的数量，初始值为50,通常通过xgboost.cv确定。

# tree参数
init_params['max_depth'] = 3              # 树的深度，通常取值在[3,10]之间，初始值常取[3,6]之间
init_params['min_child_weight']= 10       # 最小叶子节点样本权重和，越大模型越保守。
init_params['gamma']= 0                   # 节点分裂所需的最小损失函数下降值，越大模型越保守。
init_params['subsample']= 0.8             # 横向采样，样本采样比例，通常取值在 [0.5，1]之间 
init_params['colsample_bytree'] = 0.8     # 纵向采样，特征采样比例，通常取值在 [0.5，1]之间 

# regulazation参数 
# Omega(f) = gamma*T + reg_alpha* sum(abs(wj)) + reg_lambda* sum(wj**2)  

init_params['reg_alpha'] = 0              #L1 正则化项的权重系数，越大模型越保守，通常取值在[0,1]之间。
init_params['reg_lambda'] = 1             #L2 正则化项的权重系数，越大模型越保守，通常取值在[1,100]之间。

#--------------------------------------------------------------------------------
# 三，配置超参搜索范围

param_test1 = {'learning_rate': [0.1,0.2,0.5]} #此处应配置较大 learning_rate

param_test2 = { 'max_depth': [3,4,5,6,7,8,9,10], 'min_child_weight': [1,10,20,30,40,50,60,70,80,90,100] } 

param_test3 = {'gamma': [0,0.1,0.2,0.3,0.4,0.5]}

param_test4 = { 'subsample': [0.6,0.7,0.8,0.9,1],'colsample_bytree': [0.6,0.7,0.8,0.9,1] } 

param_test5 = { 'reg_alpha': [0,0.01,0.1,1] } 

param_test6 = { 'reg_lambda': [0,0.01,0.1,1] }

param_test7 = {'learning_rate':[0.05,0.01]} #此处应配置较小learning_rate
#===============================================================================








#================================================================================
#Don't change the code below!!! 以下代码请勿轻易改动。
#================================================================================
import sys,os,json
import numpy as np
import pandas as pd
import xgboost
from xgboost.sklearn import XGBClassifier
from tianjikit.tunning import Tunning
from copy import deepcopy
import pdb
__DEBUG__ = False

# 美化dataframe输出
from prettytable import PrettyTable
def pretty_dataframe(df):
    table = PrettyTable([''] + list(df.columns))
    for row in df.itertuples():
        table.add_row(row)
    return table

# 定义对numpy浮点数和整数的json序列化类
class numpyJsonEncoder(json.JSONEncoder):
    def default(self, obj): 
        if isinstance(obj,(np.float,np.float32,np.float64)): 
            return float(obj)
        elif isinstance(obj, (np.int,np.int0,np.int8,np.int16,np.int32,np.int64)): 
            return int(obj)
        else: 
            return json.JSONEncoder.default(self, obj)

def main(train_data_path,test_data_path,outputdir = './aa_tunning_results'):
    
    params_list = ['learning_rate','n_estimators','max_depth','min_child_weight','gamma','subsample',
                  'colsample_bytree','reg_alpha','reg_lambda']
    # 获取数据
    print('\n================================================================================\n')
    print('start reading data...')
    dftrain = pd.read_csv(train_data_path,sep = '\t',encoding = 'utf-8')
    dftest = pd.read_csv(test_data_path,sep = '\t',encoding = 'utf-8')
    
    print('\n================================================================================\n')
    
    # step 0
    print('step0: evaluate init params score...')
    model = XGBClassifier()
    tune = Tunning(model=model,dftrain=dftrain,dftest=dftest,cv = 5,score_func = score_func,
           score_gap_limit = score_gap_limit,params_dict=init_params,n_jobs=-1,selected_features=None)
    dfscore = tune.dfscore[['train_score','validate_score','score_gap','test_score'] ]
    print('init model scores:')
    print(pretty_dataframe(dfscore))
    print('\n')
    
    dfpa_list = tune.dfscore[params_list]
    #dfpa_list.columns = ['p' + str(i) for i in range(len(params_list))]
    print('init params values:')
    print(pretty_dataframe(dfpa_list))

    print('\n================================================================================\n')

    # step 1
    print('step1: tune n_estimators for relatively high learning_rate...')
    alpha_list = param_test1.get('learning_rate',[0.1])
    for alpha in alpha_list:
        param_test = { 'learning_rate': alpha, 'n_estimators':1000}
        tune.params_dict.update(param_test)
        tune.model.set_params(**tune.params_dict)
        tune.xgboost_cv(cv= 5, early_stopping_rounds= 50,n_jobs = -1)   
    dfscore = tune.dfscore.copy()
    dfscore_filter = dfscore.query('score_gap<{}'.format(score_gap_limit)).copy()
    if len(dfscore_filter)<1:
        dfscore_filter = pd.DataFrame(dfscore.loc[dfscore['score_gap'].astype('f4').idxmin(),:]).T
        
    best_id = dfscore_filter['validate_score'].astype('f4').idxmax()
    tune.params_dict = deepcopy(tune.dfparams.loc[best_id,'params_dict'])
    
    dfscore = tune.dfscore[['train_score','validate_score','score_gap','test_score'] ]
    print('\nhistory tunning scores:')
    print(pretty_dataframe(dfscore))
    
    dfpa_list = tune.dfscore[params_list]
    #dfpa_list.columns = ['p' + str(i) for i in range(len(params_list))]
    print('\nhistory tunning params:')
    print(pretty_dataframe(dfpa_list))
    print('\n================================================================================\n')
    
    if __DEBUG__: pdb.set_trace()##********************调试断点***********************##
    
    #step 2
    print('step2：tune max_depth & min_child_weight...') 
    best_param = tune.gridsearch_cv(param_test2,n_jobs = -1)
    dfscore = tune.dfscore[['train_score','validate_score','score_gap','test_score'] ]
    print('\nhistory tunning scores:')
    print(pretty_dataframe(dfscore))
    
    
    dfpa_list = tune.dfscore[params_list]
    #dfpa_list.columns = ['p' + str(i) for i in range(len(params_list))]
    print('\nhistory tunning params:')
    print(pretty_dataframe(dfpa_list))
    print('\n================================================================================\n')

    # step 3
    print('step3：tune gamma...')
    best_param = tune.gridsearch_cv(param_test3,n_jobs = -1)
    dfscore = tune.dfscore[['train_score','validate_score','score_gap','test_score'] ]
    print('\nhistory tunning scores:')
    print(pretty_dataframe(dfscore))
    
    dfpa_list = tune.dfscore[params_list]
    #dfpa_list.columns = ['p' + str(i) for i in range(len(params_list))]
    print('\nhistory tunning params:')
    print(pretty_dataframe(dfpa_list))
    print('\n================================================================================\n')

    # step 4
    print('step4：tune subsample & colsample_bytree...')  
    best_param = tune.gridsearch_cv(param_test4,n_jobs = -1)
    dfscore = tune.dfscore[['train_score','validate_score','score_gap','test_score'] ]
    print('\nhistory tunning scores:')
    print(pretty_dataframe(dfscore))

    
    dfpa_list = tune.dfscore[params_list]
    #dfpa_list.columns = ['p' + str(i) for i in range(len(params_list))]
    print('\nhistory tunning params:')
    print(pretty_dataframe(dfpa_list))
    print('\n================================================================================\n')

    # step 5
    print('step5: tune reg_alpha...') 
    best_param = tune.gridsearch_cv(param_test5,n_jobs = -1)
    dfscore = tune.dfscore[['train_score','validate_score','score_gap','test_score'] ]
    print('\nhistory tunning scores:')
    print(pretty_dataframe(dfscore))
    
    dfpa_list = tune.dfscore[params_list]
    #dfpa_list.columns = ['p' + str(i) for i in range(len(params_list))]
    print('\nhistory tunning params:')
    print(pretty_dataframe(dfpa_list))
    print('\n================================================================================\n')

    # step 6
    print('step6: tune reg_lambda...') 
    best_param = tune.gridsearch_cv(param_test6,n_jobs = -1)
    dfscore = tune.dfscore[['train_score','validate_score','score_gap','test_score'] ]
    print('\nhistory tunning scores:')
    print(pretty_dataframe(dfscore))
 
    dfpa_list = tune.dfscore[params_list]
    #dfpa_list.columns = ['p' + str(i) for i in range(len(params_list))]
    print('\nhistory tunning params:')
    print(pretty_dataframe(dfpa_list))
    print('\n================================================================================\n')

    if __DEBUG__: pdb.set_trace()##********************调试断点***********************##
    
    # step 7
    print('step7: lower learning_rate and rise n_estimators...')
    alpha_list = param_test7.get('learning_rate',[0.05])
    for alpha in alpha_list:
        param_test = { 'learning_rate': alpha, 'n_estimators':1000}
        tune.params_dict.update(param_test)
        tune.model.set_params(**tune.params_dict)
        tune.xgboost_cv(cv= 5, early_stopping_rounds= 100,n_jobs = -1)
    dfscore = tune.dfscore[['train_score','validate_score','score_gap','test_score'] ]
    print('\nhistory tunning scores:')
    print(pretty_dataframe(dfscore))
    
    dfpa_list = tune.dfscore[params_list]
    #dfpa_list.columns = ['p' + str(i) for i in range(len(params_list))]
    print('\nhistory tunning params:')
    print(pretty_dataframe(dfpa_list))
    print('\n================================================================================\n')
    
    if __DEBUG__: pdb.set_trace()##********************调试断点***********************##

    ###generate results
    print('generate results... ')
    dfscore = tune.dfscore.copy()
    dfscore_filter = dfscore.query('score_gap<{}'.format(score_gap_limit)).copy()
    if len(dfscore_filter)<1:
        dfscore_filter = pd.DataFrame(dfscore.loc[dfscore['score_gap'].astype('f4').idxmin(),:]).T
    best_id = dfscore_filter['validate_score'].astype('f4').idxmax()
    
    if __DEBUG__: pdb.set_trace()##********************调试断点***********************##
        
    tune.params_dict = deepcopy(tune.dfparams.loc[best_id,'params_dict'])
    print('\n========================================\n')
    print('best_params:')
    best_params = tune.dfscore.loc[[best_id],params_list]
    print(pretty_dataframe(best_params))
    print('\n========================================\n')
    print('best_score:')
    best_score = tune.dfscore.loc[[best_id],['train_score','validate_score','score_gap','test_score']]
    print(pretty_dataframe(best_score))
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    if __DEBUG__: pdb.set_trace()##********************调试断点***********************##
    
    with open(outputdir +'/best_parameters.json','w') as f:
        json.dump(tune.params_dict,f,cls = numpyJsonEncoder)
    tune.dfscore.to_excel(outputdir + '/dfscore.xlsx',encoding = 'utf-8')
    
    return(tune.params_dict)
    
if __name__ == '__main__':
    train_data_path,test_data_path = sys.argv[1],sys.argv[2]
    if len(sys.argv) >=4:
        outputdir = sys.argv[3]
    else:
        outputdir = './aa_tunning_results'
    main(train_data_path,test_data_path,outputdir)  
    
####
###
##
#
    
