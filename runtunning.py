# -*- coding:utf-8 -*-
#!/usr/bin/python2.7
from __future__ import print_function

#================================================================================
# You can change the code here below! 可以改动以下配置代码修改超参优化目标和范围。
#================================================================================
# 一，配置优化目标条件

score_func = 'ks'  #优化评估指标，可以为 'ks'或'auc'
score_gap_limit  = 0.05  #可接受train和validate最大评分差值gap
n_jobs = 10  #并行任务数量

#--------------------------------------------------------------------------------
# 二，配置超参数初始值

# 初始化参数
params_dict = dict()

# 以下为待调整参数
# booster参数
params_dict['learning_rate'] = 0.5        # 学习率，初始值为 0.1，通常越小越好。
params_dict['n_estimators'] = 150         # 加法模型树的数量，初始值为50。

# tree参数
params_dict['max_depth'] = 3              # 树的深度，通常取值在[3,10]之间，初始值常取[3,6]之间
params_dict['min_child_weight']= 10       # 最小叶子节点样本权重和，越大模型越保守。
params_dict['gamma']= 0                   # 节点分裂所需的最小损失函数下降值，越大模型越保守。
params_dict['subsample']= 0.8             # 横向采样，样本采样比例，通常取值在 [0.5，1]之间 
params_dict['colsample_bytree'] = 1.0     # 纵向采样，特征采样比例，通常取值在 [0.5，1]之间 

# regulazation参数 
# Omega(f) = gamma*T + reg_alpha* sum(abs(wj)) + reg_lambda* sum(wj**2)  

params_dict['reg_alpha'] = 0              #L1 正则化项的权重系数，越大模型越保守，通常取值在[0,1]之间。
params_dict['reg_lambda'] = 1             #L2 正则化项的权重系数，越大模型越保守，通常取值在[1,100]之间。

# 以下参数通常不需要调整
params_dict['objective'] = 'binary:logistic'
params_dict['tree_method'] = 'hist'       # 构建树的策略,可以是auto, exact, approx, hist
params_dict['eval_metric'] =  'auc'
params_dict['silent'] = 1
params_dict['scale_pos_weight'] = 1        #不平衡样本时设定为正值可以使算法更快收敛。
params_dict['seed'] = 0

#--------------------------------------------------------------------------------
# 三，配置超参搜索范围

params_test1 = {'learning_rate': [0.1,0.2]} #此处应配置较大 learning_rate

params_test2 = { 'max_depth': [3], 'min_child_weight': [10,30,50,100,200,300] } 

params_test3 = {'gamma': [0,0.1,0.5,1,10]}

params_test4 = { 'subsample': [0.8,0.9,1],'colsample_bytree': [0.8,0.9,1] } 

params_test5 = { 'reg_alpha': [0,0.1,1,10] } 

params_test6 = { 'reg_lambda': [0,0.1,1,10] }

params_test7 = {'learning_rate':[0.05,0.09,0.08]} #此处应配置较小learning_rate
#===============================================================================








#================================================================================
#Don't change the code below!!! 以下代码请勿轻易改动。
#================================================================================
import sys,os,json,datetime
import numpy as np
import pandas as pd
from tianjikit.tunning import Tunning


# 定义对numpy浮点数和整数的json序列化类
class numpyJsonEncoder(json.JSONEncoder):
    def default(self, obj): 
        if isinstance(obj,(np.float,np.float32,np.float64)): 
            return float(obj)
        elif isinstance(obj, (np.int,np.int0,np.int8,np.int16,np.int32,np.int64)): 
            return int(obj)
        else: 
            return json.JSONEncoder.default(self, obj)

def main(dftrain,dftest,outputdir = './aa_tunning_results',
         score_func = score_func, score_gap_limit = score_gap_limit,
         params_dict = params_dict,n_jobs = n_jobs):
    
    # step0: 初始化
    tune = Tunning(dftrain,dftest,score_func = score_func,score_gap_limit = score_gap_limit, params_dict=params_dict,n_jobs=n_jobs)
    
    # step1: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s'%nowtime)
    print('step1: tune n_estimators for relatively high learning_rate...')
    tune.gridsearch_cv(params_test1,cv = 5,verbose_eval = True)
    
    # step2：
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s'%nowtime)
    print('step2: tune max_depth & min_child_weight...')
    tune.gridsearch_cv(params_test2,cv = 5)
    
    
    # step3：
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s'%nowtime)
    print('step3: tune gamma...')
    tune.gridsearch_cv(params_test3,cv = 5)
    
    
    # step4：
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s'%nowtime)
    print('step4: tune subsample & colsample_bytree...')
    tune.gridsearch_cv(params_test4,cv = 5)
    
    
    # step5: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s'%nowtime)
    print('step5: tune reg_alpha...')
    tune.gridsearch_cv(params_test5,cv = 5)
   
    
    # step6: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s'%nowtime)
    print('step6: tune reg_lambda...')
    tune.gridsearch_cv(params_test6,cv = 5)
    
    
    # step7: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s'%nowtime)
    print('step7: lower learning_rate...')
    tune.gridsearch_cv(params_test7,cv = 5)

    #generate results
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s'%nowtime)
    print('save results... ')
    print('tune task end. ')
   
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    with open(outputdir +'/best_parameters.json','w') as f:
        json.dump(tune.params_dict,f,cls = numpyJsonEncoder)
    try:
        tune.dfscores.to_excel(outputdir + '/dfscores.xlsx',encoding = 'utf-8')
        tune.dfparams.to_excel(outputdir + '/dfparams.xlsx',encoding = 'utf-8')
    except:
        tune.dfscores.to_csv(outputdir + '/dfscores',sep = '\t',encoding = 'utf-8')
        tune.dfparams.to_csv(outputdir + '/dfparams',sep = '\t',encoding = 'utf-8')
        
    
    return(tune.params_dict)
    
if __name__ == '__main__':
    train_data_path,test_data_path = sys.argv[1],sys.argv[2]
    dftrain = pd.read_csv(train_data_path,sep = '\t',encoding = 'utf-8')
    dftest = pd.read_csv(test_data_path,sep = '\t',encoding = 'utf-8')
    if len(sys.argv) >=4:
        outputdir = sys.argv[3]
    else:
        outputdir = './aa_tunning_results'
    main(dftrain,dftest,outputdir)  
    
####
###
##
#
    
