# -*- coding:utf-8 -*-
#!/usr/bin/python2.7
from __future__ import print_function

#================================================================================
# You can change the code here below! 可以改动以下配置代码修改超参优化目标和范围。
#================================================================================
# 一，配置优化目标条件

task_name = 'example'
score_func = 'ks'                                 #优化评估指标，可以为 'ks'或'auc'
score_gap_limit  = 0.03                           #可接受train和validate最大评分差值gap
train_data_path = './xx_train_data'               #训练集数据位置
test_data_path = './xx_test_data'                 #测试集数据位置
outputdir = './aa_everything_result_' + task_name    #输出文件夹名
n_jobs = 16                                       #并行任务数量

#--------------------------------------------------------------------------------
# 二，配置超参数初始值

# 初始化参数
params_dict = dict()

# 以下为待调整参数
# booster参数
params_dict['learning_rate'] = 0.1        # 学习率，初始值为 0.1，通常越小越好。
params_dict['n_estimators'] = 50         # 加法模型树的数量，初始值为50。

# tree参数
params_dict['max_depth'] = 3              # 树的深度，通常取值在[3,10]之间，初始值常取[3,6]之间
params_dict['min_child_weight']= 30       # 最小叶子节点样本权重和，越大模型越保守。
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

params_test1 = {'learning_rate': [0.1],'n_estimators':[50]}  #此处应配置较大 learning_rate

params_test2 = { 'max_depth': [3], 'min_child_weight': [50,100,200] } 

params_test3 = {'gamma': [0.1,0.5,1]}

params_test4 = { 'subsample': [0.9,1.0],'colsample_bytree': [1.0] } 

params_test5 = { 'reg_alpha': [0.1,1] } 

params_test6 = { 'reg_lambda': [0,0.1] }

params_test7 = {'learning_rate':[0.09,0.08],'n_estimators':[100]} #此处应配置较小learning_rate
#===============================================================================








#================================================================================
#Don't change the code below!!! 以下代码请勿轻易改动。
#================================================================================
import sys,os,json,datetime
import numpy as np
import pandas as pd
from tianjikit.analysisfeatures import AnalysisFeatures
from tianjikit.trainxgboost import TrainXgboost
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

def main(dftrain,dftest,outputdir = outputdir,n_jobs = n_jobs,
         score_func = score_func, score_gap_limit = score_gap_limit,
         params_dict = params_dict):
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        
    #================================================================================
    # 一，特征分析
    #================================================================================
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('PART 1:START ANALYSIS FEATURES...')
    
    # 基本分析
    afs = AnalysisFeatures(dftrain,dftest)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    dfbasic = afs.basic_analysises()

    # ks有效性分析
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    dfks = afs.ks_analysises()
    
    # 稳定性分析
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    dfpsi = afs.psi_analysises()
    
    print('save results... ')
    # 保存相应文件
    dfbasic.to_csv(outputdir + '/basic_analysises.csv',sep = '\t',encoding = 'utf-8')
    dfks.to_csv(outputdir +'/ks_analysises.csv',sep = '\t',encoding = 'utf-8')
    dfpsi.to_csv(outputdir + '/psi_analysises.csv',sep = '\t',encoding = 'utf-8')
    try:
        dfbasic.to_excel(outputdir + '/basic_analysises.xlsx',encoding = 'utf-8')
        dfks.to_excel(outputdir +'/ks_analysises.xlsx',encoding = 'utf-8')
        dfpsi.to_excel(outputdir + '/psi_analysises.xlsx',encoding = 'utf-8')
    except:
        pass
    
    #================================================================================
    # 二，模型调参
    #================================================================================
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('PART 2:START TUNNING XGBOOT...')
    
    # step0: 初始化
    tune = Tunning(dftrain,dftest,score_func = score_func,score_gap_limit = score_gap_limit, 
                   params_dict=params_dict,n_jobs=n_jobs)
    
    # step1: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step1: tune n_estimators for relatively high learning_rate...')
    tune.gridsearch_cv(params_test1,cv = 5,verbose_eval = 20)
    
    # step2：
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step2: tune max_depth & min_child_weight...')
    tune.gridsearch_cv(params_test2,cv = 5,verbose_eval = 20)
    
    
    # step3：
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step3: tune gamma...')
    tune.gridsearch_cv(params_test3,cv = 5,verbose_eval = 20)
    
    
    # step4：
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step4: tune subsample & colsample_bytree...')
    tune.gridsearch_cv(params_test4,cv = 5,verbose_eval = 20)
    
    
    # step5: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step5: tune reg_alpha...')
    tune.gridsearch_cv(params_test5,cv = 5,verbose_eval = 20)
   
    
    # step6: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step6: tune reg_lambda...')
    tune.gridsearch_cv(params_test6,cv = 5,verbose_eval = 20)
    
    
    # step7: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step7: lower learning_rate...')
    tune.gridsearch_cv(params_test7,cv = 5,verbose_eval = 20)
    
    # step8: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step8: train model with tuned parameters and fully train dataset...')
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    bst,dfimportance = tune.train_best()
   
    #generate results
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('save results... ')
       
    with open(outputdir +'/best_parameters.json','w') as f:
        json.dump(tune.params_dict,f,cls = numpyJsonEncoder)
        
    tune.dfmerge.to_csv(outputdir + '/dfresults',sep = '\t',encoding = 'utf-8')
    try:
        tune.dfmerge.to_excel(outputdir + '/dfresults.xlsx',encoding = 'utf-8')
    except:
        pass
    
    bst.save_model('./bst.model')
    dfimportance.to_csv('./dfimportance.csv',sep = '\t')
    
    #================================================================================
    # 二，模型报告
    #================================================================================
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('PART 3:START MODEL REPORT...')
    
    # 训练xgboost模型
    model = TrainXgboost(dftrain = dftrain,dftest = dftest, coverage_th=0, ks_th=0,
            outliers_th=None, selected_features=None)
    bst = model.train(cv=5, model_idx=1,params_dict = tune.params_dict,n_jobs = n_jobs, verbose_eval = 20) 
    model.test(bst)
    report_info = model.report_info
    
  
    # 保存相应文件
    print('\nsave results... ') 
    with open(outputdir + '/model_report','w') as f:
        f.write(report_info) 
    
    
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('task end.\n')
    
    
if __name__ == '__main__':
    print('\ntask_name: %s '%task_name)
    dftrain = pd.read_csv(train_data_path,sep = '\t',encoding = 'utf-8')
    dftest = pd.read_csv(test_data_path,sep = '\t',encoding = 'utf-8')
    main(dftrain,dftest)  
    
####
###
##
#
    

    
