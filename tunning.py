# -*- coding:utf-8 -*-
#!/usr/bin/python2.7
###########################################################
#update_dt:2018-10-15
#author:liangyun
#usage: 超参数调试，主要针对xgboost
###########################################################

from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import stats
from copy import deepcopy

from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import xgboost
from xgboost.sklearn import XGBClassifier

#import pdb
#__DEBUG__ = False

# 定义ks评分指标,供用户调用
def ks(label,feature):
    assert len(feature) == len(label)
    df = pd.DataFrame(data = np.array([feature,label]).T,columns = ['feature','label'])
    df_0,df_1 = df[df['label']<0.5],df[df['label']>=0.5]
    ks,ks_pvalue = stats.ks_2samp(df_0['feature'].values,df_1['feature'].values)
    return ks

# 定义ks评分指标,供xgboost.cv函数的feval调用
def ks_feval(preds,xgbtrain):
    label = xgbtrain.get_label()
    ks_value = ks(label,preds)
    return 'ks',ks_value

# 定义ks评分指标,供GridSearchCV函数的scoring调用
def ks_scoring(estimater, X, y):
    preds = estimater.predict_proba(X)[:,1]
    ks_value = ks(y,preds)
    return ks_value

scoring_dict = {'auc':'roc_auc','ks':ks_scoring}
feval_dict = {'auc':None,'ks':ks_feval}
score_dict = {'auc':roc_auc_score,'ks':ks}

# 美化dataframe输出
from prettytable import PrettyTable
def pretty_dataframe(df):
    table = PrettyTable([''] + list(df.columns))
    for row in df.itertuples():
        table.add_row(row)
    return table

params_dict = dict()

# 以下为待调整参数
# booster参数
params_dict['learning_rate'] = 0.1        # 学习率，初始值为 0.1，通常越小越好。
params_dict['n_estimators'] = 50         # 加法模型树的数量，初始值为50，通常通过模型cv确认。

# tree参数
params_dict['max_depth'] = 3              # 树的深度，通常取值在[3,10]之间，初始值常取[3,6]之间
params_dict['min_child_weight']= 10         # 最小叶子节点样本权重和，越大模型越保守。
params_dict['gamma']= 0                   # 节点分裂所需的最小损失函数下降值，越大模型越保守。
params_dict['subsample']= 0.8             # 横向采样，样本采样比例，通常取值在 [0.5，1]之间 
params_dict['colsample_bytree'] = 0.8     # 纵向采样，特征采样比例，通常取值在 [0.5，1]之间 

# regulazation参数 
# Omega(f) = gamma*T + reg_alpha* sum(abs(wj)) + reg_lambda* sum(wj**2)  

params_dict['reg_alpha'] = 0              #L1 正则化项的权重系数，越大模型越保守，通常取值在[0,1]之间。
params_dict['reg_lambda'] = 1             #L2 正则化项的权重系数，越大模型越保守，通常取值在[1,100]之间。

# 以下参数通常不需要调整
params_dict['objective'] = 'binary:logistic'
params_dict['n_jobs'] = 4
params_dict['scale_pos_weight'] = 1        #不平衡样本时设定为正值可以使算法更快收敛。
params_dict['seed'] = 0


class Tunning(object):
    """  
    Examples:
    --------
    from __future__ import print_function
    import numpy as np
    import pandas as pd
    import xgboost
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from xgboost.sklearn import XGBClassifier

    from tianjikit.tunning import Tunning

    data,label = datasets.make_classification(n_samples= 10000, n_features=20, n_informative= 6 ,
                 n_classes=2, n_clusters_per_class=10,random_state=0)
    dfdata = pd.DataFrame(data,columns = [u'f'+str(i) for i in range(data.shape[1])])
    dfdata['label'] = label
    dftrain,dftest = train_test_split(dfdata)
    
    params_dict = dict()

    # 以下为待调整参数
    # booster参数
    params_dict['learning_rate'] = 0.1        # 学习率，初始值为 0.1，通常越小越好。
    params_dict['n_estimators'] = 50          # 加法模型树的数量，初始值为50，通常通过xgboost自带模型cv确认。

    # tree参数
    params_dict['max_depth'] = 3              # 树的深度，通常取值在[3,10]之间，初始值常取[3,6]之间
    params_dict['min_child_weight']=10        # 最小叶子节点样本权重和，越大模型越保守。
    params_dict['gamma']= 0                   # 节点分裂所需的最小损失函数下降值，越大模型越保守。
    params_dict['subsample']= 0.8             # 横向采样，样本采样比例，通常取值在 [0.5，1]之间 
    params_dict['colsample_bytree'] = 0.8     # 纵向采样，特征采样比例，通常取值在 [0.5，1]之间 

    # regulazation参数 
    # Omega(f) = gamma*T + reg_alpha* sum(abs(wj)) + reg_lambda  
    params_dict['reg_alpha'] = 0              #L1 正则化项的权重系数，越大模型越保守，通常取值在[0,1]之间。
    params_dict['reg_lambda'] = 1             #L2 正则化项的权重系数，越大模型越保守，通常取值在[1,100]之间。

    # 以下参数通常不需要调整
    params_dict['objective'] = 'binary:logistic'
    params_dict['n_jobs'] = 4                 #此 n_jobs为xgboost多线程参数，与GridSearchCV中的n_jobs不同。
    params_dict['scale_pos_weight'] = 1       #不平衡样本时设定为正值可以使算法更快收敛。
    params_dict['seed'] = 0
    
    # step0: 初始化
    model = XGBClassifier()
    tune = Tunning(model=model,dftrain=dftrain,dftest=dftest,cv = 5,score_func = 'ks',
           score_gap_limit = 0.05,params_dict=params_dict,n_jobs=4,selected_features=None)
    tune.dfscore
    
    # step1: tune n_estimators for relatively high learning_rate (eg: 0.1)
    param_test1 = { 'learning_rate': 0.1, 'n_estimators':1000}
    tune.params_dict.update(param_test1)
    tune.model.set_params(**tune.params_dict)
    tune.xgboost_cv(cv= 5, early_stopping_rounds= 100,n_jobs = 4,seed = 0)
    tune.dfscore
    
    # step2：tune max_depth & min_child_weight 
    param_test2 = { 'max_depth': [3,4,5,6,7], 'min_child_weight': [1,10,20,30,40,50] } 
    best_param = tune.gridsearch_cv(param_test2,n_jobs = -1)
    tune.dfscore
    
    # step3：tune gamma
    param_test3 = {'gamma': [0,0.1,0.2,0.3,0.4,0.5]]}
    best_param = tune.gridsearch_cv(param_test3,n_jobs = -1)
    tune.dfscore
    
    # step4：tune subsample & colsample_bytree 
    param_test4 = { 'subsample': [0.6,0.7,0.8,0.9,1],
                   'colsample_bytree': [0.6,0.7,0.8,0.9,1] } 
    best_param = tune.gridsearch_cv(param_test4,n_jobs = -1)
    tune.dfscore
    
    # step5: tune reg_alpha 
    param_test5 = { 'reg_alpha': [0, 0.01, 0.1, 1, 10, 100] } 
    best_param = tune.gridsearch_cv(param_test5,n_jobs = -1)
    tune.dfscore
    
    # step6: tune reg_lambda 
    param_test6 = { 'reg_lambda': [0, 0.01, 0.1, 1, 10, 100] }
    best_param = tune.gridsearch_cv(param_test6,n_jobs = -1)
    tune.dfscore
    
    # step7: lower learning_rate and rise n_estimators
    param_test7 = { 'learning_rate': 0.01, 'n_estimators':1000}
    tune.params_dict.update(param_test7)
    tune.model.set_params(**tune.params_dict)
    tune.xgboost_cv(cv= 5, early_stopping_rounds= 100,n_jobs = -1)
    tune.dfscore 
    
    """
    
    def __init__(self, model, dftrain, dftest, params_dict = params_dict, n_jobs = -1, cv = 5, 
                 score_func = 'ks',score_gap_limit = 0.05,selected_features = None):
        
        self.model,self.__score_func,self.score_gap_limit = model,score_func,score_gap_limit
        self.dftrain,self.dftest = dftrain,dftest
        
        # self.params_dict存储最新的特征
        self.params_dict = params_dict.copy()
        self.model.set_params(**params_dict)
        
        
        # self.dfscore存储全部得分记录，self.dfparams存储全部参数记录
        self.dfscore = pd.DataFrame(columns = ['model_id','train_score','validate_score','score_gap','test_score'] + 
                     ['learning_rate','n_estimators','max_depth','min_child_weight','gamma','subsample','colsample_bytree','reg_alpha','reg_lambda'])
        
        self.dfparams = pd.DataFrame(columns = ['model_id','params_dict'])
        
        # 去掉['phone','id','idcard','id_card','loan_dt','name','id_map']等非特征列
        for  col in ['name','phone','id','id-card','idcard','id_card','loan_dt','id_map','idmap','id-map']:
            if col in self.dftrain.columns:
                self.dftrain = self.dftrain.drop([col],axis = 1)
                self.dftest = self.dftest.drop([col],axis = 1)
                
        # 如果selected_features 不为空，则进行特征筛选
        if selected_features:
            remained_cols = [col for col in self.dftrain.columns if col in selected_features + ['label']]
            self.dftrain = self.dftrain[remained_cols]
            self.dftest = self.dftest[remained_cols]
                
        # 制作特征名称映射表并更改特征名，复杂的特征名可能导致xgboost出错
        self.__feature_dict = {'feature'+ str(i): name for i,name in enumerate(self.dftrain.columns.drop('label'))}
        self.__inverse_feature_dict = dict(zip(self.__feature_dict.values(),self.__feature_dict.keys()))
        self.dftrain.columns = [self.__inverse_feature_dict.get(x,x) for x in self.dftrain.columns]
        self.dftest.columns = [self.__inverse_feature_dict.get(x,x) for x in self.dftest.columns]
                    
        # 分割feature和label
        self.X_train = self.dftrain.drop(['label'],axis = 1)
        self.y_train = self.dftrain[['label']]
        self.X_test = self.dftest.drop(['label'],axis = 1) 
        self.y_test = self.dftest[['label']]
        
        # 计算初始得分
        test_param = {'n_estimators':[self.model.get_params()['n_estimators']]}
        gsearch = GridSearchCV(estimator=self.model, param_grid= test_param, 
                       scoring= scoring_dict[self.__score_func], n_jobs= n_jobs, iid=False, cv=cv,
                       return_train_score=True) 
        gsearch.fit(self.X_train, np.ravel(self.y_train)) 
        dfcv_results = pd.DataFrame(gsearch.cv_results_)
        dfcv_simple = dfcv_results[['params','mean_train_score','mean_test_score']]
        train_score,validate_score = dfcv_simple.loc[0,['mean_train_score','mean_test_score']]
        score_gap = train_score - validate_score
        test_score = score_dict[self.__score_func](np.ravel(self.y_test),gsearch.predict_proba(self.X_test)[:,1])
        
        # 录入初始得分和初始参数
        dic_score = {'model_id':0,'train_score':train_score,
                     'score_gap':score_gap,'validate_score':validate_score,'test_score':test_score}
        dic_score.update(self.params_dict)
        i = len(self.dfscore)
        self.dfscore.loc[i,:] = deepcopy(dic_score)
        self.dfparams.loc[i] = {'model_id':i,'params_dict':self.params_dict.copy()}
        
    def xgboost_cv(self, early_stopping_rounds=50, cv = 5, n_jobs=4, seed=0):
        
        xgb_param = self.model.get_xgb_params() 
        xgtrain = xgboost.DMatrix(self.X_train, label= self.y_train) 
        cvresult = xgboost.cv(xgb_param, xgtrain, num_boost_round = self.model.get_params()['n_estimators'],
                          nfold=cv, metrics='auc',feval= feval_dict[self.__score_func], seed=seed, 
                          callbacks=[xgboost.callback.early_stop(early_stopping_rounds, maximize = True)])
        
        # 在score_gap < score_gap_limit条件下找到test_score最大时的n_estimators
        cvresult['score_gap'] =  cvresult['train-{}-mean'.format(self.__score_func)] - \
                                 cvresult['test-{}-mean'.format(self.__score_func)]
        cvresult_filter = cvresult.loc[cvresult['score_gap']<=self.score_gap_limit].copy()
        
        
        if len(cvresult_filter)<1:
            cvresult_filter = pd.DataFrame(cvresult.loc[cvresult['score_gap'].idxmin(),:]).T
        num_round_best = cvresult_filter['test-{}-mean'.format(self.__score_func)].astype('f4').idxmax()
        print('Best n_estimators considered score_gap_limit: ', num_round_best) 
        self.params_dict.update({'n_estimators':num_round_best})
        
        # 计算更新n_estimators后的得分
        self.model = self.model.set_params(**self.params_dict)
        test_param = {'n_estimators':[num_round_best]}
        gsearch = GridSearchCV(estimator=self.model, param_grid= test_param, 
                       scoring=scoring_dict[self.__score_func], n_jobs=n_jobs, iid=False, cv=cv,
                       return_train_score=True) 
        gsearch.fit(self.X_train, np.ravel(self.y_train)) 
        
        dfcv_results = pd.DataFrame(gsearch.cv_results_)
        dfcv_simple = dfcv_results[['params','mean_train_score','mean_test_score']].copy()
        train_score,validate_score = dfcv_simple.loc[0,['mean_train_score','mean_test_score']]
        score_gap = train_score - validate_score
        test_score = score_dict[self.__score_func](np.ravel(self.y_test),gsearch.predict_proba(self.X_test)[:,1])
        
        #if __DEBUG__: pdb.set_trace()##********************调试断点***********************##
        
        # 录入得分和参数
        i = len(self.dfscore)
        dic_score = {'model_id':i,'train_score':train_score,'validate_score':validate_score,
                     'score_gap':score_gap,'test_score':test_score}
        dic_score.update(self.params_dict)
        self.dfscore.loc[i,:] = deepcopy(dic_score)
        self.dfparams.loc[i] = {'model_id':i,'params_dict':self.params_dict.copy()}
        
        return(num_round_best)
        
        
    def gridsearch_cv(self, test_param, cv=5, n_jobs=4): 
        
        gsearch = GridSearchCV(estimator = self.model, param_grid=test_param, 
                               scoring = scoring_dict[self.__score_func], n_jobs= n_jobs, iid=False, cv=cv,
                               return_train_score=True) 
        gsearch.fit(self.X_train, np.ravel(self.y_train)) 
        dfcv_results = pd.DataFrame(gsearch.cv_results_)
        dfcv_simple = dfcv_results[['params','mean_train_score','mean_test_score']].copy()
        
        # 在score_gap < score_gap_limit条件下找到test_score最大时的n_estimators
        dfcv_simple['score_gap'] = dfcv_simple['mean_train_score'] - dfcv_simple['mean_test_score']
        dfcv_simple_filter = dfcv_simple.query('score_gap < {}'.format(self.score_gap_limit)).copy()
        if len(dfcv_simple_filter)<1:
            dfcv_simple_filter = pd.DataFrame(dfcv_simple.loc[dfcv_simple['score_gap'].idxmin(),:]).T
        best_id = dfcv_simple_filter['mean_test_score'].astype('f4').idxmax()
        best_params = dfcv_simple.loc[best_id,'params'].copy()
        best_score = dfcv_simple.loc[best_id,'mean_test_score']
        
        print('CV results: ')
        print(pretty_dataframe(dfcv_simple))
        print('Best params this step: ')
        print(best_params) 
        print('Best score this step: ')
        print(best_score) 
        
        # 计算更新参数后的得分
        self.params_dict.update(best_params)
        self.model = self.model.set_params(**self.params_dict)
        train_score,validate_score,score_gap = dfcv_simple.loc[best_id,['mean_train_score','mean_test_score','score_gap']]
        test_score = score_dict[self.__score_func](np.ravel(self.y_test),gsearch.predict_proba(self.X_test)[:,1])
        
        #if __DEBUG__: pdb.set_trace()##********************调试断点***********************##
        
        # 录入得分和参数
        i = len(self.dfscore)
        dic_score = {'model_id':i,'train_score':train_score,'validate_score':validate_score,
                     'score_gap':score_gap,'test_score':test_score}
        dic_score.update(self.params_dict)
        self.dfscore.loc[i,:] = deepcopy(dic_score)
        self.dfparams.loc[i] = {'model_id':i,'params_dict':self.params_dict.copy()}
        
        return(best_params)
        
        
        
        
        
