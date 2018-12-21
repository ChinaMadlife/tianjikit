# -*- coding:utf-8 -*-
#!/usr/bin/python2.7
###########################################################
#update_dt:2018-12-17
#author:liangyun
#usage: 超参数调试，只针对xgboost
###########################################################
from __future__ import print_function 
import sys,os,json,datetime
import numpy as np 
import pandas as pd 
from scipy import stats
import xgboost as xgb

# 初始化参数
params_dict = dict()

# 以下为待调整参数
# booster参数
params_dict['learning_rate'] = 0.1       # 学习率，初始值为 0.1，通常越小越好。
params_dict['n_estimators'] = 60         # 加法模型树的数量，初始值为50。

# tree参数
params_dict['max_depth'] = 3              # 树的深度，通常取值在[3,10]之间，初始值常取[3,6]之间
params_dict['min_child_weight']= 30       # 最小叶子节点样本权重和，越大模型越保守。
params_dict['gamma']= 0.0                   # 节点分裂所需的最小损失函数下降值，越大模型越保守。
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

# 定义ks评分指标,供xgboost.train函数的feval调用
def ks_feval(preds,xgbtrain):
    labels = xgbtrain.get_label()
    assert len(preds) == len(labels)
    df = pd.DataFrame(data = np.array([preds,labels]).T,columns = ['preds','label'])
    df_0,df_1 = df[df['label']<0.5],df[df['label']>=0.5]
    ks,ks_pvalue = stats.ks_2samp(df_0['preds'].values,df_1['preds'].values)
    return 'ks',ks

# 美化dataframe输出
from prettytable import PrettyTable
def pretty_dataframe(df):
    table = PrettyTable([''] + list(df.columns))
    for row in df.itertuples():
        table.add_row(row)
    return table

# 自定义分层KFold交叉验证
def stratified_kfold(data,label,nfolds = 5): 
    
    label = np.array(label)
    assert len(data) == len(label), 'the length of data and label not match!'
    assert set(label) == {0,1}, 'label can only be 0 or 1!'
    index = np.arange(len(label))
    index_0 = index[label<0.5].copy()
    index_1 = index[label>0.5].copy()
    np.random.shuffle(index_0)
    np.random.shuffle(index_1)
    split_points_0 = (len(index_0) * np.arange(1,nfolds))//nfolds
    split_points_1 = (len(index_1) * np.arange(1,nfolds))//nfolds
    split_index_0_list = np.split(index_0,split_points_0)
    split_index_1_list = np.split(index_1,split_points_1)
    split_index_list = [np.concatenate((x,y)) for x,y in 
                     zip(split_index_0_list,split_index_1_list)]
    result = [(np.setdiff1d(index,x),x) for x in split_index_list] 
    return result

# 训练xgb模型
def train_xgb(params_dict,dtrain,dvalid = None,dtest = None,verbose_eval = 10):
    
    result = {}
    watchlist = [x for x in [(dtrain, 'train'),(dvalid,'valid'),(dtest,'test')] if x[0] is not None]
    datasets = [x[1] for x in watchlist]
    
    bst = xgb.train(params = params_dict, dtrain = dtrain, 
                    num_boost_round = params_dict.get('n_estimators',100), 
                    feval = ks_feval,verbose_eval= verbose_eval,
                    evals = watchlist,
                    evals_result = result)
    dfresult = pd.DataFrame({(dataset+'_'+feval): result[dataset][feval] 
               for dataset in datasets for feval in ('auc','ks')})
    
    return bst,dfresult

# 构造参数网格
params_items = []
def params_grid(params):  
    
    global params_items
    params_items = [[(k,v) for v in values]  for k,values in params.items()]    
    itemstr = '('+','.join(['p%d'%i  for i in  range(len(params_items))]) + ',)' 
    forstr = ' '.join(['for p%d in params_items[%d]'%(i,i) for i in range(len(params_items))])
    items_grid = '[' + itemstr + ' ' + forstr + ']'
    
    result = [dict(x) for x in eval(items_grid)]   
    return(result)

# 调参主类
class Tunning(object):
    """ 
    Examples:
    --------
    from __future__ import print_function
    import numpy as np
    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from tianjikit.tunning import Tunning
    
    data,label = datasets.make_classification(n_samples= 10000, n_features=20, n_informative= 6 ,
                 n_classes=2, n_clusters_per_class=10,random_state=0)
    dfdata = pd.DataFrame(data,columns = [u'f'+str(i) for i in range(data.shape[1])])
    dfdata['label'] = label
    dftrain,dftest = train_test_split(dfdata)
    
    # 构造初始化参数
    params_dict = dict()
    # 以下为待调整参数
    # booster参数
    params_dict['learning_rate'] = 0.1        # 学习率，初始值为 0.1，通常越小越好。
    params_dict['n_estimators'] = 60          # 加法模型树的数量，初始值为50，通常通过模型cv确认。
    # tree参数
    params_dict['max_depth'] = 3              # 树的深度，通常取值在[3,10]之间，初始值常取[3,6]之间
    params_dict['min_child_weight']=10        # 最小叶子节点样本权重和，越大模型越保守。
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
    params_dict['scale_pos_weight'] = 1       #不平衡样本时设定为正值可以使算法更快收敛。
    params_dict['seed'] = 0
    
    # step0: 初始化
    tune = Tunning(dftrain,dftest,score_func = 'ks',score_gap_limit = 0.05,params_dict=params_dict,n_jobs=4)
    
    # step1: tune n_estimators for relatively high learning_rate
    params_test1 = {'learning_rate': [0.1],'n_estimators':[50]} 
    tune.gridsearch_cv(params_test1,cv = 5,verbose_eval = 10)
    
    # step2：tune max_depth & min_child_weight 
    params_test2 = { 'max_depth': [3], 'min_child_weight': [50,100,200] } 
    tune.gridsearch_cv(params_test2,cv = 5,verbose_eval = 10)
    
    
    # step3：tune gamma
    params_test3 = {'gamma': [0.1,0.5,1]}
    tune.gridsearch_cv(params_test3,cv = 5,verbose_eval = 10)
    
    
    # step4：tune subsample & colsample_bytree 
    params_test4 = { 'subsample': [0.9,1.0],'colsample_bytree': [1.0] } 
    tune.gridsearch_cv(params_test4,cv = 5,verbose_eval = 10)
    
    
    # step5: tune reg_alpha 
    params_test5 = { 'reg_alpha': [0.1,1] } 
    tune.gridsearch_cv(params_test5,cv = 5,verbose_eval = 10)
   
    
    # step6: tune reg_lambda 
    params_test6 = { 'reg_lambda': [0,0.1] }
    tune.gridsearch_cv(params_test6,cv = 5,verbose_eval = 10)
    
    
    # step7: lower learning_rate and rise n_estimators
    params_test7 = { 'learning_rate':[0.08,0.09], 'n_estimators':[100]}
    tune.gridsearch_cv(params_test7,cv = 5)
    
    # step8: train model with tuned parameters and fully train dataset.
    bst,dfimportance = tune.train_best()
    bst.save_model('./bst.model')
    dfimportance.to_csv('./dfimportance.csv',sep = '\t')
    
    """
    
    def __init__(self,dftrain,dftest,score_func = 'ks',score_gap_limit = 0.05,params_dict = params_dict,n_jobs = 4):
        
        # 校验是否有label列
        assert 'label' in dftrain.columns, 'illegal input,there should be a  "label" column in dftrain!'
        
        # 校验label列的合法性
        assert set(dftrain['label']) == {0,1},'illegal label values,label can only be 0 or 1!'

         # 去掉['phone','id','idcard','id_card','loan_dt','name','id_map']等非特征列
        for  col in {'phone','id','unique_id','uniq_id','idcard','id-card','id_card','name','loan_dt','idmap','id_map','id-map'}:
            if col in dftrain.columns:
                dftrain = dftrain.drop(col,axis = 1)
                if len(dftest):dftest = dftest.drop(col,axis = 1)
                    
        # 校验是否存在非数值列 
        try:
            assert not np.dtype('O') in dftrain.dtypes.values
        except:
            object_cols = dftrain.columns[dftrain.dtypes == np.object].tolist()
            print('removed feature columns not numerical: %s'%(','.join(map(str,object_cols))),file = sys.stderr)
            dftrain = dftrain.drop(object_cols,axis = 1)
            if len(dftest):dftest = dftest.drop(object_cols,axis = 1)
        
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('\n================================================================================ %s\n'%nowtime)
        print('train set size: %d'%len(dftrain))
        print('test set size: %d'%len(dftest))
        print('feature number: %s'%str(dftrain.shape[1]))
        print('score func: %s'%score_func)
        print('score gap limit: %s'%str(score_gap_limit))
        print('n_jobs: %d'%n_jobs)
        
        # 分割feature和label
        X_train = dftrain.drop(['label'],axis = 1)
        y_train = dftrain['label']
        X_test = dftest.drop(['label'],axis = 1)
        y_test = dftest['label'] 
        
        X_train.index = range(len(X_train))
        y_train.index = range(len(X_train))
        X_test.index = range(len(X_test)) 
        y_test.index = range(len(X_test))
        
        # 预处理后的训练和验证集
        self.X_train,self.y_train = X_train,y_train
        self.X_test,self.y_test  = X_test,y_test
        
        # self.params_dict存储当前参数，self.dfscores存储历史得分记录，self.dfparams存储历史参数记录,
        # self.dfmerge是dfscores和dfparams的合并
        self.params_dict = params_dict.copy()
        self.params_dict['nthread'] = n_jobs
        
        self.dfmerge = pd.DataFrame(columns = ['model_id','train_score','validate_score','score_gap','test_score'] + 
           ['learning_rate','n_estimators','max_depth','min_child_weight','gamma','subsample','colsample_bytree','reg_alpha','reg_lambda'])
        self.dfscores = pd.DataFrame(columns = ['model_id','train_score','validate_score','score_gap','test_score'])
        self.dfparams = pd.DataFrame(columns = ['model_id','learning_rate','n_estimators','max_depth','min_child_weight',
                                       'gamma','subsample','colsample_bytree','reg_alpha','reg_lambda'])
        
        self.score_func = score_func
        self.score_gap_limit = score_gap_limit
        
    def model_cv(self,params_dict,cv = 5,verbose_eval = 10):
        
        kfold_indexes = stratified_kfold(self.X_train,self.y_train,nfolds = cv)
        dfresults_list = [np.nan]*cv
        dtest = xgb.DMatrix(self.X_test,self.y_test)
        train_score = 'train_' + self.score_func 
        valid_score = 'valid_' + self.score_func
        test_score = 'test_' + self.score_func
        for i in range(cv):
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('\n================================================================================ %s\n'%nowtime)
            print('k = %d'%(i+1))
            train_index,valid_index = kfold_indexes[i]
            dtrain = xgb.DMatrix(self.X_train.iloc[train_index,:],self.y_train.iloc[train_index])
            dvalid = xgb.DMatrix(self.X_train.iloc[valid_index,:],self.y_train.iloc[valid_index])
            bst,dfresults_list[i] = train_xgb(params_dict,dtrain,dvalid,dtest,verbose_eval)
            dfresults_list[i]['train_valid_gap'] =  dfresults_list[i][train_score] - dfresults_list[i][valid_score]
            
        def npmean(*d):
            s = d[0]
            for di in d[1:]:
                s = s + di
            s = s/float(len(d))
            return(s)    
        
        dfmean = npmean(*dfresults_list)
        
        dfmean['n_estimators'] = np.arange(1,len(dfmean)+1)
        dfans = dfmean.query('train_valid_gap < {}'.format(self.score_gap_limit))
        if len(dfans) <1: 
            dfans = dfmean.iloc[[np.argmin(dfmean['train_valid_gap'].values)],:]
        
        dic = dict(dfans.iloc[np.argmax(dfans[valid_score].values),:])
        dic['n_estimators'] = int(dic['n_estimators'])
        ans_dict = params_dict.copy()
        ans_dict.update({'n_estimators':dic['n_estimators'],'train_score':dic[train_score],
                         'validate_score':dic[valid_score],'test_score':dic[test_score],
                         'score_gap':dic['train_valid_gap']})
        return ans_dict
    
    def gridsearch_cv(self,params_test,cv = 5,verbose_eval = 10):
        
        test_params_grid = params_grid(params_test)
        params_dict = self.params_dict.copy()
        
        for d in test_params_grid:
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('\n================================================================================ %s\n'%nowtime)
            print(d)
            params_dict.update(d)
            ans_dict = self.model_cv(params_dict,cv,verbose_eval)
            dic_merge = ans_dict.copy()
            m = len(self.dfmerge)
            dic_merge.update({'model_id':m})
            self.dfmerge.loc[m,:] = dic_merge 
            self.dfscores.loc[m,:] = dic_merge
            self.dfparams.loc[m,:] = dic_merge
            
        df_filter = self.dfscores.query('score_gap < {}'.format(self.score_gap_limit))
        dfscore_best = df_filter.iloc[[np.argmax(df_filter['validate_score'].values)],:]
        dfparams_best = self.dfparams.query('model_id == {}'.format(dfscore_best['model_id'].values[0]))
        
        # 更新最优参数至当前参数,除了n_estimators
        best_params = dict(dfparams_best.iloc[0,:])
        best_params.pop('model_id')
        best_params.pop('n_estimators')    # 最优的n_estimators 不逐级传递，依赖model_cv每次确认。  
        self.params_dict.update(best_params)
        
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('\n================================================================================ %s\n'%nowtime)
        print('Tested params:')
        print(pretty_dataframe(self.dfparams))
        print('Tested scores:')
        print(pretty_dataframe(self.dfscores))
        print('Best params so far:')
        print(pretty_dataframe(dfparams_best)) 
        print('Best score so far:')
        print(pretty_dataframe(dfscore_best)) 
        
        return(dfscore_best)
    
    def train_best(self,verbose_eval = 10):
        
        dtrain = xgb.DMatrix(self.X_train,self.y_train)
        dtest = xgb.DMatrix(self.X_test,self.y_test)
        
        # 寻找历史参数序列中最优参数
        df_filter = self.dfscores.query('score_gap < {}'.format(self.score_gap_limit))
        dfscore_best = df_filter.iloc[[np.argmax(df_filter['validate_score'].values)],:]
        dfparams_best = self.dfparams.query('model_id == {}'.format(dfscore_best['model_id'].values[0]))
        
        # 更新全部最优参数至当前参数包括n_estimators
        best_params = dict(dfparams_best.iloc[0,:])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params.pop('model_id')   
        
        self.params_dict.update(best_params) 
        
        bst,_ = train_xgb(self.params_dict,dtrain,None,dtest,verbose_eval)
        dfimportance = pd.DataFrame({'feature':bst.get_score().keys(),'importance':bst.get_score().values()})
        try:
            dfimportance = dfimportance.sort_values('importance',ascending=False)
        except AttributeError as err:
            dfimportance = dfimportance.sort('importance',ascending = False)
        return(bst, dfimportance)
  
        
        
        