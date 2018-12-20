#-*-coding:utf-8-*-
#!/usr/bin/python2.7
##################################################
#update_dt:2018-12-19
#author：liangyun
#usage: train xgboost model
##################################################

from __future__ import print_function
import datetime,sys,os
import ks,outliers,dropfeature
import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb

# 配置xgboost模型参数
params_dict = dict()

# 以下为待调整参数
# booster参数
params_dict['learning_rate'] = 0.1       # 学习率，初始值为 0.1，通常越小越好。
params_dict['n_estimators'] = 60         # 加法模型树的数量，初始值为50。

# tree参数
params_dict['max_depth'] = 3              # 树的深度，通常取值在[3,10]之间，初始值常取[3,6]之间
params_dict['min_child_weight']= 30       # 最小叶子节点样本权重和，越大模型越保守。
params_dict['gamma']= 0                   # 节点分裂所需的最小损失函数下降值，越大模型越保守。
params_dict['subsample']= 0.8             # 横向采样，样本采样比例，通常取值在 [0.5，1]之间 
params_dict['colsample_bytree'] = 1.0     # 纵向采样，特征采样比例，通常取值在 [0.5，1]之间 

# regulazation参数 
# Omega(f) = gamma*T + reg_alpha* sum(abs(wj)) + reg_lambda* sum(wj**2)  

params_dict['reg_alpha'] = 0.0              #L1 正则化项的权重系数，越大模型越保守，通常取值在[0,1]之间。
params_dict['reg_lambda'] = 1.0             #L2 正则化项的权重系数，越大模型越保守，通常取值在[1,100]之间。

# 以下参数通常不需要调整
params_dict['objective'] = 'binary:logistic'
params_dict['tree_method'] = 'hist'       # 构建树的策略,可以是auto, exact, approx, hist
params_dict['eval_metric'] =  'auc'
params_dict['silent'] = 1
params_dict['nthread'] = 2
params_dict['scale_pos_weight'] = 1        #不平衡样本时设定为正值可以使算法更快收敛。
params_dict['seed'] = 0

# 定义ks评分指标,供xgboost.train函数的feval调用
def ks_feval(preds,xgbtrain):
    label = xgbtrain.get_label()
    assert len(preds) == len(label)
    df = pd.DataFrame(data = np.array([preds,label]).T,columns = ['preds','label'])
    df_0,df_1 = df[df['label']<0.5],df[df['label']>=0.5]
    ks,ks_pvalue = stats.ks_2samp(df_0['preds'].values,df_1['preds'].values)
    return 'ks',ks

def auc(labels, preds):
    """
    auc值的大小可以理解为: 随机抽一个正样本和一个负样本，正样本预测值比负样本大的概率
　　先排序，然后统计有多少正负样本对满足：正样本预测值>负样本预测值, 再除以总的正负样本对个数
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    total_pair = n_pos * n_neg
 
    labels_preds = zip(labels, preds)
    labels_preds = sorted(labels_preds, key=lambda x: x[1])
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(len(labels_preds)):
        if labels_preds[i][0] == 1:
            satisfied_pair += accumulated_neg
        else:
            accumulated_neg += 1
    return satisfied_pair / float(total_pair)
    
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

class TrainXgboost(object):
    """
    Examples
    ---------
    # 准备训练数据
    import numpy as np
    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    data,label = datasets.make_classification(n_samples= 10000, n_features=20,n_classes=2, random_state=0)
    dfdata = pd.DataFrame(data,columns = ['feature'+str(i) for i in range(data.shape[1])])
    dfdata['label'] = label
    dftrain,dftest = train_test_split(dfdata)
    dftrain,dftest = dftrain.copy(),dftest.copy()
    dftrain.index,dftest.index  = range(len(dftrain)),range(len(dftest))
    dftrain.loc[0,['feature0','feature1','feature2']] = np.nan #构造若干缺失值
    
    
    # 配置xgboost模型参数
    params_dict = dict()

    # 以下为待调整参数
    # booster参数
    params_dict['learning_rate'] = 0.1       # 学习率，初始值为 0.1，通常越小越好。
    params_dict['n_estimators'] = 60         # 加法模型树的数量，初始值为50。
    
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
    params_dict['nthread'] = 2
    params_dict['scale_pos_weight'] = 1        #不平衡样本时设定为正值可以使算法更快收敛。
    params_dict['seed'] = 0
    
    # 训练xgboost模型
    from tianjikit.trainxgboost import TrainXgboost
    model = TrainXgboost(dftrain = dftrain,dftest = dftest, coverage_th=0, ks_th=0,
            outliers_th=None, selected_features=None)
    bst = model.train(cv=5, model_idx=1,params_dict = params_dict,n_jobs = 4, verbose_eval = 10) 
    model.test(bst,dftest)
    dfimportance = model.dfimportance
    # bst.save_model('./bst.model')
    
    """
    
    def __init__(self,dftrain,dftest = '',coverage_th = 0, ks_th = 0, outliers_th = None,
                  selected_features = None):
        
        # 校验是否有label列
        assert 'label' in dftrain.columns, 'illegal input,there should be a  "label" column in dftrain!'
        
        # 校验label列的合法性
        assert set(dftrain['label']) == {0,1},'illegal label values,label can only be 0 or 1!'
        
        self.dftrain = dftrain
        self.dftest = dftest
        
        # 记录预处理参数信息
        self.coverage_th = coverage_th
        self.ks_th = ks_th
        self.outliers_th = outliers_th
        self.selected_features = selected_features
        
        X_train,y_train,X_test,y_test = self.preprocess_data(self.dftrain,self.dftest)
        
       
        # 预处理后的训练和验证集
        self.X_train,self.y_train = X_train,y_train
        self.X_test,self.y_test  = X_test,y_test
        
        self.dtrain = xgb.DMatrix(self.X_train, self.y_train['label'])
        self.dtest = xgb.DMatrix(self.X_test, self.y_test['label'])
        
        # 特征重要性
        self.dfimportance = None 
        
        # 报告信息
        self.report_info = ''
        
        
    def preprocess_data(self,dftrain,dftest):
        
        # 输出预处理提示信息
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('\n================================================================================ %s\n'%nowtime)
        print('start data preprocessing ...\n')
        print('train set size:  {}'.format(len(dftrain)))
        print('test set size:  {}'.format(len(dftest)))
        print('coverage threshold:  {}'.format(self.coverage_th))
        print('outlier threshold:  {}'.format(self.outliers_th))
        print('ks threshold:  {}'.format(self.ks_th))

       
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
        
        # 如果selected_features 不为空，则进行特征筛选
        if self.selected_features:
            remained_cols = [col for col in dftrain.columns if col in self.selected_features + ['label']]
            dftrain = dftrain[remained_cols]
            if len(dftest):dftest = dftest[remained_cols]
                    
        # 分割feature和label
        X_train = dftrain.drop(['label'],axis = 1)
        y_train = dftrain[['label']]
        X_test = dftest.drop(['label'],axis = 1) if len(dftest) else ''
        y_test = dftest[['label']] if len(dftest) else ''
        
        print('original feature number:  {}'.format(X_train.shape[1]))
        
        # drop_outliers()
        if self.outliers_th:
            for col in X_train.columns:
                X_train[col] = outliers.drop_outliers(X_train[col].values, X_train[col].values, alpha = self.outliers_th) 
                if len(dftest): X_test[col] = outliers.drop_outliers(X_train[col].values,X_test[col].values, alpha = self.outliers_th)  
                    
        # drop_feature()
        X_train, X_test = dropfeature.drop_feature(X_train,y_train,X_test,coverage_threshold = self.coverage_th, 
                          ks_threshold = self.ks_th) 
        
        print('feature number remain after dropfeature:  {}'.format(X_train.shape[1]))
        
        # 重置index  
        X_train.index = range(len(X_train))
        y_train.index = range(len(X_train))
        X_test.index = range(len(X_test)) 
        y_test.index = range(len(X_test))
        
        return(X_train,y_train,X_test,y_test)
    
    
    def train(self,cv = 5,model_idx = 5,
              params_dict = params_dict,
              n_jobs = 4,verbose_eval = 20):
        
        info = "start train xgboost model ..."
        print(info)
        self.report_info = self.report_info + info + '\n'
        
        params_dict_copy = params_dict.copy()
        params_dict_copy.update({'nthread':n_jobs})
            
        if cv:
            
            k,ks_mean_train,auc_mean_train,ks_mean_validate,auc_mean_validate = 0,0,0,0,0

            models = {}

            for train_index,validate_index in stratified_kfold(self.X_train,np.ravel(self.y_train),nfolds = cv):

                k = k + 1
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print('\n================================================================================ %s\n'%nowtime)
                info = 'k = {}'.format(k)
                print(info)
                self.report_info = self.report_info + info + '\n'
                
                X_train_k,y_train_k = self.X_train.iloc[train_index,:],self.y_train.iloc[train_index,:]
                X_validate_k,y_validate_k = self.X_train.iloc[validate_index,:],self.y_train.iloc[validate_index,:]
              
                dtrain_k = xgb.DMatrix(X_train_k,y_train_k['label'])
                dvalid_k = xgb.DMatrix(X_validate_k,y_validate_k['label'])
                
                bst,_ = train_xgb(params_dict_copy,dtrain_k,dvalid_k,None,verbose_eval)
                predict_train_k = bst.predict(dtrain_k)
                predict_validate_k = bst.predict(dvalid_k)

                dfks_train = ks.ks_analysis(predict_train_k,dtrain_k.get_label())
                dfks_validate = ks.ks_analysis(predict_validate_k,dvalid_k.get_label())

                ks_train,ks_validate = max(dfks_train['ks_value']),max(dfks_validate['ks_value'])
                
                auc_train = auc(dtrain_k.get_label(),predict_train_k)
                auc_validate = auc(dvalid_k.get_label(), predict_validate_k)
        
                ks_mean_train = ks_mean_train + ks_train
                auc_mean_train = auc_mean_train + auc_train
                ks_mean_validate = ks_mean_validate + ks_validate
                auc_mean_validate = auc_mean_validate + auc_validate

                info = '\ntrain: ks = {} \t auc = {} '.format(ks_train,auc_train)
                prettyks = ks.print_ks(predict_train_k,dtrain_k.get_label())
                info = info + '\n' + str(prettyks) + '\n'
                info = info + '\nvalidate: ks = {} \t auc = {}'.format(ks_validate,auc_validate) + '\n'
                prettyks = ks.print_ks(predict_validate_k,dvalid_k.get_label())
                info = info + str(prettyks) + '\n'
                print(info)
                self.report_info = self.report_info + info
                
                models[k] = bst

            ks_mean_train = ks_mean_train/float(k)
            auc_mean_train = auc_mean_train/float(k)
            ks_mean_validate = ks_mean_validate/float(k)
            auc_mean_validate = auc_mean_validate/float(k)
            
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            info = '\n================================================================================ %s\n'%nowtime
            info = info + 'train : ks mean {:.5f} ; auc mean {:.5f}'.format(ks_mean_train, auc_mean_train) + '\n'
            info = info + 'validate : ks mean {:.5f} ; auc mean {:.5f}'.format(ks_mean_validate, auc_mean_validate) + '\n'
            print(info)
            self.report_info = self.report_info + info

            bst = models[model_idx]
            
        # 处理 cv = 0 或 cv = None时无需交叉验证逻辑
        else:
            
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            info = '\n================================================================================ %s\n'%nowtime
            print(info)
            self.report_info = self.report_info + info
            
            bst,_ = train_xgb(params_dict_copy,self.dtrain,None,None,verbose_eval)
            predict_train = bst.predict(self.dtrain)
            dfks_train = ks.ks_analysis(predict_train,self.y_train.values)
            ks_train = max(dfks_train['ks_value'])   
            auc_train = auc(self.dtrain.get_label(),predict_train)
            
            info = '\ntrain: ks = {} \t auc = {} '.format(ks_train,auc_train) + '\n'
            prettyks = ks.print_ks(predict_train,self.y_train.values)
            info = info + str(prettyks) + '\n'
            print(info)
            self.report_info = self.report_info + info
            
        # 计算特征重要性
        feature_scores = bst.get_score()
        dfimportance = pd.DataFrame({'feature':feature_scores.keys(),'importance':feature_scores.values()})
        try:
            dfimportance = dfimportance.sort_values('importance',ascending=False)
        except AttributeError as err:
            dfimportance = dfimportance.sort('importance',ascending = False)

        dfimportance.index = range(len(dfimportance))
        
        self.dfimportance = dfimportance
        
        return(bst)
        
    def test(self,bst,dftest = pd.DataFrame()):
        
        info = "\nstart test xgboost model ... \n"
        print(info)
        self.report_info = self.report_info + info + '\n'
        
        # 若传入新的dftest，则需要再次做数据预处理
        if len(dftest)>0:
            
            print('preprocessing test data...')
            
            # 禁止数据预处理期间打印输出
            stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            X_train,y_train,X_test,y_test = self.preprocess_data(self.dftrain,dftest)
            
            # 恢复打印输出
            sys.stdout = stdout
            
            # 预处理后的测试集
            self.X_test,self.y_test  = X_test,y_test
            self.dtest = xgb.DMatrix(self.X_test, self.y_test['label'])
            
        y_test_hat = bst.predict(self.dtest)
        dfks_test = ks.ks_analysis(y_test_hat,np.ravel(self.y_test))
        ks_test = max(dfks_test['ks_value'])
        auc_test = auc(np.ravel(self.y_test),y_test_hat)
        
        info = 'test: ks = {} \t auc = {} '.format(ks_test,auc_test) + '\n'
        prettyks = ks.print_ks(y_test_hat,np.ravel(self.y_test))
        info = info + str(prettyks) + '\n'
        print(info)
        self.report_info = self.report_info + info + '\n'
