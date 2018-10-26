#coding=utf-8
#!/usr/bin/python2.7

##################################################
#update_dt:2018-10-08
#author：liangyun
#usage: run model
##################################################
from __future__ import print_function
import datetime,sys
import ks,outliers,dropfeature,fillnan,scalefeature
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn import linear_model
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


class RunModel(object):
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
    
    # 训练逻辑回归模型
    from tianjikit.runmodel import RunModel
    model = RunModel(dftrain = dftrain,dftest = dftest,coverage_th=0.1, ks_th=0, chi2_th=0, 
            outliers_th=None, fillna_method='most', scale_method=None,selected_features=None)
    lr = model.train_lr(cv=None, model_idx=5)
    model.test(lr)
    dfimportance = model.dfimportances['lr']
    

    # 训练随机森林模型
    from tianjikit.runmodel import RunModel
    model = RunModel(dftrain = dftrain,dftest = dftest,coverage_th=0.1, ks_th=0, chi2_th=0, 
            outliers_th=None, fillna_method='most', scale_method=None,selected_features=None)
    rf = model.train_rf(cv=5, model_idx=5,
          n_estimators=100, max_depth=10, min_samples_split=2,
          min_samples_leaf=1, min_weight_fraction_leaf=0.0,
          max_features='auto', max_leaf_nodes=None, n_jobs = 4)
    model.test(rf)
    dfimportance = model.dfimportances['rf']
    

    # 训练GBDT模型
    from tianjikit.runmodel import RunModel
    model = RunModel(dftrain = dftrain,dftest = dftest,coverage_th=0.1, ks_th=0, chi2_th=0, 
            outliers_th=None, fillna_method='most', scale_method=None,selected_features=None)
    gbdt = model.train_gbdt(cv=5, model_idx=5,
           learning_rate=0.01, n_estimators=1000, max_depth= 3, min_samples_split= 50, 
           min_samples_leaf= 5, subsample=0.7, max_features='sqrt',random_state= 0) 
    model.test(gbdt)
    dfimportance = model.dfimportances['gbdt']
    

    # 训练XGBOOST模型
    from tianjikit.runmodel import RunModel
    model = RunModel(dftrain = dftrain,dftest = dftest,coverage_th=0.1, ks_th=0, chi2_th=0, 
            outliers_th=None, fillna_method= None, scale_method= None,selected_features=None)
    xgb = model.train_xgb(cv=5, model_idx=5,
          learning_rate=0.1,n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, 
          subsample=0.8,colsample_bytree=0.8,scale_pos_weight=1, n_jobs=4, seed=10) 
    model.test(xgb)
    dfimportance = model.dfimportances['xgb']
    
    
    # 训练神经网络模型
    from tianjikit.runmodel import RunModel
    model = RunModel(dftrain = dftrain,dftest = dftest,coverage_th=0.1, ks_th=0, chi2_th=0, 
            outliers_th=None, fillna_method='most', scale_method= None, selected_features=None)
    nn = model.train_nn( cv = 5, model_idx = 1,
         hidden_layer_sizes=(100,20), activation='relu', alpha=0.0001, 
         learning_rate='constant', learning_rate_init=0.001, max_iter=200,tol=0.0001, 
         early_stopping=False, validation_fraction=0.1, warm_start=False, random_state = None)
    model.test(nn)

    """
    
    def __init__(self,dftrain,dftest = '',coverage_th = 0.1, ks_th = 0, chi2_th = 0, outliers_th = None,
                 fillna_method = 'infer',scale_method = 'MinMax', selected_features = None):
        
        # 输出预处理提示信息
        print('================================================================================')
        print('start data preprocessing ...\n')
        print('train set size:  {}'.format(len(dftrain)))
        print('test set size:  {}'.format(len(dftest)))
        print('coverage threshold:  {}'.format(coverage_th))
        print('outlier threshold:  {}'.format(outliers_th))
        print('ks threshold:  {}'.format(ks_th))
        print('chi2 threshold:  {}'.format(chi2_th))
        print('fillna method:  {}'.format(fillna_method))
        print('scale method:  {}'.format(scale_method))
        
       
        # 去掉['phone','id','idcard','id_card','loan_dt','name','id_map']等非特征列
        for  col in ['name','phone','id','idcard','id_card','loan_dt','id_map']:
            if col in dftrain.columns:
                dftrain = dftrain.drop([col],axis = 1)
                if len(dftest):dftest = dftest.drop([col],axis = 1)
        
        # 如果selected_features 不为空，则进行特征筛选
        if selected_features:
            remained_cols = [col for col in dftrain.columns if col in selected_features + ['label']]
            dftrain = dftrain[remained_cols]
            if len(dftest):dftest = dftest[remained_cols]
                
        # 制作特征名称映射表，复杂的特征名可能导致xgboost出错
        self.__feature_dict = {'feature'+ str(i): name for i,name in enumerate(dftrain.columns.drop('label'))}
        self.__inverse_feature_dict = dict(zip(self.__feature_dict.values(),self.__feature_dict.keys()))
        dftrain.columns = [self.__inverse_feature_dict.get(x,x) for x in dftrain.columns]
        dftest.columns = [self.__inverse_feature_dict.get(x,x) for x in dftest.columns]
                    
        # 分割feature和label
        X_train = dftrain.drop(['label'],axis = 1)
        y_train = dftrain[['label']]
        X_test = dftest.drop(['label'],axis = 1) if len(dftest) else ''
        y_test = dftest[['label']] if len(dftest) else ''
        
        print('original feature number:  {}'.format(X_train.shape[1]))
        
        # drop_outliers()
        if outliers_th:
            for col in X_train.columns:
                X_train[col] = outliers.drop_outliers(X_train[col].values, X_train[col].values, alpha = outliers_th) 
                if len(dftest): X_test[col] = outliers.drop_outliers(X_train[col].values,X_test[col].values, alpha = outliers_th)  
                    
        # dropfeature()
        X_train, X_test = dropfeature.drop_feature(X_train,y_train,X_test,coverage_threshold = coverage_th, 
                          ks_threshold = ks_th, chi2_threshold = chi2_th) 
        
        print('feature number remain after dropfeature:  {}'.format(X_train.shape[1]))
        
        
        # fillnan()
        if fillna_method:
            X_train, X_test = fillnan.fill_nan(X_train,y_train,X_test,method = fillna_method)
        
        print('feature number increased to after fill_na:  {}'.format(X_train.shape[1]))
        print('================================================================================')
        
        # scalefeature()
        if scale_method:
            X_train, X_test  = scalefeature.scale_feature(X_train,X_test,method = scale_method)
        
        
        # 预处理后的测试和验证集
        self.X_train,self.y_train = X_train,y_train
        self.X_test,self.y_test  = X_test,y_test
        
        # 特征重要性
        self.dfimportances = {} 
        
        # 报告信息
        self.report_info = ''
        
        
    def train_lr(self, cv = 5, model_idx = 1):
        
        lr = linear_model.LogisticRegressionCV()

        info = "start train logistic model ..."
        print(info)
        self.report_info = self.report_info + info + '\n'
        
        clf = self.train(lr,cv = cv,model_idx = model_idx)  
        
        # 保存特征系数
        cols = self.X_train.columns
        dfcoef = pd.DataFrame(clf.coef_.reshape(-1),columns = ['coef'])
        dfcoef.insert(0,'feature',cols)
        dfcoef['importance'] = np.abs(dfcoef['coef'])
        try:
            dfcoef = dfcoef.sort_values('importance',ascending= False)
        except AttributeError as err:
            dfcoef = dfcoef.sort('importance',ascending= False)
        dfcoef['feature'] = [self.__feature_dict.get(x,x) for x in dfcoef['feature']]
        self.dfimportances['lr'] = dfcoef
              
        return clf
    
    def train_rf(self, cv = 5, model_idx = 1,
        n_estimators=100, max_depth=10, min_samples_split=2,min_samples_leaf=1, min_weight_fraction_leaf=0.0,
        max_features='auto', max_leaf_nodes=None, n_jobs = 4,random_state = 0):
        
        rf = ensemble.RandomForestClassifier(n_estimators = n_estimators,max_depth = max_depth,
                  min_samples_split = min_samples_split,min_samples_leaf = min_samples_leaf,
                  min_weight_fraction_leaf = min_weight_fraction_leaf, max_features = max_features,
                  max_leaf_nodes = max_leaf_nodes,n_jobs = n_jobs,random_state = random_state)
        
        
        info = "start train randomforest model ..."
        print(info)
        self.report_info = self.report_info + info + '\n'

        clf = self.train(rf,cv = cv,model_idx = model_idx) 
            
        # 计算特征重要性
        cols = self.X_train.columns
        dfimportance = pd.DataFrame(clf.feature_importances_.reshape(-1),columns = ['importance'])
        dfimportance.insert(0,'feature',cols)
        try:
            dfimportance = dfimportance.sort_values('importance',ascending= False)
        except AttributeError as err:
            dfimportance = dfimportance.sort('importance',ascending= False)
        dfimportance['feature'] = [self.__feature_dict.get(x,x) for x in dfimportance['feature']]
        self.dfimportances['rf'] = dfimportance
        
        return clf
    
    def train_gbdt(self, cv = 5, model_idx = 1, learning_rate=0.1, n_estimators=100,
                   max_depth= 3, min_samples_split= 2, min_samples_leaf= 1, 
                   subsample=0.85, max_features='sqrt',random_state= 0,**kv):
        
        gbdt = ensemble.GradientBoostingClassifier(learning_rate = learning_rate, n_estimators = n_estimators,
              max_depth = max_depth,min_samples_split = min_samples_split,min_samples_leaf = min_samples_leaf,
              subsample = subsample, max_features = max_features,random_state = random_state,**kv)
        
        
        info = "start train gbdt model ..."
        print(info)
        self.report_info = self.report_info + info + '\n'
            
        clf = self.train(gbdt,cv = cv,model_idx = model_idx) 
        
        # 计算特征重要性
        cols = self.X_train.columns
        dfimportance = pd.DataFrame(clf.feature_importances_.reshape(-1),columns = ['importance'])
        dfimportance.insert(0,'feature',cols)
        try:
            dfimportance = dfimportance.sort_values('importance',ascending= False)
        except AttributeError as err:
            dfimportance = dfimportance.sort('importance',ascending= False)  
        dfimportance['feature'] = [self.__feature_dict.get(x,x) for x in dfimportance['feature']]
        self.dfimportances['gbdt'] = dfimportance
        
        return clf
    
    def train_nn(self, cv = 5, model_idx = 1,
                 hidden_layer_sizes=(100,20), activation='relu', alpha=0.0001, 
                 learning_rate='constant', learning_rate_init=0.001, max_iter=200,tol=0.0001, 
                 early_stopping=False, validation_fraction=0.1, warm_start=False, random_state= 0):
        
        nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha, 
                  learning_rate=learning_rate, learning_rate_init=learning_rate_init, max_iter=max_iter,tol=tol, 
                  early_stopping=early_stopping, validation_fraction=validation_fraction, 
                  warm_start=warm_start, random_state= random_state)
        
        info = "start train neural network model ..."
        print(info)
        self.report_info = self.report_info + info + '\n'
        
        clf = self.train(nn,cv = cv,model_idx = model_idx) 
        
        return clf
    
    def train_xgb(self, cv = 5, model_idx = 1,    
        learning_rate=0.1,n_estimators=50, max_depth=5, min_child_weight=1,gamma=0,subsample=0.8,
        colsample_bytree=1, n_jobs=-1, scale_pos_weight=1, seed=10):
        
        xgb = XGBClassifier(learning_rate = learning_rate, n_estimators = n_estimators,max_depth = max_depth,
                      min_child_weight = min_child_weight,gamma = gamma,subsample = subsample,colsample_bytree = colsample_bytree,
                      n_jobs = n_jobs, scale_pos_weight = scale_pos_weight, seed = seed)
        
        info = "start train xgboost model ..."
        print(info)
        self.report_info = self.report_info + info + '\n'
        
        clf = self.train(xgb,cv = cv,model_idx = model_idx) 
        
        # 计算特征重要性
        cols = self.X_train.columns
        dfimportance = pd.DataFrame(clf.feature_importances_.reshape(-1),columns = ['importance'])
        dfimportance.insert(0,'feature',cols)
        try:
            dfimportance = dfimportance.sort_values('importance',ascending= False)
        except AttributeError as err:
            dfimportance = dfimportance.sort('importance',ascending= False) 
        dfimportance['feature'] = [self.__feature_dict.get(x,x) for x in dfimportance['feature']]
        self.dfimportances['xgb'] = dfimportance
        return clf
    
    def train(self,clf,cv = 5,model_idx = 5):
        
        if cv:
            skf = StratifiedKFold(n_splits = cv,shuffle=True)

            k,ks_mean_train,auc_mean_train,ks_mean_validate,auc_mean_validate = 0,0,0,0,0

            models = {}

            for train_index,validate_index in skf.split(self.X_train,np.ravel(self.y_train)):

                k = k + 1
                nowtime = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
                info = '\n{}: k = {}'.format(nowtime,k)
                print(info)
                self.report_info = self.report_info + info + '\n'

                X_train_k,y_train_k = self.X_train.iloc[train_index,:],self.y_train.iloc[train_index,:]
                X_validate_k,y_validate_k = self.X_train.iloc[validate_index,:],self.y_train.iloc[validate_index,:]
                
                clf.fit(X_train_k,np.ravel(y_train_k))
                predict_train_k = clf.predict_proba(X_train_k)[:,-1]
                predict_validate_k = clf.predict_proba(X_validate_k)[:,-1]

                dfks_train = ks.ks_analysis(predict_train_k,y_train_k.values)
                dfks_validate = ks.ks_analysis(predict_validate_k,y_validate_k.values)

                ks_train,ks_validate = max(dfks_train['ks_value']),max(dfks_validate['ks_value'])

                auc_validate = metrics.roc_auc_score(np.ravel(y_validate_k), predict_validate_k)
                auc_train = metrics.roc_auc_score(np.ravel(y_train_k),predict_train_k)

                ks_mean_train = ks_mean_train + ks_train
                auc_mean_train = auc_mean_train + auc_train
                ks_mean_validate = ks_mean_validate + ks_validate
                auc_mean_validate = auc_mean_validate + auc_validate

                info = '\ntrain: ks = {} \t auc = {} '.format(ks_train,auc_train)
                prettyks = ks.print_ks(predict_train_k,y_train_k.values)
                info = info + '\n' + str(prettyks) + '\n'
                info = info + '\nvalidate: ks = {} \t auc = {}'.format(ks_validate,auc_validate) + '\n'
                prettyks = ks.print_ks(predict_validate_k,y_validate_k.values)
                info = info + str(prettyks) + '\n'
                print(info)
                self.report_info = self.report_info + info
                

                models[k] = clf

            ks_mean_train = ks_mean_train/float(k)
            auc_mean_train = auc_mean_train/float(k)
            ks_mean_validate = ks_mean_validate/float(k)
            auc_mean_validate = auc_mean_validate/float(k)
            
            info = '\n================================================================================\n'
            info = info + 'train : ks mean {:.5f} ; auc mean {:.5f}'.format(ks_mean_train, auc_mean_train) + '\n'
            info = info + 'validate : ks mean {:.5f} ; auc mean {:.5f}'.format(ks_mean_validate, auc_mean_validate) + '\n'
            print(info)
            self.report_info = self.report_info + info

            clf = models[model_idx]
            
        # 处理 cv = 0 或 cv = None时无需交叉验证逻辑
        else:
            
            nowtime = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
            info = '\n{}:'.format(nowtime)
            print(info)
            self.report_info = self.report_info + info
            
            clf.fit(self.X_train,np.ravel(self.y_train))
            predict_train = clf.predict_proba(self.X_train)[:,-1]
            dfks_train = ks.ks_analysis(predict_train,self.y_train.values)
            ks_train = max(dfks_train['ks_value'])   
            auc_train = metrics.roc_auc_score(np.ravel(self.y_train),predict_train)
            
            info = '\ntrain: ks = {} \t auc = {} '.format(ks_train,auc_train) + '\n'
            prettyks = ks.print_ks(predict_train,self.y_train.values)
            info = info + prettyks + '\n'
            print(info)
            self.report_info = self.report_info + info
            
        return(clf)
        
    def test(self,clf):
        
        info = "\nstart test model ... \n"
        print(info)
        self.report_info = self.report_info + info + '\n'
        
        y_test_hat = clf.predict_proba(self.X_test)[:,-1]
        dfks_test = ks.ks_analysis(y_test_hat,np.ravel(self.y_test))
        ks_test = max(dfks_test['ks_value'])
        auc_test = metrics.roc_auc_score(np.ravel(self.y_test),y_test_hat)
        
        info = 'test: ks = {} \t auc = {} '.format(ks_test,auc_test) + '\n'
        prettyks = ks.print_ks(y_test_hat,np.ravel(self.y_test))
        info = info + str(prettyks) + '\n'
        print(info)
        self.report_info = self.report_info + info + '\n'
    
    
                
    
