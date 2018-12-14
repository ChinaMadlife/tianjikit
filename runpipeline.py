#-*-coding:utf-8-*-
#!/usr/bin/python

from __future__ import print_function
import sys,os,pickle,json,datetime
import numpy as np
import pandas as pd
from tianjikit.analysisfeatures import AnalysisFeatures
from tianjikit.runmodel import RunModel

params_dict = {
    'learning_rate':0.1,
    'n_estimators':50,
    'max_depth':3,
    'min_child_weight':30,
    'gamma':0,
    'subsample':0.8,
    'colsample_bytree':1,
    'reg_alpha':0,
    'reg_lambda':1
   }


def main(train_data_path,test_data_path,outputdir = './aa_pipeline_reports',params_dict = params_dict):
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # 获取数据
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('start reading data...')
    dftrain = pd.read_csv(train_data_path,sep = '\t',encoding = 'utf-8')
    dftest = pd.read_csv(test_data_path,sep = '\t',encoding = 'utf-8')

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
    
    # 保存相应文件
    try:
        dfbasic.to_excel(outputdir + '/basic_analysises.xlsx',encoding = 'utf-8')
        dfks.to_excel(outputdir +'/ks_analysises.xlsx',encoding = 'utf-8')
        dfpsi.to_excel(outputdir + '/psi_analysises.xlsx',encoding = 'utf-8')
    except:
        dfbasic.to_csv(outputdir + '/basic_analysises.csv',sep = '\t',encoding = 'utf-8')
        dfks.to_csv(outputdir +'/ks_analysises.csv',sep = '\t',encoding = 'utf-8')
        dfpsi.to_csv(outputdir + '/psi_analysises.csv',sep = '\t',encoding = 'utf-8')

    # 训练XGBOOST模型
    model = RunModel(dftrain = dftrain,dftest = dftest,coverage_th=0.1, ks_th=0, 
            outliers_th=None, fillna_method= None, scale_method= None,selected_features=None)
    xgb = model.train_xgb(cv=5, model_idx=5,scale_pos_weight=1, n_jobs=-1, seed=10,**params_dict) 
    model.test(xgb)
    dfimportance = model.dfimportances['xgb']
    report_info = model.report_info
    
    # 保存相应文件
    try:
        dfimportance.to_excel(outputdir + '/feature_importance.xlsx',encoding = 'utf-8')
    except:
        dfimportance.to_csv(outputdir + '/feature_importance.csv',sep = '\t',encoding = 'utf-8')
        
    with open(outputdir +'/xgboost_model.pkl','w') as f:
        pickle.dump(xgb,f)
    with open(outputdir + '/model_report','w') as f:
        f.write(report_info)
        
if __name__ == '__main__':
    # 确定输入输出位置
    train_data_path,test_data_path = sys.argv[1],sys.argv[2]
    if len(sys.argv) >=4:
        outputdir = sys.argv[3]
    else:
        outputdir = './aa_pipeline_reports'
    if len(sys.argv) >=5:
        params_json = sys.argv[4]
        params_dict = json.load(params_json)
    main(train_data_path,test_data_path,outputdir,params_dict = params_dict)        
    