#-*-coding:utf-8-*-
#!/usr/bin/python

from __future__ import print_function
import sys,os,pickle
import numpy as np
import pandas as pd
from tianjikit.analysisfeatures import AnalysisFeatures
from tianjikit.runmodel import RunModel

if __name__ == '__main__':
    # 确定输入输出位置
    train_data_path,test_data_path = sys.argv[1],sys.argv[2]
    if len(sys.argv) >=4:
        outputdir = sys.argv[3]
    else:
        outputdir = './aa_pipeline_reports'
    main(train_data_path,test_data_path,outputdir)

def main(train_data_path,test_data_path,outputdir = './aa_pipeline_reports'):
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # 获取数据
    print('================================================================================')
    print('start reading data...\n')
    dftrain = pd.read_csv(train_data_path,sep = '\t',encoding = 'utf-8',index_col=0)
    dftest = pd.read_csv(test_data_path,sep = '\t',encoding = 'utf-8',index_col=0)

    # 基本分析
    afs = AnalysisFeatures(dftrain,dftest)
    dfbasic = afs.basic_analysises()
    dfbasic = dfbasic.drop([u't(for mean)', u't_pvalue',u'z(for coverage)', u'z_pvalue'],axis = 1)

    # ks有效性分析
    dfks = afs.ks_analysises()
    
    # 稳定性分析
    dfpsi = afs.psi_analysises()

    # 训练XGBOOST模型
    model = RunModel(dftrain = dftrain,dftest = dftest,coverage_th=0.1, ks_th=0, chi2_th=0, 
            outliers_th=None, fillna_method= None, scale_method= None,selected_features=None)
    xgb = model.train_xgb(cv=5, model_idx=5,
          learning_rate=0.1,n_estimators=50, max_depth=3, min_child_weight=10, gamma=0, 
          subsample=0.8,colsample_bytree=0.9,scale_pos_weight=1, n_jobs=-1, seed=10) 
    model.test(xgb)
    dfimportance = model.dfimportances['xgb']
    report_info = model.report_info

    # 保存相应文件共6个
    dfbasic.to_excel(outputdir + '/basic_analysises.xlsx',encoding = 'utf-8')
    dfks.to_excel(outputdir +'/ks_analysises.xlsx',encoding = 'utf-8')
    dfpsi.to_excel(outputdir + '/psi_analysises.xlsx',encoding = 'utf-8')
    dfimportance.to_excel(outputdir + '/feature_importance.xlsx',encoding = 'utf-8')
    with open(outputdir +'/xgboost_model.pkl','w') as f:
        pickle.dump(xgb,f)
    with open(outputdir + '/model_report') as f:
        f.write(report_info)
        
        