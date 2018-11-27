import numpy as np
import pandas as pd

def analysislabels(sample_file):
    dflabel = pd.read_csv(sample_file,sep = '\t',
              encoding = 'utf-8',header = None,names = ['label','idcard','phone','loan_dt'])
    dflabel['month'] = [x[0:7] for x in dflabel['loan_dt']]

    dfpt = pd.pivot_table(dflabel[['month','label']],index = ['month'],aggfunc = [np.mean,len],margins = True)
    dfpt.columns = ['overdue_ratio','order_num']
    return(dfpt)
#
#
#