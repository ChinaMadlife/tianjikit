#coding=utf-8
#!/usr/bin/python2.7

##################################################
#update_dt: 2018-10-12
#author：梁云
#usage: 美化dataframe的打印输出字符串
##################################################

import numpy as np
import pandas as pd
from prettytable import PrettyTable

def pretty_dataframe(df):
    table = PrettyTable([''] + list(df.columns))
    for row in df.itertuples():
        table.add_row(row)
    return table