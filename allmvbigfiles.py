# -*-coding:utf-8-*-
'''
usage
------
当home_path占用空间大于20G时，
其一级子路径下大于1G的文件将移动到data_path。
其一级子路径下大于4G的文件夹中任何大于1G的文件将递归移动到data_path。
example
------
python allmvbigfiles.py 
'''
from __future__ import print_function
import os
import time
import pandas as pd

home_path = '/home/users/liangyun'  # ~主路径
data_path = '/data/liangyun'        #目标移动路径

SAFESIZE = 20000000 #20G    home_path占用空间小于此安全规模则整体不做任何操作
BIGDIRSIZE = 4000000 # 4G   home_path下的第一级文件夹小于此规模的不做任何操作
BIGFILESIZE = 1000000 # 1G  不符合以上两点规则且大于此规模的单个文件会被移动

def mv_big_files(name):
    '''
    usage:
    ------
    当name为文件时且大小大于 BIGFILESIZE时会被移动。
    当name为路径时，其下面所有大于BIGFILESIZE的文件会被移动。递归作用。
    注：不得将此函数直接应用到home路径。
    example:
    ------
    from allmvbigfiles import mv_big_files
    mv_big_files('~/xx_raw_data')  #移动文件 
    mv_big_files('~/app_list')  #递归移动文件夹
    '''

    if os.path.isfile(name):
        if os.path.getsize(name) > BIGFILESIZE:
            os.system('mv %s %s'%(name, data_path))
            print('mv %s %s'%(name, data_path))
    else:
        os.system('du -s %s/* >zz_big_file'%name)
        time.sleep(0.5)
        dfname = pd.read_csv('zz_big_file',sep = '\t',header = None, names = ['size','path'])
        dfname = dfname.query('size > %s'%BIGFILESIZE)
        for path in dfname['path']:
            mv_big_files(path)

def main():
    os.system('du -s %s >zz_big_file'%home_path)
    time.sleep(0.5)
    df = pd.read_csv('zz_big_file',sep = '\t',header = None, names = ['size','path'])
    print('Your home_path size is %s G...'%str(df['size'][0]/1000000.0))
    if df['size'][0]<SAFESIZE:
        print('No big files need to move...')
        return None

    os.system('du -s %s/* >zz_big_file'%home_path)
    time.sleep(0.5)
    df = pd.read_csv('zz_big_file',sep = '\t',header = None, names = ['size','path'])
    
    df = df.query('size >%s'%BIGFILESIZE)
    for size,name in zip(df['size'],df['path']):
        if os.path.isfile(name) or size> BIGDIRSIZE:
            mv_big_files(name)
    
    os.system('du -s %s >zz_big_file'%home_path)
    time.sleep(0.5)
    df = pd.read_csv('zz_big_file',sep = '\t',header = None, names = ['size','path'])
    print('Now your home_path size is %s G...'%str(df['size'][0]/1000000.0))

if __name__ == '__main__':
    main()

