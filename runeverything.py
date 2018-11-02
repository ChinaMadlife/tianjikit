#-*-coding:utf-8-*-
#!/usr/bin/python
from __future__ import print_function
import sys
from tianjikit import runtunning
from tianjikit import runpipeline

def main(train_data_path,test_data_path,outputdir = './aa_everything_result'):
    print('\n********** runtunning.main() **********')
    best_params = runtunning.main(train_data_path,test_data_path,outputdir)
    print('\n********** runpipeline.main() **********')
    runpipeline.main(train_data_path,test_data_path,outputdir,best_params)
    
if __name__ == 'main':
    train_data_path,test_data_path = sys.argv[1],sys.argv[2]
    if len(sys.argv) >=4:
        outputdir = sys.argv[3]
    else:
        outputdir = './aa_everything_result'
    main(train_data_path,test_data_path,outputdir)
    
