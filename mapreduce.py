#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
from __future__ import print_function
import os,sys

#================================================================================
#change the nine arguments here!!!

job_name = 'ly_word_count'
hdfs_input = '/user/hive/warehouse/tj_tmp.db/ly_wordcount_input'
hdfs_output = '/user/hive/warehouse/tj_tmp.db/ly_wordcount_output'
mapper_file = 'word_count_mapper.py'
reducer_file = 'word_count_reducer.py'
map_argv_files = '' 
reduce_argv_files = ''
other_relayed_files = ''  #files to import or other use
getmerge_file = ''  #file to getmerge the output

# if there are more than 1 map_argv_files or reduce_argv_files 
# or other_relayed_files or hdfs_input or hdfs_output
# seperate them with a blank like below
# map_argv_files = 'file1 file2 file3'

# don't change the code below!!!
#================================================================================

def main(job_name,hdfs_input,hdfs_output,mapper_file,reducer_file,map_argv_files = "",
    reduce_argv_files = "",other_relayed_files = "",getmerge_file = ""): 
    '''
    Examples:
    --------
    import mapreduce
    kv = {"job_name":"ly_word_count",
          "hdfs_input":"/user/hive/warehouse/tj_tmp.db/ly_wordcount_input/*",
          "hdfs_output":"/user/hive/warehouse/tj_tmp.db/ly_wordcount_output",
          "mapper_file":"word_count_mapper.py",
          "reducer_file":"word_count_reducer.py",
          "map_argv_files":"",
          "reduce_argv_files":"",
          "other_relayed_files":"",  #files to import or other use
          "getmerge_file":"" #file to getmerge the output
         }
    mapreduce.main(**kv)
    
    # if there are more than 1 map_argv_files or reduce_argv_files or other_relayed_files
    # seperate them with a blank like below
    # {"map_argv_files":"file1 file2 file3"}
    
    mapreduce.get_logs('2019878423_98765','zz_mr_logs')
    # if there are some erros, use get_logs to get yarn logs.
    '''
    
    set_mapper = ('"python2.7 %s %s "' % (mapper_file,map_argv_files)) \
                if mapper_file else 'cat '
    set_reducer = ('"python2.7 %s %s "' % (reducer_file,reduce_argv_files)) \
                if reducer_file else 'NONE '
    files = [mapper_file,reducer_file,map_argv_files,reduce_argv_files,other_relayed_files]
    set_files = ','.join(filter(lambda x:x,files)).replace(' ',',')
    
    print('start %s...'%job_name)
    print('================================================================================\n')
    print('input: %s'%hdfs_input)
    print('output: %s'%hdfs_output)
    print('map: %s'%set_mapper)
    print('reduce: %s'%set_reducer)

    COMMAND_HEAD = "hadoop jar /opt/cloudera/parcels/CDH-5.3.1-1.cdh5.3.1.p0.5/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -libjars  /opt/cloudera/parcels/CDH-5.3.1-1.cdh5.3.1.p0.5/jars/hive-exec-0.13.1-cdh5.3.1.jar "

    command = 'hadoop fs -rm -r ' + hdfs_output
    os.system(command)

    print('================================================================================\n')

    command =  COMMAND_HEAD + \
            ' -files ' + set_files +\
            ' -D mapreduce.job.map.tasks=2000' +\
            ' -D mapreduce.job.map.capacity=1000' +\
            ' -D mapreduce.reduce.tasks=400' +\
            ' -D mapreduce.job.reduce.capacity=200' +\
            ' -D stream.non.zero.exit.is.failure=false' +\
            ' -D mapreduce.job.priority=VERY_HIGH' +\
            ' -D stream.num.map.output.key.fields=1' +\
            ' -D num.key.fields.for.partition=1' +\
            ' -D mapreduce.job.name=%s' % job_name +\
            ' -input ' + hdfs_input + \
            ' -output ' + hdfs_output + \
            ' -mapper '+ set_mapper + \
            ' -reducer '+ set_reducer + \
            ' -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner'

    ret = os.system(command)
    if ret != 0:
        print(job_name + ' mr job error!',file = sys.stderr)

    lastcommand = 'hdfs dfs -getmerge '+ hdfs_output + ' '+ getmerge_file

    if getmerge_file:
        os.system(lastcommand)

def get_logs(taskid,logfile):
    logcommand = 'yarn logs -applicationId application_{} >{}'.format(taskid,logfile)
    os.system(logcommand)
    

if __name__ == '__main__':
    
    main(job_name,hdfs_input,hdfs_output,mapper_file,reducer_file,
        map_argv_files,reduce_argv_files,other_relayed_files,getmerge_file)

######
#####
####
###
##
#
