# -*- coding:utf-8 -*-
#!/usr/bin/python2.7
from __future__ import print_function

#================================================================================
# You can change the code here below! 可以改动以下配置代码修改超参优化目标和范围。
#================================================================================
# 一，配置优化目标条件

task_name = 'example'
score_func = 'ks'                                 #优化评估指标，可以为 'ks'或'auc',如xgboost版本低于0.71，只能为'ks'.
score_gap_limit  = 0.03                           #可接受train和validate最大评分差值gap
train_data_path = './xx_train_data'               #训练集数据位置
test_data_path = './xx_test_data'                 #测试集数据位置
outputdir = './aa_tunning_result_' + task_name    #输出文件夹名
n_jobs = 16                                       #并行任务数量

#--------------------------------------------------------------------------------
# 二，配置超参数初始值

# 初始化参数
params_dict = dict()

# 以下为待调整参数
# booster参数
params_dict['learning_rate'] = 0.1        # 学习率，初始值为 0.1，通常越小越好。
params_dict['n_estimators'] = 50          # 加法模型树的数量，初始值为50。

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
params_dict['tree_method'] = 'auto'       # 构建树的策略,可以是auto, exact, approx, hist, 如xgboost版本低于0.71,不能为hist.
params_dict['eval_metric'] =  'auc'       # 评分指标，如xgboost版本低于0.71,auc将不被计算。
params_dict['silent'] = 1
params_dict['scale_pos_weight'] = 1        #不平衡样本时设定为正值可以使算法更快收敛。
params_dict['seed'] = 0

#--------------------------------------------------------------------------------
# 三，配置超参搜索范围

params_test1 = {'learning_rate': [0.1],'n_estimators':[50]}  #此处应配置较大 learning_rate

params_test2 = { 'max_depth': [3], 'min_child_weight': [50,100,200] } 

params_test3 = {'gamma': [0.1,0.5,1]}

params_test4 = { 'subsample': [0.9,1.0],'colsample_bytree': [1.0] } 

params_test5 = { 'reg_alpha': [0.1,1] } 

params_test6 = { 'reg_lambda': [0,0.1] }

params_test7 = {'learning_rate':[0.09,0.08],'n_estimators':[100]} #此处应配置较小learning_rate
#===============================================================================



#================================================================================
# Don't change the code below!!! 以下代码请勿轻易改动。
#================================================================================
#================================================================================
# Don't change the code below!!! 以下代码请勿轻易改动。
#================================================================================
#================================================================================
# Don't change the code below!!! 以下代码请勿轻易改动。
#================================================================================



#Tips : If you want to understand the whole file, you should first understand the function 
# train_xgb  below

'''
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
    try:
        dfresult = pd.DataFrame({(dataset+'_'+feval): result[dataset][feval] 
                   for dataset in datasets for feval in ('auc','ks')})
    except:
        dfresult = pd.DataFrame({(dataset+'_'+feval): result[dataset][feval] 
                   for dataset in datasets for feval in ['ks']})
    
    return bst,dfresult
'''

#================================================================================
#================================================================================
# The code below define class PrettyTable : from prettytable.py
#================================================================================
#================================================================================

#!/usr/bin/env python
#
# Copyright (c) 2009-2014, Luke Maurits <luke@maurits.id.au>
# All rights reserved.
# With contributions from:
#  * Chris Clark
#  * Klein Stephane
#  * John Filleau
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * The name of the author may not be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import copy
import csv
import itertools
import math
import random
import re
import sys
import textwrap
import unicodedata

py3k = sys.version_info[0] >= 3
if py3k:
    unicode = str
    basestring = str
    itermap = map
    iterzip = zip
    uni_chr = chr
    from html.parser import HTMLParser
else: 
    itermap = itertools.imap
    iterzip = itertools.izip
    uni_chr = unichr
    from HTMLParser import HTMLParser

if py3k and sys.version_info[1] >= 2:
    from html import escape
else:
    from cgi import escape

# hrule styles
FRAME = 0
ALL   = 1
NONE  = 2
HEADER = 3

# Table styles
DEFAULT = 10
MSWORD_FRIENDLY = 11
PLAIN_COLUMNS = 12
RANDOM = 20

_re = re.compile("\033\[[0-9;]*m")

def _get_size(text):
    lines = text.split("\n")
    height = len(lines)
    width = max([_str_block_width(line) for line in lines])
    return (width, height)
        
class PrettyTable(object):

    def __init__(self, field_names=None, **kwargs):

        """Return a new PrettyTable instance

        Arguments:

        encoding - Unicode encoding scheme used to decode any encoded input
        title - optional table title
        field_names - list or tuple of field names
        fields - list or tuple of field names to include in displays
        start - index of first data row to include in output
        end - index of last data row to include in output PLUS ONE (list slice style)
        header - print a header showing field names (True or False)
        header_style - stylisation to apply to field names in header ("cap", "title", "upper", "lower" or None)
        border - print a border around the table (True or False)
        hrules - controls printing of horizontal rules after rows.  Allowed values: FRAME, HEADER, ALL, NONE
        vrules - controls printing of vertical rules between columns.  Allowed values: FRAME, ALL, NONE
        int_format - controls formatting of integer data
        float_format - controls formatting of floating point data
        min_table_width - minimum desired table width, in characters
        max_table_width - maximum desired table width, in characters
        padding_width - number of spaces on either side of column data (only used if left and right paddings are None)
        left_padding_width - number of spaces on left hand side of column data
        right_padding_width - number of spaces on right hand side of column data
        vertical_char - single character string used to draw vertical lines
        horizontal_char - single character string used to draw horizontal lines
        junction_char - single character string used to draw line junctions
        sortby - name of field to sort rows by
        sort_key - sorting key function, applied to data points before sorting
        valign - default valign for each row (None, "t", "m" or "b")
        reversesort - True or False to sort in descending or ascending order
        oldsortslice - Slice rows before sorting in the "old style" """

        self.encoding = kwargs.get("encoding", "UTF-8")

        # Data
        self._field_names = []
        self._rows = []
        self.align = {}
        self.valign = {}
        self.max_width = {}
        self.min_width = {}
        self.int_format = {}
        self.float_format = {}
        if field_names:
            self.field_names = field_names
        else:
            self._widths = []

        # Options
        self._options = "title start end fields header border sortby reversesort sort_key attributes format hrules vrules".split()
        self._options.extend("int_format float_format min_table_width max_table_width padding_width left_padding_width right_padding_width".split())
        self._options.extend("vertical_char horizontal_char junction_char header_style valign xhtml print_empty oldsortslice".split())
        self._options.extend("align valign max_width min_width".split())
        for option in self._options:
            if option in kwargs:
                self._validate_option(option, kwargs[option])
            else:
                kwargs[option] = None

        self._title = kwargs["title"] or None
        self._start = kwargs["start"] or 0
        self._end = kwargs["end"] or None
        self._fields = kwargs["fields"] or None
        
        if kwargs["header"] in (True, False):
            self._header = kwargs["header"]
        else:
            self._header = True
        self._header_style = kwargs["header_style"] or None
        if kwargs["border"] in (True, False):
            self._border = kwargs["border"]
        else:
            self._border = True
        self._hrules = kwargs["hrules"] or FRAME
        self._vrules = kwargs["vrules"] or ALL

        self._sortby = kwargs["sortby"] or None
        if kwargs["reversesort"] in (True, False):
            self._reversesort = kwargs["reversesort"]
        else:
            self._reversesort = False
        self._sort_key = kwargs["sort_key"] or (lambda x: x)

        # Column specific arguments, use property.setters
        self.align = kwargs["align"] or {}
        self.valign = kwargs["valign"] or {}
        self.max_width = kwargs["max_width"] or {}
        self.min_width = kwargs["min_width"] or {}
        self.int_format = kwargs["int_format"] or {}
        self.float_format = kwargs["float_format"] or {}
        
        self._min_table_width = kwargs["min_table_width"] or None
        self._max_table_width = kwargs["max_table_width"] or None
        self._padding_width = kwargs["padding_width"] or 1
        self._left_padding_width = kwargs["left_padding_width"] or None
        self._right_padding_width = kwargs["right_padding_width"] or None

        self._vertical_char = kwargs["vertical_char"] or self._unicode("|")
        self._horizontal_char = kwargs["horizontal_char"] or self._unicode("-")
        self._junction_char = kwargs["junction_char"] or self._unicode("+")
        
        if kwargs["print_empty"] in (True, False):
            self._print_empty = kwargs["print_empty"]
        else:
            self._print_empty = True
        if kwargs["oldsortslice"] in (True, False):
            self._oldsortslice = kwargs["oldsortslice"]
        else:
            self._oldsortslice = False
        self._format = kwargs["format"] or False
        self._xhtml = kwargs["xhtml"] or False
        self._attributes = kwargs["attributes"] or {}
   
    def _unicode(self, value):
        if not isinstance(value, basestring):
            value = str(value)
        if not isinstance(value, unicode):
            value = unicode(value, self.encoding, "strict")
        return value

    def _justify(self, text, width, align):
        excess = width - _str_block_width(text)
        if align == "l":
            return text + excess * " "
        elif align == "r":
            return excess * " " + text
        else:
            if excess % 2:
                # Uneven padding
                # Put more space on right if text is of odd length...
                if _str_block_width(text) % 2:
                    return (excess//2)*" " + text + (excess//2 + 1)*" "
                # and more space on left if text is of even length
                else:
                    return (excess//2 + 1)*" " + text + (excess//2)*" "
                # Why distribute extra space this way?  To match the behaviour of
                # the inbuilt str.center() method.
            else:
                # Equal padding on either side
                return (excess//2)*" " + text + (excess//2)*" "

    def __getattr__(self, name):

        if name == "rowcount":
            return len(self._rows)
        elif name == "colcount":
            if self._field_names:
                return len(self._field_names)
            elif self._rows:
                return len(self._rows[0])
            else:
                return 0
        else:
            raise AttributeError(name)
 
    def __getitem__(self, index):

        new = PrettyTable()
        new.field_names = self.field_names
        for attr in self._options:
            setattr(new, "_"+attr, getattr(self, "_"+attr))
        setattr(new, "_align", getattr(self, "_align"))
        if isinstance(index, slice):
            for row in self._rows[index]:
                new.add_row(row)
        elif isinstance(index, int):
            new.add_row(self._rows[index])
        else:
            raise Exception("Index %s is invalid, must be an integer or slice" % str(index))
        return new

    if py3k:
        def __str__(self):
            return self.__unicode__()
    else:
        def __str__(self):
            return self.__unicode__().encode(self.encoding)

    def __unicode__(self):
        return self.get_string()

    ##############################
    # ATTRIBUTE VALIDATORS       #
    ##############################

    # The method _validate_option is all that should be used elsewhere in the code base to validate options.
    # It will call the appropriate validation method for that option.  The individual validation methods should
    # never need to be called directly (although nothing bad will happen if they *are*).
    # Validation happens in TWO places.
    # Firstly, in the property setters defined in the ATTRIBUTE MANAGMENT section.
    # Secondly, in the _get_options method, where keyword arguments are mixed with persistent settings

    def _validate_option(self, option, val):
        if option in ("field_names"):
            self._validate_field_names(val)
        elif option in ("start", "end", "max_width", "min_width", "min_table_width", "max_table_width", "padding_width", "left_padding_width", "right_padding_width", "format"):
            self._validate_nonnegative_int(option, val)
        elif option in ("sortby"):
            self._validate_field_name(option, val)
        elif option in ("sort_key"):
            self._validate_function(option, val)
        elif option in ("hrules"):
            self._validate_hrules(option, val)
        elif option in ("vrules"):
            self._validate_vrules(option, val)
        elif option in ("fields"):
            self._validate_all_field_names(option, val)
        elif option in ("header", "border", "reversesort", "xhtml", "print_empty", "oldsortslice"):
            self._validate_true_or_false(option, val)
        elif option in ("header_style"):
            self._validate_header_style(val)
        elif option in ("int_format"):
            self._validate_int_format(option, val)
        elif option in ("float_format"):
            self._validate_float_format(option, val)
        elif option in ("vertical_char", "horizontal_char", "junction_char"):
            self._validate_single_char(option, val)
        elif option in ("attributes"):
            self._validate_attributes(option, val)

    def _validate_field_names(self, val):
        # Check for appropriate length
        if self._field_names:
            try:
                assert len(val) == len(self._field_names)
            except AssertionError:
                raise Exception("Field name list has incorrect number of values, (actual) %d!=%d (expected)" % (len(val), len(self._field_names)))
        if self._rows:
            try:
                assert len(val) == len(self._rows[0])
            except AssertionError:
                raise Exception("Field name list has incorrect number of values, (actual) %d!=%d (expected)" % (len(val), len(self._rows[0])))
        # Check for uniqueness
        try:
            assert len(val) == len(set(val))
        except AssertionError:
            raise Exception("Field names must be unique!")

    def _validate_header_style(self, val):
        try:
            assert val in ("cap", "title", "upper", "lower", None)
        except AssertionError:
            raise Exception("Invalid header style, use cap, title, upper, lower or None!")

    def _validate_align(self, val):
        try:
            assert val in ["l","c","r"]
        except AssertionError:
            raise Exception("Alignment %s is invalid, use l, c or r!" % val)

    def _validate_valign(self, val):
        try:
            assert val in ["t","m","b",None]
        except AssertionError:
            raise Exception("Alignment %s is invalid, use t, m, b or None!" % val)

    def _validate_nonnegative_int(self, name, val):
        try:
            assert int(val) >= 0
        except AssertionError:
            raise Exception("Invalid value for %s: %s!" % (name, self._unicode(val)))

    def _validate_true_or_false(self, name, val):
        try:
            assert val in (True, False)
        except AssertionError:
            raise Exception("Invalid value for %s!  Must be True or False." % name)

    def _validate_int_format(self, name, val):
        if val == "":
            return
        try:
            assert type(val) in (str, unicode)
            assert val.isdigit()
        except AssertionError:
            raise Exception("Invalid value for %s!  Must be an integer format string." % name)

    def _validate_float_format(self, name, val):
        if val == "":
            return
        try:
            assert type(val) in (str, unicode)
            assert "." in val
            bits = val.split(".")
            assert len(bits) <= 2
            assert bits[0] == "" or bits[0].isdigit()
            assert bits[1] == "" or bits[1].isdigit()
        except AssertionError:
            raise Exception("Invalid value for %s!  Must be a float format string." % name)

    def _validate_function(self, name, val):
        try:
            assert hasattr(val, "__call__")
        except AssertionError:
            raise Exception("Invalid value for %s!  Must be a function." % name)

    def _validate_hrules(self, name, val):
        try:
            assert val in (ALL, FRAME, HEADER, NONE)
        except AssertionError:
            raise Exception("Invalid value for %s!  Must be ALL, FRAME, HEADER or NONE." % name)

    def _validate_vrules(self, name, val):
        try:
            assert val in (ALL, FRAME, NONE)
        except AssertionError:
            raise Exception("Invalid value for %s!  Must be ALL, FRAME, or NONE." % name)

    def _validate_field_name(self, name, val):
        try:
            assert (val in self._field_names) or (val is None)
        except AssertionError:
            raise Exception("Invalid field name: %s!" % val)

    def _validate_all_field_names(self, name, val):
        try:
            for x in val:
                self._validate_field_name(name, x)
        except AssertionError:
            raise Exception("fields must be a sequence of field names!")

    def _validate_single_char(self, name, val):
        try:
            assert _str_block_width(val) == 1
        except AssertionError:
            raise Exception("Invalid value for %s!  Must be a string of length 1." % name)

    def _validate_attributes(self, name, val):
        try:
            assert isinstance(val, dict)
        except AssertionError:
            raise Exception("attributes must be a dictionary of name/value pairs!")

    ##############################
    # ATTRIBUTE MANAGEMENT       #
    ##############################

    @property
    def field_names(self):
        """List or tuple of field names"""
        return self._field_names

    @field_names.setter
    def field_names(self, val):
        val = [self._unicode(x) for x in val]
        self._validate_option("field_names", val)
        if self._field_names:
            old_names = self._field_names[:]
        self._field_names = val
        if self._align and old_names:
            for old_name, new_name in zip(old_names, val):
                self._align[new_name] = self._align[old_name]
            for old_name in old_names:
                if old_name not in self._align:
                    self._align.pop(old_name)
        else:
            self.align = "c"
        if self._valign and old_names:
            for old_name, new_name in zip(old_names, val):
                self._valign[new_name] = self._valign[old_name]
            for old_name in old_names:
                if old_name not in self._valign:
                    self._valign.pop(old_name)
        else:
            self.valign = "t"
    @property
    def align(self):
        """Controls alignment of fields
        Arguments:

        align - alignment, one of "l", "c", or "r" """
        return self._align

    @align.setter
    def align(self, val):
        if not self._field_names:
            self._align = {}
        elif val is None or (isinstance(val,dict) and len(val) is 0):
            for field in self._field_names:
                self._align[field] = "c"
        else:
            self._validate_align(val)
            for field in self._field_names:
                self._align[field] = val

    @property
    def valign(self):
        """Controls vertical alignment of fields
        Arguments:

        valign - vertical alignment, one of "t", "m", or "b" """
        return self._valign
    @valign.setter
    def valign(self, val):
        if not self._field_names:
            self._valign = {}
        elif val is None or (isinstance(val,dict) and len(val) is 0):
            for field in self._field_names:
                self._valign[field] = "t"
        else:
            self._validate_valign(val)
            for field in self._field_names:
                self._valign[field] = val

    @property
    def max_width(self):
        """Controls maximum width of fields
        Arguments:

        max_width - maximum width integer"""
        return self._max_width
    @max_width.setter
    def max_width(self, val):
        if val is None or (isinstance(val,dict) and len(val) is 0):
            self._max_width = {}
        else:
            self._validate_option("max_width",val)
            for field in self._field_names:
                self._max_width[field] = val
   
    @property
    def min_width(self):
        """Controls minimum width of fields
        Arguments:

        min_width - minimum width integer"""
        return self._min_width
    @min_width.setter
    def min_width(self, val):
        if val is None or (isinstance(val,dict) and len(val) is 0):
            self._min_width = {}
        else:
            self._validate_option("min_width",val)
            for field in self._field_names:
                self._min_width[field] = val

    @property
    def min_table_width(self):
        return self._min_table_width

    @min_table_width.setter
    def min_table_width(self, val):
        self._validate_option("min_table_width", val)
        self._min_table_width = val

    @property
    def max_table_width(self):
        return self._max_table_width

    @max_table_width.setter
    def max_table_width(self, val):
        self._validate_option("max_table_width", val)
        self._max_table_width = val

    @property
    def fields(self):
        """List or tuple of field names to include in displays"""
        return self._fields

    @fields.setter
    def fields(self, val):
        self._validate_option("fields", val)
        self._fields = val

    @property
    def title(self):
        """Optional table title

        Arguments:

        title - table title"""
        return self._title

    @title.setter
    def title(self, val):
        self._title = self._unicode(val)

    @property
    def start(self):
        """Start index of the range of rows to print

        Arguments:

        start - index of first data row to include in output"""
        return self._start

    @start.setter
    def start(self, val):
        self._validate_option("start", val)
        self._start = val

    @property
    def end(self):
        """End index of the range of rows to print

        Arguments:

        end - index of last data row to include in output PLUS ONE (list slice style)"""
        return self._end
    @end.setter
    def end(self, val):
        self._validate_option("end", val)
        self._end = val

    @property
    def sortby(self):
        """Name of field by which to sort rows

        Arguments:

        sortby - field name to sort by"""
        return self._sortby
    @sortby.setter
    def sortby(self, val):
        self._validate_option("sortby", val)
        self._sortby = val

    @property
    def reversesort(self):
        """Controls direction of sorting (ascending vs descending)

        Arguments:

        reveresort - set to True to sort by descending order, or False to sort by ascending order"""
        return self._reversesort
    @reversesort.setter
    def reversesort(self, val):
        self._validate_option("reversesort", val)
        self._reversesort = val

    @property
    def sort_key(self):
        """Sorting key function, applied to data points before sorting

        Arguments:

        sort_key - a function which takes one argument and returns something to be sorted"""
        return self._sort_key
    @sort_key.setter
    def sort_key(self, val):
        self._validate_option("sort_key", val)
        self._sort_key = val
 
    @property
    def header(self):
        """Controls printing of table header with field names

        Arguments:

        header - print a header showing field names (True or False)"""
        return self._header
    @header.setter
    def header(self, val):
        self._validate_option("header", val)
        self._header = val

    @property
    def header_style(self):
        """Controls stylisation applied to field names in header

        Arguments:

        header_style - stylisation to apply to field names in header ("cap", "title", "upper", "lower" or None)"""
        return self._header_style
    @header_style.setter
    def header_style(self, val):
        self._validate_header_style(val)
        self._header_style = val

    @property
    def border(self):
        """Controls printing of border around table

        Arguments:

        border - print a border around the table (True or False)"""
        return self._border
    @border.setter
    def border(self, val):
        self._validate_option("border", val)
        self._border = val

    @property
    def hrules(self):
        """Controls printing of horizontal rules after rows

        Arguments:

        hrules - horizontal rules style.  Allowed values: FRAME, ALL, HEADER, NONE"""
        return self._hrules
    @hrules.setter
    def hrules(self, val):
        self._validate_option("hrules", val)
        self._hrules = val

    @property
    def vrules(self):
        """Controls printing of vertical rules between columns

        Arguments:

        vrules - vertical rules style.  Allowed values: FRAME, ALL, NONE"""
        return self._vrules
    @vrules.setter
    def vrules(self, val):
        self._validate_option("vrules", val)
        self._vrules = val

    @property
    def int_format(self):
        """Controls formatting of integer data
        Arguments:

        int_format - integer format string"""
        return self._int_format
    @int_format.setter
    def int_format(self, val):
        if val is None or (isinstance(val,dict) and len(val) is 0):
            self._int_format = {}
        else:
            self._validate_option("int_format",val)
            for field in self._field_names:
                self._int_format[field] = val

    @property
    def float_format(self):
        """Controls formatting of floating point data
        Arguments:

        float_format - floating point format string"""
        return self._float_format
    @float_format.setter
    def float_format(self, val):
        if val is None or (isinstance(val,dict) and len(val) is 0):
            self._float_format = {}
        else:
            self._validate_option("float_format",val)
            for field in self._field_names:
                self._float_format[field] = val

    @property
    def padding_width(self):
        """The number of empty spaces between a column's edge and its content

        Arguments:

        padding_width - number of spaces, must be a positive integer"""
        return self._padding_width
    @padding_width.setter
    def padding_width(self, val):
        self._validate_option("padding_width", val)
        self._padding_width = val

    @property
    def left_padding_width(self):
        """The number of empty spaces between a column's left edge and its content

        Arguments:

        left_padding - number of spaces, must be a positive integer"""
        return self._left_padding_width
    @left_padding_width.setter
    def left_padding_width(self, val):
        self._validate_option("left_padding_width", val)
        self._left_padding_width = val

    @property
    def right_padding_width(self):
        """The number of empty spaces between a column's right edge and its content

        Arguments:

        right_padding - number of spaces, must be a positive integer"""
        return self._right_padding_width
    @right_padding_width.setter
    def right_padding_width(self, val):
        self._validate_option("right_padding_width", val)
        self._right_padding_width = val

    @property
    def vertical_char(self):
        """The charcter used when printing table borders to draw vertical lines

        Arguments:

        vertical_char - single character string used to draw vertical lines"""
        return self._vertical_char
    @vertical_char.setter
    def vertical_char(self, val):
        val = self._unicode(val)
        self._validate_option("vertical_char", val)
        self._vertical_char = val

    @property
    def horizontal_char(self):
        """The charcter used when printing table borders to draw horizontal lines

        Arguments:

        horizontal_char - single character string used to draw horizontal lines"""
        return self._horizontal_char
    @horizontal_char.setter
    def horizontal_char(self, val):
        val = self._unicode(val)
        self._validate_option("horizontal_char", val)
        self._horizontal_char = val

    @property
    def junction_char(self):
        """The charcter used when printing table borders to draw line junctions

        Arguments:

        junction_char - single character string used to draw line junctions"""
        return self._junction_char
    @junction_char.setter
    def junction_char(self, val):
        val = self._unicode(val)
        self._validate_option("vertical_char", val)
        self._junction_char = val

    @property
    def format(self):
        """Controls whether or not HTML tables are formatted to match styling options

        Arguments:

        format - True or False"""
        return self._format
    @format.setter
    def format(self, val):
        self._validate_option("format", val)
        self._format = val

    @property
    def print_empty(self):
        """Controls whether or not empty tables produce a header and frame or just an empty string

        Arguments:

        print_empty - True or False"""
        return self._print_empty
    @print_empty.setter
    def print_empty(self, val):
        self._validate_option("print_empty", val)
        self._print_empty = val

    @property
    def attributes(self):
        """A dictionary of HTML attribute name/value pairs to be included in the <table> tag when printing HTML

        Arguments:

        attributes - dictionary of attributes"""
        return self._attributes

    @attributes.setter
    def attributes(self, val):
        self._validate_option("attributes", val)
        self._attributes = val

    @property
    def oldsortslice(self):
        """ oldsortslice - Slice rows before sorting in the "old style" """
        return self._oldsortslice
    @oldsortslice.setter
    def oldsortslice(self, val):
        self._validate_option("oldsortslice", val)
        self._oldsortslice = val

    ##############################
    # OPTION MIXER               #
    ##############################

    def _get_options(self, kwargs):

        options = {}
        for option in self._options:
            if option in kwargs:
                self._validate_option(option, kwargs[option])
                options[option] = kwargs[option]
            else:
                options[option] = getattr(self, "_"+option)
        return options

    ##############################
    # PRESET STYLE LOGIC         #
    ##############################

    def set_style(self, style):

        if style == DEFAULT:
            self._set_default_style()
        elif style == MSWORD_FRIENDLY:
            self._set_msword_style()
        elif style == PLAIN_COLUMNS:
            self._set_columns_style()
        elif style == RANDOM:
            self._set_random_style()
        else:
            raise Exception("Invalid pre-set style!")

    def _set_default_style(self):

        self.header = True
        self.border = True
        self._hrules = FRAME
        self._vrules = ALL
        self.padding_width = 1
        self.left_padding_width = 1
        self.right_padding_width = 1
        self.vertical_char = "|"
        self.horizontal_char = "-"
        self.junction_char = "+"

    def _set_msword_style(self):

        self.header = True
        self.border = True
        self._hrules = NONE
        self.padding_width = 1
        self.left_padding_width = 1
        self.right_padding_width = 1
        self.vertical_char = "|"

    def _set_columns_style(self):

        self.header = True
        self.border = False
        self.padding_width = 1
        self.left_padding_width = 0
        self.right_padding_width = 8

    def _set_random_style(self):

        # Just for fun!
        self.header = random.choice((True, False))
        self.border = random.choice((True, False))
        self._hrules = random.choice((ALL, FRAME, HEADER, NONE))
        self._vrules = random.choice((ALL, FRAME, NONE))
        self.left_padding_width = random.randint(0,5)
        self.right_padding_width = random.randint(0,5)
        self.vertical_char = random.choice("~!@#$%^&*()_+|-=\{}[];':\",./;<>?")
        self.horizontal_char = random.choice("~!@#$%^&*()_+|-=\{}[];':\",./;<>?")
        self.junction_char = random.choice("~!@#$%^&*()_+|-=\{}[];':\",./;<>?")

    ##############################
    # DATA INPUT METHODS         #
    ##############################

    def add_row(self, row):

        """Add a row to the table

        Arguments:

        row - row of data, should be a list with as many elements as the table
        has fields"""

        if self._field_names and len(row) != len(self._field_names):
            raise Exception("Row has incorrect number of values, (actual) %d!=%d (expected)" %(len(row),len(self._field_names)))
        if not self._field_names:
            self.field_names = [("Field %d" % (n+1)) for n in range(0,len(row))]
        self._rows.append(list(row))

    def del_row(self, row_index):

        """Delete a row to the table

        Arguments:

        row_index - The index of the row you want to delete.  Indexing starts at 0."""

        if row_index > len(self._rows)-1:
            raise Exception("Cant delete row at index %d, table only has %d rows!" % (row_index, len(self._rows)))
        del self._rows[row_index]

    def add_column(self, fieldname, column, align="c", valign="t"):

        """Add a column to the table.

        Arguments:

        fieldname - name of the field to contain the new column of data
        column - column of data, should be a list with as many elements as the
        table has rows
        align - desired alignment for this column - "l" for left, "c" for centre and "r" for right
        valign - desired vertical alignment for new columns - "t" for top, "m" for middle and "b" for bottom"""

        if len(self._rows) in (0, len(column)):
            self._validate_align(align)
            self._validate_valign(valign)
            self._field_names.append(fieldname)
            self._align[fieldname] = align
            self._valign[fieldname] = valign
            for i in range(0, len(column)):
                if len(self._rows) < i+1:
                    self._rows.append([])
                self._rows[i].append(column[i])
        else:
            raise Exception("Column length %d does not match number of rows %d!" % (len(column), len(self._rows)))

    def clear_rows(self):

        """Delete all rows from the table but keep the current field names"""

        self._rows = []

    def clear(self):

        """Delete all rows and field names from the table, maintaining nothing but styling options"""

        self._rows = []
        self._field_names = []
        self._widths = []

    ##############################
    # MISC PUBLIC METHODS        #
    ##############################

    def copy(self):
        return copy.deepcopy(self)

    ##############################
    # MISC PRIVATE METHODS       #
    ##############################

    def _format_value(self, field, value):
        if isinstance(value, int) and field in self._int_format:
            value = self._unicode(("%%%sd" % self._int_format[field]) % value)
        elif isinstance(value, float) and field in self._float_format:
            value = self._unicode(("%%%sf" % self._float_format[field]) % value)
        return self._unicode(value)

    def _compute_table_width(self, options):
        table_width = 2 if options["vrules"] in (FRAME, ALL) else 0
        per_col_padding = sum(self._get_padding_widths(options))
        for index, fieldname in enumerate(self.field_names):
            if not options["fields"] or (options["fields"] and fieldname in options["fields"]):
                table_width += self._widths[index] + per_col_padding
        return table_width

    def _compute_widths(self, rows, options):
        if options["header"]:
            widths = [_get_size(field)[0] for field in self._field_names]
        else:
            widths = len(self.field_names) * [0]

        for row in rows:
            for index, value in enumerate(row):
                fieldname = self.field_names[index]
                if fieldname in self.max_width:
                    widths[index] = max(widths[index], min(_get_size(value)[0], self.max_width[fieldname]))
                else:
                    widths[index] = max(widths[index], _get_size(value)[0])
                if fieldname in self.min_width:
                    widths[index] = max(widths[index], self.min_width[fieldname])
        self._widths = widths

        # Are we exceeding max_table_width?
        if self._max_table_width:
            table_width = self._compute_table_width(options)
            if table_width > self._max_table_width:
                # Shrink widths in proportion
                scale = 1.0*self._max_table_width / table_width
                widths = [int(math.floor(w*scale)) for w in widths]
                self._widths = widths

        # Are we under min_table_width or title width?
        if self._min_table_width or options["title"]:
            if options["title"]:
                title_width = len(options["title"])+sum(self._get_padding_widths(options))
                if options["vrules"] in (FRAME, ALL):
                    title_width += 2
            else:
                title_width = 0
            min_table_width = self.min_table_width or 0
            min_width = max(title_width, min_table_width)
            table_width = self._compute_table_width(options)
            if table_width < min_width:
                # Grow widths in proportion
                scale = 1.0*min_width / table_width
                widths = [int(math.ceil(w*scale)) for w in widths]
                self._widths = widths

    def _get_padding_widths(self, options):

        if options["left_padding_width"] is not None:
            lpad = options["left_padding_width"]
        else:
            lpad = options["padding_width"]
        if options["right_padding_width"] is not None:
            rpad = options["right_padding_width"]
        else:
            rpad = options["padding_width"]
        return lpad, rpad

    def _get_rows(self, options):
        """Return only those data rows that should be printed, based on slicing and sorting.

        Arguments:

        options - dictionary of option settings."""
      
        if options["oldsortslice"]:
            rows = copy.deepcopy(self._rows[options["start"]:options["end"]])
        else:
            rows = copy.deepcopy(self._rows)

        # Sort
        if options["sortby"]:
            sortindex = self._field_names.index(options["sortby"])
            # Decorate
            rows = [[row[sortindex]]+row for row in rows]
            # Sort
            rows.sort(reverse=options["reversesort"], key=options["sort_key"])
            # Undecorate
            rows = [row[1:] for row in rows]

        # Slice if necessary
        if not options["oldsortslice"]:
            rows = rows[options["start"]:options["end"]]

        return rows
        
    def _format_row(self, row, options):
        return [self._format_value(field, value) for (field, value) in zip(self._field_names, row)]

    def _format_rows(self, rows, options):
        return [self._format_row(row, options) for row in rows]
 
    ##############################
    # PLAIN TEXT STRING METHODS  #
    ##############################

    def get_string(self, **kwargs):

        """Return string representation of table in current state.

        Arguments:

        title - optional table title
        start - index of first data row to include in output
        end - index of last data row to include in output PLUS ONE (list slice style)
        fields - names of fields (columns) to include
        header - print a header showing field names (True or False)
        border - print a border around the table (True or False)
        hrules - controls printing of horizontal rules after rows.  Allowed values: ALL, FRAME, HEADER, NONE
        vrules - controls printing of vertical rules between columns.  Allowed values: FRAME, ALL, NONE
        int_format - controls formatting of integer data
        float_format - controls formatting of floating point data
        padding_width - number of spaces on either side of column data (only used if left and right paddings are None)
        left_padding_width - number of spaces on left hand side of column data
        right_padding_width - number of spaces on right hand side of column data
        vertical_char - single character string used to draw vertical lines
        horizontal_char - single character string used to draw horizontal lines
        junction_char - single character string used to draw line junctions
        sortby - name of field to sort rows by
        sort_key - sorting key function, applied to data points before sorting
        reversesort - True or False to sort in descending or ascending order
        print empty - if True, stringify just the header for an empty table, if False return an empty string """

        options = self._get_options(kwargs)

        lines = []

        # Don't think too hard about an empty table
        # Is this the desired behaviour?  Maybe we should still print the header?
        if self.rowcount == 0 and (not options["print_empty"] or not options["border"]):
            return ""

        # Get the rows we need to print, taking into account slicing, sorting, etc.
        rows = self._get_rows(options)

        # Turn all data in all rows into Unicode, formatted as desired
        formatted_rows = self._format_rows(rows, options)

        # Compute column widths
        self._compute_widths(formatted_rows, options)
        self._hrule = self._stringify_hrule(options)

        # Add title
        title = options["title"] or self._title
        if title:
            lines.append(self._stringify_title(title, options))

        # Add header or top of border
        if options["header"]:
            lines.append(self._stringify_header(options))
        elif options["border"] and options["hrules"] in (ALL, FRAME):
            lines.append(self._hrule)

        # Add rows
        for row in formatted_rows:
            lines.append(self._stringify_row(row, options))

        # Add bottom of border
        if options["border"] and options["hrules"] == FRAME:
            lines.append(self._hrule)
        
        return self._unicode("\n").join(lines)

    def _stringify_hrule(self, options):

        if not options["border"]:
            return ""
        lpad, rpad = self._get_padding_widths(options)
        if options['vrules'] in (ALL, FRAME):
            bits = [options["junction_char"]]
        else:
            bits = [options["horizontal_char"]]
        # For tables with no data or fieldnames
        if not self._field_names:
                bits.append(options["junction_char"])
                return "".join(bits)
        for field, width in zip(self._field_names, self._widths):
            if options["fields"] and field not in options["fields"]:
                continue
            bits.append((width+lpad+rpad)*options["horizontal_char"])
            if options['vrules'] == ALL:
                bits.append(options["junction_char"])
            else:
                bits.append(options["horizontal_char"])
        if options["vrules"] == FRAME:
            bits.pop()
            bits.append(options["junction_char"])
        return "".join(bits)

    def _stringify_title(self, title, options):

        lines = []
        lpad, rpad = self._get_padding_widths(options)
        if options["border"]:
            if options["vrules"] == ALL:
                options["vrules"] = FRAME
                lines.append(self._stringify_hrule(options))
                options["vrules"] = ALL
            elif options["vrules"] == FRAME:
                lines.append(self._stringify_hrule(options))
        bits = []
        endpoint = options["vertical_char"] if options["vrules"] in (ALL, FRAME) else " "
        bits.append(endpoint)
        title = " "*lpad + title + " "*rpad
        bits.append(self._justify(title, len(self._hrule)-2, "c"))
        bits.append(endpoint)
        lines.append("".join(bits))
        return "\n".join(lines)

    def _stringify_header(self, options):

        bits = []
        lpad, rpad = self._get_padding_widths(options)
        if options["border"]:
            if options["hrules"] in (ALL, FRAME):
                bits.append(self._hrule)
                bits.append("\n")
            if options["vrules"] in (ALL, FRAME):
                bits.append(options["vertical_char"])
            else:
                bits.append(" ")
        # For tables with no data or field names
        if not self._field_names:
            if options["vrules"] in (ALL, FRAME):
                bits.append(options["vertical_char"])
            else:
                bits.append(" ")
        for field, width, in zip(self._field_names, self._widths):
            if options["fields"] and field not in options["fields"]:
                continue
            if self._header_style == "cap":
                fieldname = field.capitalize()
            elif self._header_style == "title":
                fieldname = field.title()
            elif self._header_style == "upper":
                fieldname = field.upper()
            elif self._header_style == "lower":
                fieldname = field.lower()
            else:
                fieldname = field
            bits.append(" " * lpad + self._justify(fieldname, width, self._align[field]) + " " * rpad)
            if options["border"]:
                if options["vrules"] == ALL:
                    bits.append(options["vertical_char"])
                else:
                    bits.append(" ")
        # If vrules is FRAME, then we just appended a space at the end
        # of the last field, when we really want a vertical character
        if options["border"] and options["vrules"] == FRAME:
            bits.pop()
            bits.append(options["vertical_char"])
        if options["border"] and options["hrules"] != NONE:
            bits.append("\n")
            bits.append(self._hrule)
        return "".join(bits)

    def _stringify_row(self, row, options):
       
        for index, field, value, width, in zip(range(0,len(row)), self._field_names, row, self._widths):
            # Enforce max widths
            lines = value.split("\n")
            new_lines = []
            for line in lines: 
                if _str_block_width(line) > width:
                    line = textwrap.fill(line, width)
                new_lines.append(line)
            lines = new_lines
            value = "\n".join(lines)
            row[index] = value

        row_height = 0
        for c in row:
            h = _get_size(c)[1]
            if h > row_height:
                row_height = h

        bits = []
        lpad, rpad = self._get_padding_widths(options)
        for y in range(0, row_height):
            bits.append([])
            if options["border"]:
                if options["vrules"] in (ALL, FRAME):
                    bits[y].append(self.vertical_char)
                else:
                    bits[y].append(" ")

        for field, value, width, in zip(self._field_names, row, self._widths):

            valign = self._valign[field]
            lines = value.split("\n")
            dHeight = row_height - len(lines)
            if dHeight:
                if valign == "m":
                    lines = [""] * int(dHeight / 2) + lines + [""] * (dHeight - int(dHeight / 2))
                elif valign == "b":
                    lines = [""] * dHeight + lines
                else:
                    lines = lines + [""] * dHeight

            y = 0
            for l in lines:
                if options["fields"] and field not in options["fields"]:
                    continue

                bits[y].append(" " * lpad + self._justify(l, width, self._align[field]) + " " * rpad)
                if options["border"]:
                    if options["vrules"] == ALL:
                        bits[y].append(self.vertical_char)
                    else:
                        bits[y].append(" ")
                y += 1

        # If vrules is FRAME, then we just appended a space at the end
        # of the last field, when we really want a vertical character
        for y in range(0, row_height):
            if options["border"] and options["vrules"] == FRAME:
                bits[y].pop()
                bits[y].append(options["vertical_char"])
        
        if options["border"] and options["hrules"]== ALL:
            bits[row_height-1].append("\n")
            bits[row_height-1].append(self._hrule)

        for y in range(0, row_height):
            bits[y] = "".join(bits[y])

        return "\n".join(bits)

    def paginate(self, page_length=58, **kwargs):

        pages = []
        kwargs["start"] = kwargs.get("start", 0)
        true_end = kwargs.get("end", self.rowcount)
        while True:
            kwargs["end"] = min(kwargs["start"] + page_length, true_end)
            pages.append(self.get_string(**kwargs))
            if kwargs["end"] == true_end:
                break
            kwargs["start"] += page_length 
        return "\f".join(pages)

    ##############################
    # HTML STRING METHODS        #
    ##############################

    def get_html_string(self, **kwargs):

        """Return string representation of HTML formatted version of table in current state.

        Arguments:

        title - optional table title
        start - index of first data row to include in output
        end - index of last data row to include in output PLUS ONE (list slice style)
        fields - names of fields (columns) to include
        header - print a header showing field names (True or False)
        border - print a border around the table (True or False)
        hrules - controls printing of horizontal rules after rows.  Allowed values: ALL, FRAME, HEADER, NONE
        vrules - controls printing of vertical rules between columns.  Allowed values: FRAME, ALL, NONE
        int_format - controls formatting of integer data
        float_format - controls formatting of floating point data
        padding_width - number of spaces on either side of column data (only used if left and right paddings are None)
        left_padding_width - number of spaces on left hand side of column data
        right_padding_width - number of spaces on right hand side of column data
        sortby - name of field to sort rows by
        sort_key - sorting key function, applied to data points before sorting
        attributes - dictionary of name/value pairs to include as HTML attributes in the <table> tag
        xhtml - print <br/> tags if True, <br> tags if false"""

        options = self._get_options(kwargs)

        if options["format"]:
            string = self._get_formatted_html_string(options)
        else:
            string = self._get_simple_html_string(options)

        return string

    def _get_simple_html_string(self, options):

        lines = []
        if options["xhtml"]:
            linebreak = "<br/>"
        else:
            linebreak = "<br>"

        open_tag = []
        open_tag.append("<table")
        if options["attributes"]:
            for attr_name in options["attributes"]:
                open_tag.append(" %s=\"%s\"" % (attr_name, options["attributes"][attr_name]))
        open_tag.append(">")
        lines.append("".join(open_tag))

        # Title
        title = options["title"] or self._title
        if title:
            cols = len(options["fields"]) if options["fields"] else len(self.field_names)
            lines.append("    <tr>")
            lines.append("        <td colspan=%d>%s</td>" % (cols, title))
            lines.append("    </tr>")

        # Headers
        if options["header"]:
            lines.append("    <tr>")
            for field in self._field_names:
                if options["fields"] and field not in options["fields"]:
                    continue
                lines.append("        <th>%s</th>" % escape(field).replace("\n", linebreak))
            lines.append("    </tr>")

        # Data
        rows = self._get_rows(options)
        formatted_rows = self._format_rows(rows, options)
        for row in formatted_rows:
            lines.append("    <tr>")
            for field, datum in zip(self._field_names, row):
                if options["fields"] and field not in options["fields"]:
                    continue
                lines.append("        <td>%s</td>" % escape(datum).replace("\n", linebreak))
            lines.append("    </tr>")

        lines.append("</table>")

        return self._unicode("\n").join(lines)

    def _get_formatted_html_string(self, options):

        lines = []
        lpad, rpad = self._get_padding_widths(options)
        if options["xhtml"]:
            linebreak = "<br/>"
        else:
            linebreak = "<br>"

        open_tag = []
        open_tag.append("<table")
        if options["border"]:
            if options["hrules"] == ALL and options["vrules"] == ALL:
                open_tag.append(" frame=\"box\" rules=\"all\"")
            elif options["hrules"] == FRAME and options["vrules"] == FRAME:
                open_tag.append(" frame=\"box\"")
            elif options["hrules"] == FRAME and options["vrules"] == ALL:
                open_tag.append(" frame=\"box\" rules=\"cols\"")
            elif options["hrules"] == FRAME:
                open_tag.append(" frame=\"hsides\"")
            elif options["hrules"] == ALL:
                open_tag.append(" frame=\"hsides\" rules=\"rows\"")
            elif options["vrules"] == FRAME:
                open_tag.append(" frame=\"vsides\"")
            elif options["vrules"] == ALL:
                open_tag.append(" frame=\"vsides\" rules=\"cols\"")
        if options["attributes"]:
            for attr_name in options["attributes"]:
                open_tag.append(" %s=\"%s\"" % (attr_name, options["attributes"][attr_name]))
        open_tag.append(">")
        lines.append("".join(open_tag))

        # Title
        title = options["title"] or self._title
        if title:
            cols = len(options["fields"]) if options["fields"] else len(self.field_names)
            lines.append("    <tr>")
            lines.append("        <td colspan=%d>%s</td>" % (cols, title))
            lines.append("    </tr>")

        # Headers
        if options["header"]:
            lines.append("    <tr>")
            for field in self._field_names:
                if options["fields"] and field not in options["fields"]:
                    continue
                lines.append("        <th style=\"padding-left: %dem; padding-right: %dem; text-align: center\">%s</th>" % (lpad, rpad, escape(field).replace("\n", linebreak)))
            lines.append("    </tr>")

        # Data
        rows = self._get_rows(options)
        formatted_rows = self._format_rows(rows, options)
        aligns = []
        valigns = []
        for field in self._field_names:
            aligns.append({ "l" : "left", "r" : "right", "c" : "center" }[self._align[field]])
            valigns.append({"t" : "top", "m" : "middle", "b" : "bottom"}[self._valign[field]])
        for row in formatted_rows:
            lines.append("    <tr>")
            for field, datum, align, valign in zip(self._field_names, row, aligns, valigns):
                if options["fields"] and field not in options["fields"]:
                    continue
                lines.append("        <td style=\"padding-left: %dem; padding-right: %dem; text-align: %s; vertical-align: %s\">%s</td>" % (lpad, rpad, align, valign, escape(datum).replace("\n", linebreak)))
            lines.append("    </tr>")
        lines.append("</table>")

        return self._unicode("\n").join(lines)

##############################
# UNICODE WIDTH FUNCTIONS    #
##############################

def _char_block_width(char):
    # Basic Latin, which is probably the most common case
    #if char in xrange(0x0021, 0x007e):
    #if char >= 0x0021 and char <= 0x007e:
    if 0x0021 <= char <= 0x007e:
        return 1
    # Chinese, Japanese, Korean (common)
    if 0x4e00 <= char <= 0x9fff:
        return 2
    # Hangul
    if 0xac00 <= char <= 0xd7af:
        return 2
    # Combining?
    if unicodedata.combining(uni_chr(char)):
        return 0
    # Hiragana and Katakana
    if 0x3040 <= char <= 0x309f or 0x30a0 <= char <= 0x30ff:
        return 2
    # Full-width Latin characters
    if 0xff01 <= char <= 0xff60:
        return 2
    # CJK punctuation
    if 0x3000 <= char <= 0x303e:
        return 2
    # Backspace and delete
    if char in (0x0008, 0x007f):
        return -1
    # Other control characters
    elif char in (0x0000, 0x000f, 0x001f):
        return 0
    # Take a guess
    return 1

def _str_block_width(val):

    return sum(itermap(_char_block_width, itermap(ord, _re.sub("", val))))

##############################
# TABLE FACTORIES            #
##############################

def from_csv(fp, field_names = None, **kwargs):

    fmtparams = {}
    for param in ["delimiter", "doublequote", "escapechar", "lineterminator",
            "quotechar", "quoting", "skipinitialspace", "strict"]:
        if param in kwargs:
            fmtparams[param] = kwargs.pop(param)
    if fmtparams:
        reader = csv.reader(fp, **fmtparams)
    else:
        dialect = csv.Sniffer().sniff(fp.read(1024))
        fp.seek(0)
        reader = csv.reader(fp, dialect)

    table = PrettyTable(**kwargs)
    if field_names:
        table.field_names = field_names
    else:
        if py3k:
            table.field_names = [x.strip() for x in next(reader)]
        else:
            table.field_names = [x.strip() for x in reader.next()]

    for row in reader:
        table.add_row([x.strip() for x in row])

    return table

def from_db_cursor(cursor, **kwargs):

    if cursor.description:
        table = PrettyTable(**kwargs)
        table.field_names = [col[0] for col in cursor.description]
        for row in cursor.fetchall():
            table.add_row(row)
        return table

class TableHandler(HTMLParser):

    def __init__(self, **kwargs):
        HTMLParser.__init__(self)
        self.kwargs = kwargs
        self.tables = []
        self.last_row = []
        self.rows = []
        self.max_row_width = 0
        self.active = None
        self.last_content = ""
        self.is_last_row_header = False
        self.colspan = 0

    def handle_starttag(self,tag, attrs):
        self.active = tag
        if tag == "th":
            self.is_last_row_header = True
        for (key, value) in attrs:
            if key == "colspan":
                self.colspan = int(value)


    def handle_endtag(self,tag):
        if tag in ["th", "td"]:
            stripped_content = self.last_content.strip()
            self.last_row.append(stripped_content)
            if self.colspan:
                for i in range(1, self.colspan):
                    self.last_row.append("")
                self.colspan = 0

        if tag == "tr":
            self.rows.append(
                (self.last_row, self.is_last_row_header))
            self.max_row_width = max(self.max_row_width, len(self.last_row))
            self.last_row = []
            self.is_last_row_header = False
        if tag == "table":
            table = self.generate_table(self.rows)
            self.tables.append(table)
            self.rows = []
        self.last_content = " "
        self.active = None


    def handle_data(self, data):
        self.last_content += data

    def generate_table(self, rows):
        """
        Generates from a list of rows a PrettyTable object.
        """
        table = PrettyTable(**self.kwargs)
        for row in self.rows:
            if len(row[0]) < self.max_row_width:
                appends = self.max_row_width - len(row[0])
                for i in range(1,appends):
                    row[0].append("-")

            if row[1] == True:
                self.make_fields_unique(row[0])
                table.field_names = row[0]
            else:
                table.add_row(row[0])
        return table

    def make_fields_unique(self, fields):
        """
        iterates over the row and make each field unique
        """
        for i in range(0, len(fields)):
            for j in range(i+1, len(fields)):
                if fields[i] == fields[j]:
                    fields[j] += "'"

def from_html(html_code, **kwargs):
    """
    Generates a list of PrettyTables from a string of HTML code. Each <table> in
    the HTML becomes one PrettyTable object.
    """

    parser = TableHandler(**kwargs)
    parser.feed(html_code)
    return parser.tables

def from_html_one(html_code, **kwargs):
    """
    Generates a PrettyTables from a string of HTML code which contains only a
    single <table>
    """

    tables = from_html(html_code, **kwargs)
    try:
        assert len(tables) == 1
    except AssertionError:
        raise Exception("More than one <table> in provided HTML code!  Use from_html instead.")
    return tables[0]

##############################
# MAIN (TEST FUNCTION)       #
##############################
'''
def main():

    print("Generated using setters:")
    x = PrettyTable(["City name", "Area", "Population", "Annual Rainfall"])
    x.title = "Australian capital cities"
    x.sortby = "Population"
    x.reversesort = True
    x.int_format["Area"] = "04"
    x.float_format = "6.1"
    x.align["City name"] = "l" # Left align city names
    x.add_row(["Adelaide", 1295, 1158259, 600.5])
    x.add_row(["Brisbane", 5905, 1857594, 1146.4])
    x.add_row(["Darwin", 112, 120900, 1714.7])
    x.add_row(["Hobart", 1357, 205556, 619.5])
    x.add_row(["Sydney", 2058, 4336374, 1214.8])
    x.add_row(["Melbourne", 1566, 3806092, 646.9])
    x.add_row(["Perth", 5386, 1554769, 869.4])
    print(x)
    
    print
    
    print("Generated using constructor arguments:")
    
    y = PrettyTable(["City name", "Area", "Population", "Annual Rainfall"],
        title = "Australian capital cities",
        sortby = "Population",
        reversesort = True,
        int_format = "04",
        float_format = "6.1",
        max_width = 12,
        min_width = 4,
        align = "c",
        valign = "t")
    y.align["City name"] = "l" # Left align city names
    y.add_row(["Adelaide", 1295, 1158259, 600.5])
    y.add_row(["Brisbane", 5905, 1857594, 1146.4])
    y.add_row(["Darwin", 112, 120900, 1714.7])
    y.add_row(["Hobart", 1357, 205556, 619.5])
    y.add_row(["Sydney", 2058, 4336374, 1214.8])
    y.add_row(["Melbourne", 1566, 3806092, 646.9])
    y.add_row(["Perth", 5386, 1554769, 869.4])
    print(y)
    
if __name__ == "__main__":
    main()
'''
#########
#######
#####
###
##
#


#================================================================================
#================================================================================
# The code below define class Tunning : from tunning.py.  
#================================================================================
#================================================================================

import sys,os,json,datetime
import numpy as np 
import pandas as pd 
from scipy import stats
import xgboost as xgb

# 定义ks评分指标,供xgboost.train函数的feval调用
def ks_feval(preds,xgbtrain):
    label = xgbtrain.get_label()
    assert len(preds) == len(label)
    df = pd.DataFrame(data = np.array([preds,label]).T,columns = ['preds','label'])
    df_0,df_1 = df[df['label']<0.5],df[df['label']>=0.5]
    ks,ks_pvalue = stats.ks_2samp(df_0['preds'].values,df_1['preds'].values)
    return 'ks',ks

# 美化dataframe输出
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
    try:
        dfresult = pd.DataFrame({(dataset+'_'+feval): result[dataset][feval] 
                   for dataset in datasets for feval in ('auc','ks')})
    except:
        dfresult = pd.DataFrame({(dataset+'_'+feval): result[dataset][feval] 
                   for dataset in datasets for feval in ['ks']})
    
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
    
#########
#######
#####
###
##
#
    

#================================================================================
#================================================================================
# The code below carry out the tunning steps: from runtunning.py
#================================================================================
#================================================================================
#-*-coding:utf-8-*-
import sys,os,json,datetime
import numpy as np
import pandas as pd

# 定义对numpy浮点数和整数的json序列化类
class numpyJsonEncoder(json.JSONEncoder):
    def default(self, obj): 
        if isinstance(obj,(np.float,np.float32,np.float64)): 
            return float(obj)
        elif isinstance(obj, (np.int,np.int0,np.int8,np.int16,np.int32,np.int64)): 
            return int(obj)
        else: 
            return json.JSONEncoder.default(self, obj)

        
def main(dftrain,dftest,outputdir = outputdir,n_jobs = n_jobs,
         score_func = score_func, score_gap_limit = score_gap_limit,
         params_dict = params_dict):
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    # step0: 初始化
    tune = Tunning(dftrain,dftest,score_func = score_func,score_gap_limit = score_gap_limit, params_dict=params_dict,n_jobs=n_jobs)
    
    # step1: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step1: try relatively high learning_rate...')
    tune.gridsearch_cv(params_test1,cv = 5,verbose_eval = 10)
    
    # step2：
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step2: tune max_depth & min_child_weight...')
    tune.gridsearch_cv(params_test2,cv = 5,verbose_eval = 20)
    
    
    # step3：
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step3: tune gamma...')
    tune.gridsearch_cv(params_test3,cv = 5,verbose_eval = 20)
    
    
    # step4：
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step4: tune subsample & colsample_bytree...')
    tune.gridsearch_cv(params_test4,cv = 5,verbose_eval = 20)
    
    
    # step5: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step5: tune reg_alpha...')
    tune.gridsearch_cv(params_test5,cv = 5,verbose_eval = 20)
   
    
    # step6: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step6: tune reg_lambda...')
    tune.gridsearch_cv(params_test6,cv = 5,verbose_eval = 20)
    
    
    # step7: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step7: try relatively low learning_rate...')
    tune.gridsearch_cv(params_test7,cv = 5,verbose_eval = 20)
    
    # step8: 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('step8: train model with tuned parameters and fully train dataset...')
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    bst,dfimportance = tune.train_best()
   
    #generate results
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n================================================================================ %s\n'%nowtime)
    print('save results...\n\n ')
       
    with open(outputdir +'/best_parameters.json','w') as f:
        json.dump(tune.params_dict,f,cls = numpyJsonEncoder)
        
    tune.dfmerge.to_csv(outputdir + '/dfresults',sep = '\t',encoding = 'utf-8')
    try:
        tune.dfmerge.to_excel(outputdir + '/dfresults.xlsx',encoding = 'utf-8')
    except:
        pass
    
    bst.save_model(outputdir + '/bst.model')
    
    dfimportance.to_csv(outputdir + '/dfimportance',sep = '\t',encoding = 'utf-8')
    try:
        dfimportance.to_excel(outputdir + '/dfimportance.xlsx',encoding = 'utf-8')
    except:
        pass

    return(tune.params_dict)
    
if __name__ == '__main__':
    print('\ntask_name:%s'%task_name)
    dftrain = pd.read_csv(train_data_path,sep = '\t',encoding = 'utf-8')
    dftest = pd.read_csv(test_data_path,sep = '\t',encoding = 'utf-8')
    main(dftrain,dftest)  
    
####
###
##
#

