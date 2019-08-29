# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:58:12 2019

@author: z003vrzk
"""

from JVWork_AccuracyVisual import (plt_accuracy, plt_accuracy2, 
                                   plt_distance, import_error_dfs,
                                   ExtractLabels)
import numpy as np
import pandas as pd

records = import_error_dfs()

extract = ExtractLabels()
labels, hyper_dict = extract.calc_labels(records, r'D:\Z - Saved SQL Databases\44OP-093324_Baylor_Bric_Bldg\JobDB.mdf', best_n=3)





#plt_accuracy(records)
#plt_accuracy2(records)
#plt_distance([data1], sort=True, sort_on='correct_k')
#plt_distance([data1],sort=True, sort_on='n_points', closest_meth=True)
#for record in records:
#    plt_accuracy2([record])