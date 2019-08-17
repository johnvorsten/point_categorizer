# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:58:12 2019

@author: z003vrzk
"""

from JVWork_AccuracyVisual import plt_accuracy, plt_accuracy2, plt_distance, import_error_dfs


[data1, data2, data3, data4, data5, data6] = import_error_dfs()
records = [data1, data2, data3, data4, data5, data6]

df = data1.dataframe
new_col = df['n_points'] + df['n_points']



plt_accuracy(records)
plt_accuracy2(records)
plt_distance([data1], sort=True, sort_on='correct_k')
plt_distance([data1],sort=True, sort_on='n_points', closest_meth=True)
for record in records:
    plt_accuracy2([record])