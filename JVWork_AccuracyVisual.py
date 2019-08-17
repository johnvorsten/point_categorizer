# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 22:04:17 2019

@author: z003vrzk
"""

import pandas as pd
from collections import namedtuple
import numpy as np
import matplotlib.markers as markers
import matplotlib.pyplot as plt

class Record():
    """Keep track of individual dataframes and their related information"""
    
    def __init__(self, dataframe, parent_file, hyper_dict):
        self.dataframe = dataframe
        self.parent_file = parent_file
        self.hyper_dict = hyper_dict
        self.hyper_str = self.dict2str()
        self.accuracy_df = self.accuracy_simple()
        self.accuracy_df_log = self.accuracy_log()
        self.best_cols = self.best_metric()
        
    def accuracy_simple(self):
        
        non_opt_cols = ['DBPath', 'correct_k','n_points', 'n_len1', 
                        'n_len2', 'n_len3', 'n_len4',
                        'n_len5', 'n_len6', 'n_len7']

        accuracy_df = pd.DataFrame()
        for opt_k in set(self.dataframe.columns).difference(set(non_opt_cols)):
            try:
                accuracy_simple = ((self.dataframe.correct_k - self.dataframe[opt_k])
                /self.dataframe.correct_k)
                loc = len(accuracy_df.columns)
                accuracy_df.insert(loc=loc, column=opt_k, value=accuracy_simple)
            except:
                continue
        
        return accuracy_df
    
    def accuracy_log(self):
        
        non_opt_cols = ['DBPath', 'correct_k','n_points', 'n_len1', 
                        'n_len2', 'n_len3', 'n_len4',
                        'n_len5', 'n_len6', 'n_len7']
        accuracy_df = pd.DataFrame()
        for opt_k in set(self.dataframe.columns).difference(set(non_opt_cols)):
            try:
                accuracy_simple = ((self.dataframe.correct_k - self.dataframe[opt_k])
                /self.dataframe.correct_k)
                accuracy_log = np.log(abs(accuracy_simple)+1)
                loc = len(accuracy_df.columns)
                accuracy_df.insert(loc=loc, column=opt_k, value=accuracy_log)
            except:
                continue
        
        return accuracy_df
    
    def dict2str(self):
        x = str()
        for key, value in self.hyper_dict.items():
            x = x + str(value) + '\n'
        return x
    
    def best_metric(self):
        sequence_tag = 'DBPath'
        non_opt_cols = ['DBPath', 'correct_k','n_points', 'n_len1', 
                        'n_len2', 'n_len3', 'n_len4',
                        'n_len5', 'n_len6', 'n_len7']
        Y = pd.DataFrame()
        for col in set(self.dataframe.columns).difference(set(non_opt_cols)):
            diff = self.dataframe['correct_k'] - self.dataframe[col]
            Ysub = pd.DataFrame(diff, columns=[col])
            Y = Y.join(Ysub, how='outer')
        #Find abs(argmin()) of all rows
        cols = np.argmin(abs(Y.values), axis=1)
        cols_name = [Y.columns[col] for col in cols]
        cols_df = pd.DataFrame(cols_name, index=self.dataframe[sequence_tag])
        return cols_df
        

def import_error_dfs():
    """Imports error_df csv files and converts to custom Record objects
    Returns
    -------
    [data1, data2, data3, data4, data5,data6]"""
    
    error_df_list = [r"C:\Users\z003vrzk\.spyder-py3\Scripts\ML\Point database categorizer\error_df.csv",
                     r"C:\Users\z003vrzk\.spyder-py3\Scripts\ML\Point database categorizer\error_df 8-7 (recalc2, by_size=False).csv",
                     r"C:\Users\z003vrzk\.spyder-py3\Scripts\ML\Point database categorizer\error_df 8-6 (recalc1).csv"
                     ]
    
    error_df_0 = pd.read_csv(error_df_list[0], header=0, index_col=0)
    error_df_1 = pd.read_csv(error_df_list[1], header=0, index_col=0)
    error_df_2 = pd.read_csv(error_df_list[2], header=0, index_col=0)
    
    reduce_cols = ['DBPath', 'correct_k', 'optk_MDS_gap_max', 'optk_MDS_gap_Tib',
           'optk_MDS_gap*_max', 'n_points', 'n_len1', 'n_len2', 'n_len3', 'n_len4',
           'n_len5', 'n_len6', 'n_len7']
    standard_cols = ['DBPath', 'correct_k', 'optk_X_gap_max', 'optk_X_gap_Tib',
           'optk_X_gap*_max', 'n_points', 'n_len1', 'n_len2', 'n_len3', 'n_len4',
           'n_len5', 'n_len6', 'n_len7']
    
    #Possible hyperparameter trees
    hyper_dict = {'by_size':[True, False],
                  'method':['kmeans','agglomerative'],
                  'distance':['eucledian', 'minkowski'],
                  'reduce':['MDS', 'TSNE', False],
                  }
    #Actual Hyperparameters, by order in error_df_list
    hyper_1 = {'by_size':True,
                  'method':'kmeans',
                  'distance':'eucledian',
                  'reduce':False,
                  }
    dataset_1 = error_df_0[standard_cols]
    
    hyper_2 = {'by_size':True,
                  'method':'kmeans',
                  'distance':'eucledian',
                  'reduce':'MDS',
                  }
    dataset_2 = error_df_0[reduce_cols]
    
    hyper_3 = {'by_size':True,
                  'method':'kmeans',
                  'distance':'eucledian',
                  'reduce':False,
                  }
    dataset_3 = error_df_2[standard_cols]
    
    hyper_4 = {'by_size':True,
                  'method':'kmeans',
                  'distance':'eucledian',
                  'reduce':'MDS',
                  }
    dataset_4 = error_df_2[reduce_cols]
    
    hyper_5 = {'by_size':False,
                  'method':'kmeans',
                  'distance':'eucledian',
                  'reduce':False,
                  }
    dataset_5 = error_df_1[standard_cols]
    
    hyper_6 = {'by_size':False,
                  'method':'kmeans',
                  'distance':'eucledian',
                  'reduce':'MDS',
                  }
    dataset_6 = error_df_1[reduce_cols]
    
    data1 = Record(dataset_1, error_df_list[0], hyper_1)
    data2 = Record(dataset_2, error_df_list[0], hyper_2)
    data3 = Record(dataset_3, error_df_list[2], hyper_3)
    data4 = Record(dataset_4, error_df_list[2], hyper_4)
    data5 = Record(dataset_5, error_df_list[1], hyper_5)
    data6 = Record(dataset_6, error_df_list[1], hyper_6)
    
    return [data1, data2, data3, data4, data5, data6]


def plt_accuracy(records):
    """parameters
    -------
    records : iterable of Record objects
    """
    global rect
    fig, ax = plt.subplots(1)
    tags = np.arange(len(records))
    hypers = {}
    
    for tag in tags:
        x = str()
        for key, value in records[tag].hyper_dict.items():
            x = x + str(value) + '\n'
        hypers[tag] = x
    accuracies = {}
    
    for tag in tags:
        accuracies[tag] = {}
        for col in records[tag].accuracy_df.columns:
            y = np.mean(abs(records[tag].accuracy_df.loc[:,col]))
            accuracies[tag][col] = y
            
    ind = np.arange(len(records))
    width = 0.2
    rects = []
    labels = []
    
    for tag in tags:
        for idx, (col, accuracy) in enumerate(accuracies[tag].items()):
            rect = ax.bar(ind[tag] + width*idx, accuracy, width) ##
            rects.append(rect)
            labels.append(col)
    
    ax.set_ylabel('Hyperparameter sets')
    ax.set_title('Hyperparameter sets')
    ax.set_xticks(ind + width)
    ax.set_xticklabels([hypers[tag] for tag in tags])
    
    def autolabel(bar_objs, labels):
        
        for bar_obj, label in zip(bar_objs, labels):
            for rect in bar_obj:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        label,
                        ha='center', va='bottom')
            
    autolabel(rects, labels)
    plt.show()

def plt_accuracy2(records):
    """parameters
    -------
    records : iterable of Record objects
    """
    sequence_tag = 'DBPath'
    fig, ax = plt.subplots(1)
    
    cols_set = set() #For marker type
    dbs_set = set() #For x indicies
    hyper_set = set()
    marker_keys = list(markers.MarkerStyle.markers.keys())
    #Get unique list of all column names
    for record in records:
        col_set = set(record.accuracy_simple().columns)
        cols_set = cols_set.union(col_set)
        db_set = set(record.dataframe[sequence_tag])
        dbs_set = dbs_set.union(db_set)
        hyper_set.add(record.hyper_str)
        
    dbs_list = list(dbs_set)
    cols_list = list(cols_set)
    hyper_list = list(hyper_set)
    colors = [np.array(plt.cm.hsv(i/float(len(hyper_list)))).reshape(1,-1) for i in range(len(hyper_list)+1)]
    marker_list = [marker_keys[i] for i in range(len(cols_list)+1)]
    
    for idx, record in enumerate(records):
        #Get X, get Y, get marker, get color
        Xs = [dbs_list.index(name) for name in record.dataframe[sequence_tag]]
        for col in record.accuracy_df.columns:
            Ys = record.accuracy_simple()[col]
            ax.scatter(Xs, Ys, s=5,
                       c=colors[hyper_list.index(record.hyper_str)], 
                       marker=marker_list[cols_list.index(col)])
    
    ax.set_ylim(-2,2)
    ax.set_ylabel('Accuracy simple')
    ax.set_xlabel('Dataset -> integer')
    ax.set_title('Accuracies')
    ax.grid(True)
    ax.legend()

def plt_distance(records, sort=True, sort_on='correct_k', closest_meth=False):
    """Plot dataset versus distance between correct_k and optimal_k
    parameters
    -------
    records : iterable of Record objects
    sort : sort by database
    sort_on : key in record.dataframe to sort on. Must be one of 'correct_k',
    'n_lenn', 'n_points' or any column of the dataframe
    """
    global dbs_dict
    sequence_tag = 'DBPath'
    fig, ax = plt.subplots(1)
    
    cols_set = set() #For marker type
    dbs_dict = {} #For x indicies
    hyper_set = set() #For legend
    marker_keys = list(markers.MarkerStyle.markers.keys())
    #Get unique list of all column names, databases
    for record in records:
        col_set = set(record.accuracy_simple().columns)
        cols_set = cols_set.union(col_set)
        dbs = record.dataframe.set_index(record.dataframe[sequence_tag])
        db_dict = dbs.to_dict()
        for key, value in db_dict.items():
            try:
                dbs_dict[key].update(value)
            except:
                dbs_dict[key] = value
        hyper_set.add(record.hyper_str)
    
    if sort:
        dbs_list = sorted(dbs_dict[sort_on], key=dbs_dict[sort_on].get)
    else:
        dbs_list = list(dbs_dict[sequence_tag].keys())
    cols_list = list(cols_set)
    hyper_list = list(hyper_set)
    colors = [np.array(plt.cm.hsv(i/float(len(hyper_list)))).reshape(1,-1) for i in range(len(hyper_list)+1)]
    marker_list = [marker_keys[i] for i in range(len(cols_list)+1)]
    
    Y_record = []
    for idx, record in enumerate(records):
        #Get X, get Y, get marker, get color
        Xs = [dbs_list.index(name) for name in record.dataframe[sequence_tag]]
        if closest_meth:
            Y = pd.DataFrame()
            for col in record.accuracy_simple().columns:
                #Plot closest distances
                diff = record.dataframe['correct_k'] - record.dataframe[col]
                Ysub = pd.DataFrame(diff, columns=[col])
                Y = Y.join(Ysub, how='outer')
            #Find abs(argmin()) of all rows
            cols = np.argmin(abs(Y.values), axis=1)
            Ys = Y.values[np.arange(len(Y)),cols]
            Y_record.append(list(Ys))
            ax.scatter(Xs, Ys,
                       s=5,
                       c=colors[hyper_list.index(record.hyper_str)],
                       marker=marker_list[0])
        else:
            for col in record.accuracy_simple().columns:
                #Plot Distances
                Ys = record.dataframe['correct_k'] - record.dataframe[col]
                Y_record.append(list(Ys))
                ax.scatter(Xs, Ys, 
                           s=5,
                           c=colors[hyper_list.index(record.hyper_str)], 
                           marker=marker_list[cols_list.index(col)])
    
    def get_limits(values, symmetric_z):
        """symmetric_z : value used for z score cutoff"""
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        Z = (values - mean) / std
        idx_high = (np.abs(Z - symmetric_z)).argmin()
        idx_low = (np.abs(Z + symmetric_z)).argmin()
        return (values[idx_low], values[idx_high])
    
    Y_record = [item for sublist in Y_record for item in sublist]
    ax.set_ylim(get_limits(Y_record, 1.65))
    ax.set_ylabel('Distance simple')
    ax.set_xlabel('Dataset -> integer')
    ax.set_title('Distance')
    ax.grid(True)
    ax.legend()






