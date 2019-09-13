# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 22:04:17 2019

#Example Usage

#Import required modules
from JVWork_AccuracyVisual import (plt_accuracy, plt_accuracy2, 
                                   plt_distance, import_error_dfs)
import numpy as np
import pandas as pd

#Import records
records = import_error_dfs()

#Fun plotting functions
#plot all datasets (bar graph hyperparameter set v accuracy)
plt_accuracy([records[1]])

#plot a single dataset (bar graph hyperparameter set v accuracy)
plt_accuracy([records[1]], instance_tag=r'D:\Z - Saved SQL Databases\44OP-093324_Baylor_Bric_Bldg\JobDB.mdf')

plt_accuracy2([records[1]])

#Sort the x axis on correct number of clusters
plt_distance([records[1]], sort=True, sort_on='correct_k')

#Sort the x axis on number of database points & only plot the closest optimal k index (best prediction)
plt_distance([records[1]], sort=True, sort_on='n_points', closest_meth=True)

for record in records:
    plt_accuracy2([record])
    
#%% Relationship hyperparameters v. loss plotting

# Third party imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import OneHotEncoder

# Local Imports
from JVWork_AccuracyVisual import import_error_dfs
from JVWork_AccuracyVisual import plt_hyperparameters
from JVWork_Labeling import ExtractLabels
Extract = ExtractLabels()


# Import your database (.csv)
csv_file = r'data\master_pts_db.csv'
sequence_tag = 'DBPath'

# Get unique names
unique_tags = pd.read_csv(csv_file, index_col=0, usecols=['DBPath'])
unique_tags = list(set(unique_tags.index))

# Calculate best hyperparameters
# Labels is a nested dictionary of all hyperparameters ranked best to worst
tag = unique_tags[0]
records = import_error_dfs()
labels, hyper_dict = Extract.calc_labels(records, tag, best_n='all')

# Plot 'by_size' versus position
plt_hyperparameters(list(labels.values()), 
                    hyper_param_field = 'by_size', 
                    hyper_param_value=False, 
                    plt_density=True, 
                    plt_values=False)

# Plot 'clusterer' versus position
plt_hyperparameters(list(labels.values()), 
                    hyper_param_field = 'clusterer', 
                    hyper_param_value=False, 
                    plt_density=True, 
                    plt_values=False)

# Plot 'n_components' versus position
plt_hyperparameters(list(labels.values()), 
                    hyper_param_field = 'n_components', 
                    hyper_param_value=False, 
                    plt_density=True, 
                    plt_values=False)

# Plot 'reduce' versus position
plt_hyperparameters(list(labels.values()), 
                    hyper_param_field = 'reduce', 
                    hyper_param_value=False, 
                    plt_density=True, 
                    plt_values=False)

# Plot 'index' versus position
plt_hyperparameters(list(labels.values()), 
                    hyper_param_field = 'index', 
                    hyper_param_value=False, 
                    plt_density=True, 
                    plt_values=False)

@author: z003vrzk
"""

import pandas as pd
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.markers as markers
import matplotlib.pyplot as plt
import os
from path import Path
import re

class Record():
    """Keep track of individual dataframes and their related information
    parameters
    -------
    dataframe : a dataframe containing error metric information. 
    See import_error_dfs()
    parent_file : original csv file
    hyper_dict : a dictionary on the predicted sets hyperparameters. For example
    {'hyper1':value1, [...]}"""
    
    def __init__(self, dataframe, parent_file, hyper_dict):
        self.dataframe = dataframe
        self.parent_file = parent_file
        self.hyper_dict = dict(sorted(hyper_dict.items())) #TODO Change
        self.hyper_str = self.dict2str()
        self.accuracy_df = self.accuracy_simple()
        self.accuracy_df_log = self.accuracy_log()
        self.best_cols = self.best_metric()
        
    def accuracy_simple(self):
        """Calculate hte l2 loss for a set of optimal k predictions"""
        
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
            except: #TODO handle exception
                continue
        
        return accuracy_df
    
    def accuracy_log(self):
        """Calculate the log l2 loss for a set of optimal k predictions"""
        
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
            except: #TODO handle exception
                continue
        
        return accuracy_df
    
    def dict2str(self):
        """Convert the record hyperparameter into a string for storing in
        dictionary keys"""
        
        x = str()
        for key, value in self.hyper_dict.items():
            x = x + str(key) + ':' + str(value) + '\n'
        return x
    
    def best_metric(self):
        """This function is really only useful for plotting. Keeping for 
        possible future uses only"""
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
    
    #Possible values, for notes
    hyper_dict = {'by_size':[True, False],
                  'clusterer':['kmeans','agglomerative'],
                  'distance':['eucledian', 'minkowski'],
                  'reduce':['MDS', 'TSNE', False],
                  'n_components':[2,5,10]
                  }
    def get_error_dfs(base_dir=r'error_dfs'):
        #regex
        hyper_pattern = r'hyper'
        std_pattern = 'error_df.csv'
        
        files = os.listdir(base_dir)
        for idx, _file in enumerate(files):
            new_file = base_dir + '\\' + _file
            files[idx] = new_file
        
        new_files = []
        for idx, _file in enumerate(files):
            is_standard = bool(re.search(std_pattern, _file))
            is_hyper = bool(re.search(hyper_pattern, _file))
            if not any((is_standard, is_hyper)):
                new_files.append(_file)
#                print('popped {} at {}'.format(_file, idx))
                
        return new_files
    
    def get_hyper_dict(df_path):
        #Get directory files same as df_path
        #Find the file that matches df_path and has hyper
        #read to dataframe
        #return as dictionary
        csvpath = Path(df_path)
        parts = csvpath.splitpath()
        base_dir = parts[0]
        name = parts[1]

        hyper_path = list(base_dir.glob('*' + name[:-4] + '*' + 'hyper' + '*'))
        if len(hyper_path) == 1:
            hyper_path = hyper_path[0]
        else:
            print('Error with hyper_path : {}'.format(hyper_path))
            print('Error with name : {}'.format(name))
            print(len(hyper_path))
            print('Error with df_path : {}'.format(df_path))
            raise LookupError('Hyperparameter file matches multiple possibilities')
            
        hyper_df = pd.read_csv(hyper_path, index_col=0)
        
        hyper_dict = {'by_size':hyper_df.loc[0, 'by_size'],
                      'clusterer':hyper_df.loc[0, 'clusterer'],
                      'distance':hyper_df.loc[0, 'distance'],
                      'reduce':hyper_df.loc[0, 'reduce'],
                      'n_components':hyper_df.loc[0, 'n_components']}
        
        return hyper_dict
    
    error_dfs = []
    records = []
    error_df_list = get_error_dfs()
    for idx, error_df_path in enumerate(error_df_list):
        error_df = pd.read_csv(error_df_path, header=0, index_col=0)
        error_dfs.append(error_df)
        hyper_dict = get_hyper_dict(error_df_path)
        record = Record(error_df, error_df_path, hyper_dict)
        records.append(record)
    
    return records


def plt_accuracy(records, 
                 column_tag='DBPath', 
                 instance_tag=None):
    """parameters
    -------
    records : iterable of Record objects
    output
    -------
    A bar graph showing the hyperparameters on the x axis, and the simple accuracy of each
    clustering index on the y axis. This is useful for seeing how well a clustering index
    performs on a dataset. Multiple hyperparameter sets can be graphed, but
    the plot quickly becomes crowded
    """
    fig, ax = plt.subplots(1)
    tags = np.arange(len(records))
    hypers = {}
    
    #Create all unique x labels from hyperparameter sets
    for tag in tags:
        x = str()
        for key, value in records[tag].hyper_dict.items():
            x = x + str(value) + '\n'
        hypers[tag] = x
    
    #Calculate accuracies on each optimal_k index for the hyperparameter set
    accuracies = {}
    for tag in tags:
        accuracies[tag] = {}
        for col in records[tag].accuracy_df.columns:
            if instance_tag is None:
                y = np.mean(abs(records[tag].accuracy_df.loc[:,col]))
            else:
                df = records[tag].dataframe
                idx = df[df[column_tag]==instance_tag].index
                y = np.mean(abs(records[tag].accuracy_df.loc[idx, col]))
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
    pass

def plt_accuracy2(records):
    """parameters
    -------
    records : iterable of Record objects
    output
    -------
    A plot showing each dataset on the x axis, and the simple accuracy of each
    clustering index on the y axis. This is useful for seeing how error is distributed
    relative to the "correct_k". Simple accuracy (y axis) values of 1 indicate
    the clustering index underestimated the number of clusters 
    (correct_k - optimal_k) / (correct_k) ~ 1
    Negative Simple accuracy (y axis) values indicate the clustering index overestimated 
    the number of clusters (correct_k - optimal_k) / (correct_k) < 0
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

def plt_hyperparameters(label_list, 
                        hyper_param_field, 
                        hyper_param_value=None,
                        plt_density=True,
                        plt_values=False):
    """Plot a set of ordered hyper parameter arrays. 
    Inputs
    -------
    label_list : A list of hyperparameter dictiodnaries, ranked by order of their 
    effectiveness / accuracy / loss
    hyper_param_field : the general name of the hyperparameter you want to plot
    hyper_param_value : (optional) the specific hyperparameter value relating
    to hyper_param_field to plot
    plt_density : plot the values using a gaussian density kernel probability 
    density function (kernel density estimation)
    plt_values : plot the actual values of the hyperparameter. Not useful in string
    or non-plottable data types"""
    
    # Create X axis for plotting
    X = np.arange(len(label_list))

    # Extract only the specific hyper parameter from the dictionaries in label_list
    y = [dictionary[hyper_param_field] for dictionary in label_list]
    y = np.array(y)
    feature_names = list(set(y))
    
    # Legacy
#    y_onehot = one_hot.fit_transform(y.reshape(-1,1)).toarray()
    
#    for y_col, name in zip(np.rollaxis(y_onehot, 1), one_hot.get_feature_names()):
#        if hyper_param_value and not name==hyper_param_value:
#            continue
#        plt.scatter(X, y_col, label=name)
        
    for name in feature_names:
        if plt_density:
            kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X[y==name].reshape(-1,1))
            log_density = kde.score_samples(X.reshape(-1,1))
            density = np.exp(log_density)
            plt.plot(X, density, label=str(name) + ' kde')
            
        if plt_values:
            plt.scatter(X[y==name], y[y==name], label=name)
        
        
    plt.title(f'Accuracy ranking v. {hyper_param_field}')
    plt.xlabel('Accuracy Ranking (sequential)')
    plt.ylabel(f'{hyper_param_field}')
    plt.legend()
    pass


def plt_loss_curve(labels):
    """Plot the curve of losses of hyperparameter rankings
    input
    -------
    labels : a list of labels calculated from ExtractLabels.calc_labels()
    output
    -------
    A graph of ranked hyperparamter set losses. The curve will probably
    show elbows of loss, indicating a desirable set of hyperparameters that
    minimize the error in estimating optimal_k
    
    Example usage :
    #Import your database (.csv)
    csv_file = r'data\master_pts_db.csv'
    sequence_tag = 'DBPath'
    
    #Get unique names
    unique_tags = pd.read_csv(csv_file, index_col=0, usecols=['DBPath'])
    unique_tags = list(set(unique_tags.index))
    
    tag_iter = iter(unique_tags)
    tag = next(tag_iter)
    labels, hyper_dict = Extract.calc_labels(records, tag, best_n='all')
    
    ## OR ## With MongoDB ##
    a = collection.find_one({'database_tag':tag}) # Or iterate over documents
    db_dict = next(a)
    labels = db_dict['hyper_labels']
    
    # Plot
    plt_loss_curve(labels)
    
    
    """
    
    X = list(labels.keys())
    Y = [_label['loss'] for _key, _label in labels.items()]
    Yd1 = np.gradient(Y)
    Yd2 = np.gradient(Yd1)
    
    plt.figure(1)
    plt.plot(X, Y, label='loss')
    plt.title('loss curve')
    plt.xlabel('Ranking')
    plt.ylabel('loss')
    
    plt.figure(2)
    plt.plot(X, Yd1, label='1st diff')
    plt.ylim((-50,100))
    plt.title('loss curve (1st diff)')
    plt.xlabel('Ranking')
    plt.ylabel('loss (1st diff)')
    
    plt.figure(3)
    plt.plot(X, Yd2, label='2nd diff')
    plt.ylim((-50,50))
    plt.title('loss curve (2nd diff)')
    plt.xlabel('Ranking')
    plt.ylabel('loss (2nd diff)')
    
    pass




