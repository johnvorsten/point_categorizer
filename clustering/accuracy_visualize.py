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
# Python imports
import os
import sys
from pathlib import Path
import re
from collections import namedtuple

# Third party imports
import pandas as pd
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.markers as markers
import matplotlib.pyplot as plt
import sqlalchemy

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

from extract import extract
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev, Customers,
                                              ClusteringHyperparameter, Labeling)

Insert = extract.Insert(server_name='.\DT_SQLEXPR2008',
                        driver_name='SQL Server Native Client 10.0',
                        database_name='Clustering')


#%%

class Record():
    """Keep track of individual dataframes and their related information
    parameters
    -------
    dataframe : a dataframe containing error metric information.
    See import_error_dfs()
    parent_file : original csv file
    hyper_dict : a dictionary on the predicted sets hyperparameters. For example
    {'hyper1':value1, [...]}"""

    def __init__(self, indicies_dictionary, hyperparameter_dictionary):
        """Calculations on predicted number of clusters
        inputs
        -------
        dataframe : (pd.DataFrame) of clustering metrics/indicies
        hyper_dict : (dict) of hyperparameters used to calculate indicies"""
        self.indicies_dictionary = indicies_dictionary
        self.hyper_dict = hyperparameter_dictionary


    @staticmethod
    def accuracy_simple(predictions, correct_k):
        """Calculate hte l2 loss for a set of optimal k predictions
        inputs
        -------
        predictions : (dict) where key is the indicy metric, and value is the
        predicted number of clusters with that metric
        correct_k : (int) correct number of clusters
        outputs
        -------
        accuracy_dict : (dict) where key is the indicy metric, and value
        is a namedtuple of original predicted number of clusters and
        simple accuracy"""

        accuracy = namedtuple('accuracy',['prediction','accuracy_simple'])

        indicies = ['KL','CH','Hartigan','CCC','Marriot','TrCovW'
                    'TraceW','Friedman','Rubin','Cindex','DB','Silhouette',
                    'Duda','PseudoT2','Beale','Ratkowsky','Ball','PtBiserial',
                    'Frey','McClain','Dunn','Hubert','SDindex','Dindex','SDbw',
                    'gap_tib','gap_star','gap_max','Scott']

        accuracy_dict = {}

        for indicy in indicies:
            try:
                accuracy_simple = (correct_k - predictions[indicy]) / correct_k
                acc = accuracy(predictions[indicy], accuracy_simple)
                accuracy_dict[indicy] = acc
            except (KeyError, TypeError):
                # Key doesent exist or key value is None
                acc = accuracy(None, None)
                accuracy_dict[indicy] = acc
                continue

        return accuracy_dict

    @staticmethod
    def accuracy_log(predictions, correct_k):
        """Calculate the log l2 loss for a set of optimal k predictions"""

        accuracy = namedtuple('accuracy',['prediction','accuracy_log'])

        indicies = ['KL','CH','Hartigan','CCC','Marriot','TrCovW'
                    'TraceW','Friedman','Rubin','Cindex','DB','Silhouette',
                    'Duda','PseudoT2','Beale','Ratkowsky','Ball','PtBiserial',
                    'Frey','McClain','Dunn','Hubert','SDindex','Dindex','SDbw',
                    'gap_tib','gap_star','gap_max','Scott']

        accuracy_dict = {}

        for indicy in indicies:
            try:
                accuracy_simple = (correct_k - predictions[indicy]) / correct_k
                accuracy_log = np.log(abs(accuracy_simple)+1)
                acc = accuracy(predictions[indicy], accuracy_log)
                accuracy_dict[indicy] = acc
            except (KeyError, TypeError):
                acc = accuracy(None, None)
                accuracy_dict[indicy] = acc
                continue

        return accuracy_dict

    @staticmethod
    def dict2str(hyperparameters):
        """Convert the record hyperparameter into a string for storing in
        dictionary keys"""

        hyperparameters = dict(sorted(hyperparameters.items()))
        x = str()
        for key, value in hyperparameters.items():
            x = x + str(key) + ':' + str(value) + '\n'
        return x

    @staticmethod
    def best_metric(predictions):
        """This function is really only useful for plotting. Keeping for
        possible future uses only"""

        accuracy = namedtuple('accuracy',['prediction','accuracy'])

        indicies = ['KL','CH','Hartigan','CCC','Marriot','TrCovW'
                    'TraceW','Friedman','Rubin','Cindex','DB','Silhouette',
                    'Duda','PseudoT2','Beale','Ratkowsky','Ball','PtBiserial',
                    'Frey','McClain','Dunn','Hubert','SDindex','Dindex','SDbw',
                    'gap_tib','gap_star','gap_max','Scott']

        accuracy_dict = {}

        for indicy in indicies:
            try:
                accuracy = abs(correct_k - predictions[indicy])
                acc = accuracy(predictions[indicy], accuracy)
                accuracy_dict[indicy] = acc
            except KeyError:
                print('Key {} not found in predictions'.format(indicy))
                acc = accuracy(None, None)
                accuracy_dict[indicy] = acc
                continue

        minimum = min(accuracy.items(), key=lambda x : abs(x.accuracy))

        return minimum

    def __repr__(self):
        return "Record<id=%r,customer_id=%r,hyperparameter_id=%r>" % \
            (self.indicies_dictionary['id'],
             self.indicies_dictionary['customer_id'],
             self.indicies_dictionary['hyperparameter_id'])



def import_error_dfs(base_dir):
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

    error_dfs = []
    records = []
    error_df_list = get_error_dfs(base_dir)
    for idx, error_df_path in enumerate(error_df_list):
        error_df = pd.read_csv(error_df_path, header=0, index_col=0)
        error_dfs.append(error_df)
        hyper_dict = get_hyper_dict(error_df_path)
        record = Record(error_df, error_df_path, hyper_dict)
        records.append(record)

    return records


def get_error_dfs(base_dir):
    """Search for a specific file name in a directory"""

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

    return new_files


def get_hyper_dict(df_path):
    """Get directory files same as df_path
    Find the file that matches df_path and has hyper
    read to dataframe
    return as dictionary"""

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



def get_records(primary_keys):
    """
    inputs
    -------
    primary_keys : (list) of integers representing SQL primary keys OR
        (str) 'all' representing you want the whole Table. Primary keys are
        on the 'clustering' table. The function retruns the related
        clustering hyperparameters for each row in 'clustering'
    outputs
    -------
    records : (list) of Record objects (see above)"""

    if primary_keys == 'all':
        # Query whole database
        sel_indicies = sqlalchemy.select([Clustering])
        sel_hyperparams = sqlalchemy.select([ClusteringHyperparameter])
        indicies = Insert.core_select_execute(sel_indicies)
        hyperparameters = Insert.core_select_execute(sel_hyperparams)
    else:
        sel_indicies = sqlalchemy.select([Clustering]).where(Clustering.id.in_(primary_keys))
        indicies = Insert.core_select_execute(sel_indicies)
        foreign_keys = [indicy.hyperparameter_id for indicy in indicies]
        sel_hyperparams = sqlalchemy.select([ClusteringHyperparameter]).where(ClusteringHyperparameter.id.in_(foreign_keys))
        hyperparameters = Insert.core_select_execute(sel_hyperparams)

    # Construct records
    records = []
    for indicy in indicies:
        # Look for the hyperparameter primary key that matches the indicy foreign key
        for hyperparam in hyperparameters:
            if hyperparam.id == indicy.hyperparameter_id:
                _hyperparam = dict(hyperparam)
                _hyperparam.pop('id')
        # Construct record
        record = Record(indicies_dictionary=dict(indicy),
                        hyperparameter_dictionary=dict(_hyperparam))
        records.append(record)

    return records



def plt_indicy_accuracy_bar(records,
                 customer_id=None):
    """parameters
    -------
    records : (iterable) of one record object. Only one object may be passed
        because the graph gets crowded
    output
    -------
    A bar graph showing the hyperparameters on the x axis, and the simple
    accuracy of each clustering index on the y axis. This is useful for
    seeing how well a clustering index performs on a dataset.
    Multiple hyperparameter sets can be graphed, but the plot quickly becomes
    crowded
    inputs
    -------
    records : (list) list of Record objects
    column_tag : (str) name of column in dataframe that defines the customer name
    instance_tag : (str) customer name. Only plot indicies of a specific customer
    """
    fig, ax = plt.subplots(1)
    tags = np.arange(len(records))
    hypers = {}

    #Create all unique x labels from hyperparameter sets
    for tag in tags:
        hypers[tag] = records[tag].dict2str(records[tag].hyper_dict)

    #Calculate accuracies on each optimal_k index for the hyperparameter set
    """
    accuracies = {tag1 : {index1:accuracy1,index2:accuracy2},
                  tag2 : {index1:accuracy1,index2:accuracy2}, ...}"""

    accuracies = {}
    for tag in tags:
        record = records[tag]

        if customer_id is None:
            # Accuracy is a tuple of (prediction, accuracy)
            accuracy = record.accuracy_simple(record.indicies_dictionary,
                                              record.indicies_dictionary['correct_k'])
            accuracies[tag] = accuracy
        elif customer_id == record.indicies_dictionary['customer_id']:
            # Accuracy is a tuple of (prediction, accuracy)
            accuracy = record.accuracy_simple(record.indicies_dictionary,
                                              record.indicies_dictionary['correct_k'])
            accuracies[tag] = accuracy
        else:
            continue

    ind = np.arange(len(records))
    width = 0.15
    rects = []
    labels = []

    # Plot each hyperparameter set
    for tag in tags:
        # Plot each indicy accuracy
        for idx, (indicy, accuracy) in enumerate(accuracies[tag].items()):
            if accuracy.accuracy_simple is None:
                continue
            else:
                rect = ax.bar(ind[tag] + width*idx, accuracy.accuracy_simple, width)
                rects.append(rect)
                labels.append(indicy)

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

    return None



def plt_indicy_accuracy_scatter(records,
                                indicy_filter=['KL','CH','Hartigan',
                                               'CCC','Scott','Marriot']):
    """parameters
    -------
    records : iterable of Record objects
    indicy_filter : (iterable) of strings representing indicies to display on the
        plot. If not in this list then their accuracy is not displayed
    output
    -------
    A plot showing each dataset on the x axis, and the simple accuracy of each
    clustering index on the y axis. This is useful for seeing how error is
    distributed relative to the "correct_k".

    Simple accuracy (y axis) values of 1 indicate
    the clustering index underestimated the number of clusters
    (correct_k - optimal_k) / (correct_k) ~ 1

    Negative Simple accuracy (y axis) values indicate the clustering index
    overestimated the number of clusters
    (correct_k - optimal_k) / (correct_k) < 0
    """
    fig, ax = plt.subplots(1)
    indicies = ['KL','CH','Hartigan','CCC','Marriot','TrCovW'
            'TraceW','Friedman','Rubin','Cindex','DB','Silhouette',
            'Duda','PseudoT2','Beale','Ratkowsky','Ball','PtBiserial',
            'Frey','McClain','Dunn','Hubert','SDindex','Dindex','SDbw',
            'gap_tib','gap_star','gap_max','Scott']

    if indicy_filter is None:
        pass # Do not change indicies to be plotted
    elif hasattr(indicy_filter, '__iter__'):
        for indicy in indicy_filter:
            assert indicies.__contains__(indicy), ('{} is not in available'+
               ' list of indicies {}'.format(indicy, indicies))
        indicies = indicy_filter

    customer_id_set = set() # For x indicies
    hyperparameter_set = set()
    marker_keys = list(markers.MarkerStyle.markers.keys())

    #Get unique list of all column names
    for record in records:
        customer_id_set.add(record.indicies_dictionary['customer_id'])
        hyperparameter = record.dict2str(record.hyper_dict)
        hyperparameter_set.add(hyperparameter)

    customer_ids = list(customer_id_set)
    hyperparameters = list(hyperparameter_set)
    colors = [np.array(plt.cm.hsv(i/float(len(hyperparameters))))\
              .reshape(1,-1) for i in range(len(hyperparameters)+1)]
    marker_list = [marker_keys[i] for i in range(len(indicies)+1)]

    Xs = np.arange(len(customer_ids))
    for idx, record in enumerate(records):
        #Get Y, marker, color
        accuracies = record.accuracy_simple(record.indicies_dictionary,
                                            record.indicies_dictionary['correct_k'])
        x = Xs[customer_ids.index(record.indicies_dictionary['customer_id'])]
        hyperparameter = record.dict2str(record.hyper_dict)
        for indicy in indicies:
            y = accuracies[indicy].accuracy_simple
            ax.scatter(x, y, s=30,
                       c=colors[hyperparameters.index(hyperparameter)],
                       marker=marker_list[indicies.index(indicy)],
                       label=indicy)

    ax.set_ylim(-2,2)
    ax.set_ylabel('Accuracy simple')
    ax.set_xlabel('Dataset')
    ax.set_title('Accuracies')
    ax.grid(True)
    ax.legend(indicies)

    return None



def plt_distance(records,
                 sort_on='correct_k',
                 indicy_filter=['KL','CH','Hartigan','CCC','Scott','Marriot']):
    """Plot dataset versus distance between correct_k and optimal_k
    parameters
    -------
    records : (iterable) of Record objects
    sort_on : (str) or None, key in record.dataframe to sort on.
        Must be one of ['correct_k', 'n_points', None]
    indicy_filter : (iterable) of strings representing indicies to display on the
        plot. If not in this list then their accuracy is not displayed
    """
    assert sort_on in ['correct_k','n_points', None], 'ValueError sort_on'

    # Get unique list of all indicies in records
    indicies = ['KL','CH','Hartigan','CCC','Marriot','TrCovW',
            'TraceW','Friedman','Rubin','Cindex','DB','Silhouette',
            'Duda','PseudoT2','Beale','Ratkowsky','Ball','PtBiserial',
            'Frey','McClain','Dunn','Hubert','SDindex','Dindex','SDbw',
            'gap_tib','gap_star','gap_max','Scott']
    if indicy_filter is None:
        pass # Do not change indicies to be plotted
    elif hasattr(indicy_filter, '__iter__'):
        for indicy in indicy_filter:
            assert indicies.__contains__(indicy), ('{} is not in available'+
               ' list of indicies {}'.format(indicy, indicies))
        indicies = indicy_filter

    # Sort records
    if sort_on:
        records = sorted(records, key=lambda record: record.indicies_dictionary[sort_on])

    # Get unique list of all hyperparameter strings in records
    hyper_set = set()
    for record in records:
        _hyperparameters = record.dict2str(record.hyper_dict)
        hyper_set.add(_hyperparameters)
    hyper_list = list(hyper_set)

    # Unique list of customer_id
    _ids = [record.indicies_dictionary['customer_id'] for record in records]
    customer_ids = set(_ids)
    X = np.array(list(customer_ids))

    # Plotting
    fig, ax = plt.subplots(1)
    marker_keys = list(markers.MarkerStyle.markers.keys())
    colors = [np.array(plt.cm.hsv(i/float(len(hyper_list))))\
              .reshape(1,-1) for i in range(len(hyper_list)+1)]
    marker_list = [marker_keys[i] for i in range(len(indicies)+1)]

    # Plot indicy accuries versus customer id for all indicies
    for record in records:
        customer_id = record.indicies_dictionary['customer_id']
        x = np.where( X==customer_id )[0][0]
        hyper_str = record.dict2str(record.hyper_dict)
        for ind in indicies:
            try:
                # Get distance
                distance = record.indicies_dictionary['correct_k'] - \
                            record.indicies_dictionary[ind]
                ax.scatter(x, distance,
                            s=5,
                            c=colors[hyper_list.index(hyper_str)],
                            marker=marker_list[indicies.index(ind)])
            except TypeError:
                # None type arithmetic
                continue

    def get_limits(values, symmetric_z):
        """symmetric_z : value used for z score cutoff"""
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        Z = (values - mean) / std
        idx_high = (np.abs(Z - symmetric_z)).argmin()
        idx_low = (np.abs(Z + symmetric_z)).argmin()
        return (values[idx_low], values[idx_high])

    # Restrict axis by retrieving data from graph
    Y_record = [collection.get_offsets().data[0][1] for collection in ax.collections]
    ax.set_ylim(get_limits(Y_record, 1.65))
    ax.set_ylabel('Distance simple')
    ax.set_xlabel('Dataset -> integer')
    ax.set_title('Distance')
    ax.grid(True)
    ax.legend()

    return None



def plt_best_n_indicies(records,
                        best_n=3,
                        sort_on='correct_k',
                        indicy_filter=None):
    """Plot dataset versus accuracy of predicted number of clusters to
    correct number of clusters
    parameters
    -------
    records : (iterable) of Record objects
    best_n : (int) Number of indicies to plot. The best n indicies are plotted
    sort_on : (str) or None, key in record.dataframe to sort on.
        Must be one of ['correct_k', 'n_points', None]
    indicy_filter : (iterable) of strings representing indicies to display on the
        plot. If not in this list then their accuracy is not displayed
    """
    assert sort_on in ['correct_k','n_points', None], 'ValueError sort_on'

    # Get unique list of all indicies in records
    indicies = ['KL','CH','Hartigan','CCC','Marriot','TrCovW',
            'TraceW','Friedman','Rubin','Cindex','DB','Silhouette',
            'Duda','PseudoT2','Beale','Ratkowsky','Ball','PtBiserial',
            'Frey','McClain','Dunn','Hubert','SDindex','Dindex','SDbw',
            'gap_tib','gap_star','gap_max','Scott']
    if indicy_filter is None:
        pass # Do not change indicies to be plotted
    elif hasattr(indicy_filter, '__iter__'):
        for indicy in indicy_filter:
            assert indicies.__contains__(indicy), ('{} is not in available'+
               ' list of indicies {}'.format(indicy, indicies))
        indicies = indicy_filter

    # Sort records
    if sort_on:
        records = sorted(records, key=lambda record: record.indicies_dictionary[sort_on])

    # Get unique list of all hyperparameter strings in records
    hyper_set = set()
    for record in records:
        _hyperparameters = record.dict2str(record.hyper_dict)
        hyper_set.add(_hyperparameters)
    hyper_list = list(hyper_set)

    # Unique list of customer_id
    _ids = [record.indicies_dictionary['customer_id'] for record in records]
    customer_ids = set(_ids)
    X = np.array(list(customer_ids))

    # Get the best n indicies for each record. Remove indicies not in best n
    items = []
    for record in records:
        correct_k = record.indicies_dictionary['correct_k']
        accuracies = record.accuracy_simple(record.indicies_dictionary, correct_k)
        x = []
        for key, value in accuracies.items():
            if value.accuracy_simple is None:
                continue
            else:
                x.append((key, value.accuracy_simple))
        best = sorted(x, key=lambda item: abs(item[1]))
        best_indicies = best[:best_n]
        items.append((record, best_indicies))

    # Plotting
    fig, ax = plt.subplots(1)
    marker_keys = list(markers.MarkerStyle.markers.keys())
    colors = [np.array(plt.cm.hsv(i/float(len(hyper_list))))\
              .reshape(1,-1) for i in range(len(hyper_list)+1)]
    marker_list = [marker_keys[i] for i in range(len(indicies)+1)]

    # Plot indicy accuries versus customer id for all indicies
    for record, best_indicies in items:
        customer_id = record.indicies_dictionary['customer_id']
        x = np.where( X==customer_id )[0][0]
        hyper_str = record.dict2str(record.hyper_dict)
        for best_indicy in best_indicies:
            # best_indicy is tuple (indicy_string, accuracy)
            ax.scatter(x, best_indicy[1],
                        s=80,
                        c=colors[hyper_list.index(hyper_str)],
                        marker=marker_list[indicies.index(best_indicy[0])])

    def get_limits(values, symmetric_z):
        """symmetric_z : value used for z score cutoff"""
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        Z = (values - mean) / std
        idx_high = (np.abs(Z - symmetric_z)).argmin()
        idx_low = (np.abs(Z + symmetric_z)).argmin()
        return (values[idx_low], values[idx_high])

    # Restrict axis by retrieving data from graph
    Y_record = [collection.get_offsets().data[0][1] for collection in ax.collections]
    ax.set_ylim(get_limits(Y_record, 1.65))
    ax.set_ylabel('Accuracy simple')
    ax.set_xlabel('Dataset integer')
    ax.set_title('Best indicies for each dataset')
    ax.grid(True)
    ax.legend(indicies)

    return None



def plt_hyperparameters(records,
                        hyperparameter_name,
                        plt_density=True,
                        plt_values=False):
    """Plot clustering hyperparameters versus how accurately that hyperparameter
    performed at clustering the dataset to the correct number of clusters.
    The X axis is a sequential list of clustered instances
    The Y axis is existance of that hyperparameter related to a clustered
        instance.
    If plt_density=True the Y axis is the estimated kernel density of the
    hyperparameter existing at that position on the X axis. If the density
    close to the beginning of the X axis is high, then that clustering
    hyperparameter occured often on the most accurate clustered instances

    If plt_values=True the Y axis is the existance of a hyperparameter on
    a clustered instance. If a hyperparameter exists frequently in the best
    clustered instances, then the hyperparameter will show up frequently at
    the beginning of the X axis
    Inputs
    -------
    records : (iterable) of Record objects
    hyperparameter_name : (str) Name of the hyperparameter you want to plot.
        Should be one of the keys in record.hyper_dict
    plt_density : (bool) plot hyperparameter accuracy using gaussian density
        kernel probability density function (kernel density estimation)
    plt_values : (bool) plot the actual values of the hyperparameter.
        Not useful in string or non-plottable data types"""

    # Sort records by average accuracy
    accuracies = []
    for record in records:
        # Calculate accuracy of all indicies under hyperparameter dictionary
        correct_k = record.indicies_dictionary['correct_k']
        accuracy = record.accuracy_simple(record.indicies_dictionary, correct_k)
        # Get mean accuracy across all indicies
        accuracy_mean = np.mean([x[1] for x in accuracy.values() if x[1] is not None])
        accuracies.append([record, accuracy_mean])
    accuracies = sorted(accuracies, key=lambda tup: abs(tup[1]))
    # Get sorted records
    records = [x[0] for x in accuracies]

    # X Axis values
    X = np.arange(0, len(records))

    # Get hyperparameter values
    y = []
    for record in records:
        y.append(record.hyper_dict[hyperparameter_name])
    y = np.array(y)

    # Feature names
    feature_names = list(set(y))

    for name in feature_names:
        if plt_density:
            # Use kernel density function to estimate 1 where the hyperparameter
            kde = KernelDensity(kernel='gaussian', bandwidth=0.75)
            kde.fit(X[y==name].reshape(-1,1))
            log_density = kde.score_samples(X.reshape(-1,1))
            density = np.exp(log_density)
            plt.plot(X, density, label=str(name) + 'kde')

        elif plt_values:
            plt.scatter(X[y==name], y==name, label=name)

    plt.title('Accuracy ranking v. {}'.format(hyperparameter_name))
    plt.xlabel('Accuracy Ranking (sequential)')
    plt.ylabel(str(hyperparameter_name))
    plt.legend()

    return None



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

    return None




