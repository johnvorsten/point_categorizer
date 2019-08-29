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
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
import os
from path import Path

class Record():
    """Keep track of individual dataframes and their related information
    parameters
    -------
    dataframe : a dataframe containing error metric information. 
    See import_error_dfs()
    parent_file : original csv file
    hyper_dict : #TODO Learn from imported file"""
    
    def __init__(self, dataframe, parent_file, hyper_dict):
        self.dataframe = dataframe
        self.parent_file = parent_file
        self.hyper_dict = dict(sorted(hyper_dict.items())) #TODO Change
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
            except: #TODO handle exception
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
            except: #TODO handle exception
                continue
        
        return accuracy_df
    
    def dict2str(self):
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
        
class ExtractLabels():
    
    def __init__(self):
        pass

    def calc_features(self, database, pipeline, tag):
        """Calculate the features of a dataset. These will be used to 
        predict a good clustering algorithm.
        Inputs
        -------
        database : Your database. Its input does not matter as long as the pipeline
        passed outputs a numpy array. 
        pipeline : Your finalized pipeline. See sklearn.Pipeline. The output of
        the pipelines .fit_transform() method should be your encoded array
        of instances and features of size (n,p) n=#instances, p=#features. 
        Alternatively, you may pass a sequence of tuples containing names and 
        pipeline objects : [('pipe1', myPipe1()), ('pipe2',myPipe2())]
        tag : unique tag to identify each instance. The instance with feature 
        vectors will be returned as a pandas dataframe with the tag on the index.
        Returns
        -------
        df : a pandas dataframe with the following features 		
            a. Number of points
    		b. Correct number of clusters
    		c. Typical cluster size (n_instances / n_clusters)
    		d. Word length
    		e. Variance of words lengths?
    		f. Total number of unique words
            g. n_instances / Total number of unique words
            
        #Example Usage
        from JVWork_UnClusterAccuracy import AccuracyTest
        myTest = AccuracyTest()
        text_pipe = myDBPipe.text_pipeline(vocab_size='all', attributes='NAME',
                                   seperator='.')
        clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=False, 
                                      replace_numbers=False, 
                                      remove_virtual=True)
        mypipe = Pipeline([('clean_pipe', clean_pipe),
                           ('text_pipe',text_pipe)
                           ])
        correct_k = myTest.get_correct_k(db_name, df_clean, manual=True)
        
        
        data_transform = DataFeatures()
        db_feat = data_transform.calc_features(database, mypipe, 
                                               tag=db_name, correct_k=correct_k)
        """
        
        if hasattr(pipeline, '__iter__'):
            pipeline = Pipeline(pipeline)
        else:
            pass
        
        data = pipeline.fit_transform(database)
        if isinstance(data, csr_matrix):
            data = data.toarray()
        
        #Number of points 
        n_points = data.shape[0]
        
        #Number of features
        n_features = data.shape[1]
        
        #Word lengths
        count_dict = self.get_word_dictionary(data)
        
        #Variance of lengths
        lengths = []
        for key, value in count_dict.items():
            lengths.extend([int(key[-1])] * value)
        len_var = np.array(lengths).var(axis=0)
        
        features_dict = {
                'instance':tag,
                'n_instance':n_points,
                'n_features':n_features,
                'len_var':len_var,
                'uniq_ratio':n_points/n_features,
                }
        features_dict = {**features_dict, **count_dict}
        features_df = pd.DataFrame(features_dict, index=[tag])
        
        return features_df
    
    def get_word_dictionary(self, word_array):
    
        count_dict = {}
        
        for row in word_array:
            count = sum(row>0)
            try:
                count_dict[count] += 1
            except KeyError:
                count_dict[count] = 1
        
        for key, label in count_dict.items():
            count_dict[key] = count_dict[key] / len(word_array) #Percentage
        
        max_key = max(count_dict.keys())
        old_keys = list(count_dict.keys())
        new_keys = ['n_len' + str(old_key) for old_key in old_keys]
        
        for old_key, new_key in zip(old_keys, new_keys):
            count_dict[new_key] = count_dict.pop(old_key)
            
        required_keys = ['n_len' + str(key) for key in range(1,max_key+1)]
        for key in required_keys:
            count_dict.setdefault(key, 0)
        
        return count_dict

    def calc_labels(self, records, instance_name, best_n):
        """Given an instance_name (database name in this case) and set of records
        output the correct labels for that database.
        inputs
        -------
        records : a list of records from the Record class
        instance_name : The column/key of the database sequence you wish to 
        return labels for. In my case, it will be a path similar to D:\[...]\JobDB.mdf
        output
        -------
        A list a labels in string form. The list includes : 
            by_size
            clusterer
            distance metric
            reduced dimensionality
            best index 1
            best index 2
            best index 3
        """ 
        non_opt_cols = ['DBPath', 'correct_k','n_points', 'n_len1', 
                        'n_len2', 'n_len3', 'n_len4',
                        'n_len5', 'n_len6', 'n_len7']
        sequence_tag = 'DBPath'
        hyper_dict = {}
        loss = namedtuple('loss',['clusters', 'l2error', 'variance', 'loss'])
        losses = []
        var_scale = 0.3 #% Contribute to total loss
        error_scale = 0.7 #% Contribute to total loss
        
        for idx, record in enumerate(records): #Dictionary of unique dictionaries
            try:
                hyper_dict[self.dict2str(record.hyper_dict)]['records'].append(record)
            except KeyError:
                hyper_dict[self.dict2str(record.hyper_dict)] = {'records':[record]}
        
        non_opt_cols2 = ['instance','correct_k','records']
        
        for hyper_set, subdict in hyper_dict.items():
            subdict['instance'] = instance_name
            for record in subdict['records']:
                idx = np.where(record.dataframe[sequence_tag] == instance_name)[0]
                subdict['correct_k'] = record.dataframe.loc[idx[0], 'correct_k']
                
                for opt_k in set(record.dataframe.columns).difference(set(non_opt_cols)):
                    #Easier to access values corresponding to a database instance
                    col_name = str(opt_k)
                    try:
                        _loss = loss(record.dataframe.loc[idx[0], opt_k], None, None, None)
        #                hyper_dict[hyper_set][col_name].append(record.dataframe.loc[idx[0], opt_k])
                        hyper_dict[hyper_set][col_name].clusters.append(_loss.clusters)
                    except KeyError:
                        hyper_dict[hyper_set][col_name] = loss([_loss.clusters], None, None, None)
        #                hyper_dict[hyper_set][col_name] = [record.dataframe.loc[idx[0], opt_k]]
        
            #Calculate relevant metrics for each database on each hyperparameter set
            #Replace optimal_k calculations with errors (l2)
            for opt_k in set(subdict.keys()).difference(set(non_opt_cols2)):
                l2norm = sum(abs(np.array(subdict[opt_k].clusters) - subdict['correct_k'])**2)
                
                if len(subdict[opt_k].clusters) == 1:
                    variance = l2norm #Variance of predictions
                else:
                    variance = np.var(np.array(subdict[opt_k].clusters)) #Variance of predictions
                
                calc_loss = l2norm*error_scale + variance*len(subdict[opt_k].clusters)*var_scale
                subdict[opt_k] = loss(subdict[opt_k].clusters,
                        l2norm, 
                       variance*len(subdict[opt_k].clusters), 
                       calc_loss)
                losses.append(calc_loss)
        
        best_labels = {}
        for i in range(1, best_n+1):
            best_labels[i] = {}
        best_errors = sorted(losses)[:best_n]
        for hyper_set, subdict in hyper_dict.items():
            for opt_k in set(subdict.keys()).difference(set(non_opt_cols2)):
                if subdict[opt_k].loss in best_errors:
                    position = best_errors.index(subdict[opt_k].loss) + 1
                    best_labels[position]['by_size'] = subdict['records'][0].hyper_dict['by_size']
                    best_labels[position]['distance'] = subdict['records'][0].hyper_dict['distance']
                    best_labels[position]['clusterer'] = subdict['records'][0].hyper_dict['clusterer']
                    best_labels[position]['n_components'] = subdict['records'][0].hyper_dict['n_components']
                    best_labels[position]['reduce'] = subdict['records'][0].hyper_dict['reduce']
                    best_labels[position]['index'] = opt_k
                    
        return best_labels, hyper_dict
    
    def dict2str(self, hyper_dict):
        x = str()
        for key, value in hyper_dict.items():
            x = x + str(key) + ':' + str(value) + '\n'
        return x

            


def import_error_dfs():
    """Imports error_df csv files and converts to custom Record objects
    Returns
    -------
    [data1, data2, data3, data4, data5,data6]"""
    
    error_df_list = [r"C:\Users\z003vrzk\.spyder-py3\Scripts\ML\Point database categorizer\error_dfs\error_df 8-6 (1).csv",
                     r"C:\Users\z003vrzk\.spyder-py3\Scripts\ML\Point database categorizer\error_dfs\error_df 8-6 (2).csv",
                     r"C:\Users\z003vrzk\.spyder-py3\Scripts\ML\Point database categorizer\error_dfs\error_df 8-6 (3).csv",
                     r"C:\Users\z003vrzk\.spyder-py3\Scripts\ML\Point database categorizer\error_dfs\error_df 8-7 (2).csv",
                     r"C:\Users\z003vrzk\.spyder-py3\Scripts\ML\Point database categorizer\error_dfs\error_df 8-20 (1).csv",
                     r"C:\Users\z003vrzk\.spyder-py3\Scripts\ML\Point database categorizer\error_dfs\error_df 8-20 (2).csv"
                     ]
    reduce_cols = ['DBPath', 'correct_k', 'optk_MDS_gap_max', 'optk_MDS_gap_Tib',
           'optk_MDS_gap*_max', 'n_points', 'n_len1', 'n_len2', 'n_len3', 'n_len4',
           'n_len5', 'n_len6', 'n_len7']
    standard_cols = ['DBPath', 'correct_k', 'optk_X_gap_max', 'optk_X_gap_Tib',
           'optk_X_gap*_max', 'n_points', 'n_len1', 'n_len2', 'n_len3', 'n_len4',
           'n_len5', 'n_len6', 'n_len7']
    #Possible values, for notes
    hyper_dict = {'by_size':[True, False],
                  'clusterer':['kmeans','agglomerative'],
                  'distance':['eucledian', 'minkowski'],
                  'reduce':['MDS', 'TSNE', False],
                  'n_components':[2,5,10]
                  }
    
    def get_hyper_dict(df_path):
        #Get directory files same as df_path
        #Find the file that matches df_path and has hyper
        #read to dataframe
        #return as dictionary
        csvpath = Path(df_path)
        parts = csvpath.splitpath()
        base_dir = parts[0]
        name = parts[1]

        hyper_path = base_dir.glob('*' + name[:-4] + '*' + 'hyper' + '*')
        if len(hyper_path) == 1:
            hyper_path = hyper_path[0]
        else:
            print('Error with hyper_path : {}'.format(hyper_path))
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
    for idx, error_df_path in enumerate(error_df_list):
        error_df = pd.read_csv(error_df_path, header=0, index_col=0)
        error_dfs.append(error_df)
        hyper_dict = get_hyper_dict(error_df_path)
        record = Record(error_df, error_df_path, hyper_dict)
        records.append(record)
    
    return records


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






