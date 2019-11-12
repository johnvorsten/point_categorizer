# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:43:28 2019

@author: z003vrzk
"""

# Third party imports
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# Local imports

#%% 

""" Example usage of this module

#Intantiate the class
myDBPipe = JVDBPipe()
myClustering = JVClusterTools()

#Load DataBase from memory
_master_pts_db = r"D:\Z - Saved SQL Databases\master_pts_db.csv"
my_iter = myClustering.read_database_set(_master_pts_db)
ind, df = next(my_iter)

#Apply custom cleaning/alteration pipelines to data
df_clean = myDBPipe.cleaning_pipeline(df, remove_dupe=False, 
                                      replace_numbers=False, remove_virtual=True)
df_text = myDBPipe.text_pipeline(df_clean, vocab_size='all')

#Create a vocabulary or words for human comprehension
_vocabulary = df_text.columns.tolist()

_df_systems = df_clean['SYSTEM']
X = df_text.values

#Create the data, one system at a time for manual labeling/inspection
group_iterator = myClustering.get_database_set(_df_systems, X)
_indicies, X = next(group_iterator)

#Visualize some data
_point_names = myClustering.get_word_name(X, _vocabulary)
print(df_clean.loc[_indicies[0], 'DBPath'])
for _i in range(0, min(5, len(_point_names))):
    print(_point_names[_i])
"""


class JVReadingTools():
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_database_set(names, features):
        """Generates an iterator which yields dataframes that all have a common
        attribute in common. Each names and features must have the same length,
        where every feature/row in features must correspond to a name in names
        parameters
        -------
        dbnames : a dataframe, list, or np.array with the names corresponding to
            the features we want to yield
        features : a dataframe, list, or np.array with the features corresponding
            to dbnames
        returns
        -------
        an iterator over featuers that returns a np.array of features[index,:"]
        where index is determined by all indicies in names taht have a common name/label"""
        assert len(features) == len(names), 'Features and names must be same length'
        
        if type(features) == pd.DataFrame:
            features = features.values
        
        if type(names) == pd.DataFrame or pd.Series:
            names = names.values
        
        if type(names) == list:
            names = np.array(names)
        
        unique_names = list(set(names.flat))
        
        for name in unique_names:
            
            indicies = np.where(names==name)[0]
            feature_rows = features[indicies,:]
            
            yield indicies, feature_rows
    
    @staticmethod
    def read_database_set(database_name, column_name='DBPath'):
        """Yields sequential data from memory.
        parameters
        -------
        database_name : path to csv database (string)
        column_tag : column name that contains labels for each sequential set.
            Must be included on each row.
        output
        -------
        iterator over a database grouped by a common column_tag
        yield (indicies, sequence).
        indicies : indicies of pandas dataframe
        sequence : pandas dataframe of database
        
        Example
        my_iter = read_database_set(db_path, column_tag='DBPath')
        'ind, df = next(my_iter)
        print(ind[0],":",ind[-1], " ", df['DBPath'][0])"""
    
        csv_iterator = pd.read_csv(database_name,
                                   index_col=0,
                                   iterator=True,
                                   chunksize=50000,
                                   encoding='mac_roman'
                                   )
        for chunk in csv_iterator:
            
            partial_set = set(chunk[column_name])
            unique_names = list(partial_set)
            
            for name in unique_names:
                
                indicies = np.where(chunk[column_name]==name)[0]
                sequence = chunk.iloc[indicies]
                
                yield indicies, sequence
    
    @staticmethod
    def read_database_ontag(file_path, column_name, column_tag):
        """Let Y denotate the label space. X denotates the instance space.
        Retrieves all axis-0 indicies of column_tag in column_name. This is 
        useful for retrieving all instances in {(Xi, yi) | 1<i<m} whose yi
        match column_tag (assuming column_tag is in the space of Y).
        parameters
        -------
        file_path : path to file
        column_name : column that contains all yi for 1<i<m
        column_tag : value from Y to match for each yi in 1<i<m"""
        
        df = pd.read_csv(file_path, 
                         index_col=0, 
                         usecols=[column_name],
                         encoding='mac_roman')
        
        cols = pd.read_csv(file_path, 
                           index_col=0,
                           encoding='mac_roman',
                           nrows=0).columns.tolist()
        indicies = np.where(df.index == column_tag)[0] + 1
        
        df_whole = pd.read_csv(file_path, 
                         names=cols,
                         encoding='mac-roman',
                         skiprows = lambda x: x not in indicies)
        df_whole.reset_index(drop=True, inplace=True)
        return df_whole
        
    
    
    @staticmethod
    def get_word_name(features, vocabulary):
        """Prints the associated words of a one-hot encoded text phrase
        from the vocabulary. Assumes the order of features and vocabulary
        is in the same order
        parameters
        -------
        features : one-hot encoded feature vector (single vector or array). Must
            be of type np.array or pd.DataFrame
        vocabulary : list or np.array of strings
        output
        -------
        words : nested list of decoded words"""
        assert features.shape[1] == len(vocabulary), 'Features and Vocab must be same length'
        
        if type(features) == pd.DataFrame:
            features = features.values
        
        if type(vocabulary) == pd.DataFrame:
            vocabulary = vocabulary.values
        
        if type(vocabulary) == list:
            vocabulary = np.array(vocabulary)
            
        words = []
        for vector in features:
            
            indicies = np.where(vector==1)[0]
            words_iter = vocabulary[indicies]
            words.append(words_iter)
            
        return words
    
    @staticmethod
    def optimalK2(data, nrefs=3, maxClusters=15):
        """
        Calculates KMeans optimal K using Gap Statistic from Tibshirani, 
        Walther, Hastie
        Params:
            data: ndarry of shape (n_samples, n_features)
            nrefs: number of sample reference datasets to create
            maxClusters: Maximum number of clusters to test for
        Returns: (gaps, optimalK)
        """
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = pd.DataFrame({'n_clusters':[], 
                                  'gap_value':[],  'gap*':[], 
                                  'obs_dispersion':[], 
                                  'ref_dispersion':[] })
        for gap_index, k in enumerate(range(1, maxClusters)):
    
            # Holder for reference dispersion results
            refDisps = np.zeros(nrefs)
    
            # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
            for i in range(nrefs):
                
                # Create new random reference set
                randomReference = np.random.random_sample(size=data.shape)
                
                # Fit to it
                km = KMeans(k)
                km.fit(randomReference)
                
                refDisp = km.inertia_
                refDisps[i] = refDisp
    
            # Fit cluster to original data and create dispersion
            km = KMeans(k)
            km.fit(data)
            
            origDisp = km.inertia_
            refDisp_mean = np.mean(refDisps)
    
            # Calculate gap statistic
            gap = np.mean(np.log(refDisps)) - np.log(origDisp) 
            gap_star = np.mean(refDisps) - origDisp
    
            # Assign this loop's gap statistic to gaps
            gaps[gap_index] = gap
            
            resultsdf = resultsdf.append({'n_clusters':k, 'gap_value':gap, 
                                          'ref_disp':refDisps, 'obs_dispersion':origDisp,
                                          'ref_dispersion':refDisp_mean, 'gap*':gap_star}, ignore_index=True)
    
        return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


#%% Predicting best number of clusters
"""Use this method to predict the best number of clusters given clustering 
hyperparameters
"""

# Local imports
from JVrpy2 import nbclust_calc
from JVWork_WholeDBPipeline import JVDBPipe
from model_serving import LoadSerializedAndServe

# Third party imports
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from gap_statistic import OptimalK
from pymongo import MongoClient
import pandas as pd
import numpy as np
from collections import namedtuple

# Instantiate local classes
myDBPipe = JVDBPipe()
client = MongoClient('localhost', 27017)
db = client['master_points']
collection = db['raw_databases']

def predict_n_clusters(document, hyperparam_list):
    """Use this method to predict the best number of clusters given clustering 
    hyperparameters
    
    inputs
    -------
    document : (dict) mongodb document
    hyperparam_list : (List) of hyperparameters you want to cluster using.
    Each item in the list should be a dictinoary of hyperparameters definitions
    that will be used to run various clustering algorithms.
    """
    pass

    
#%% Finsihed

def get_max_nc(X):
    """Return maximum number of clusters to try when clustering. Assume one
    cluster for every 2 points (2 points/cluster) for small datasets, and 
    grow the number as number of points increases"""
    n_points = X.shape[0]
    
    # Define points per cluster at various number of points
    ppc_le_50 = 1.25
    ppc_le_100 = 1.9
    ppc_le_150 = 2.5
    ppc_le_200 = 2.8
    ppc_le_250 = 3.3
    ppc_le_400 = 4
    ppc_le_750 = 4.8
    
    if n_points <= 50:
        max_nc = int(n_points / ppc_le_50)
    elif n_points <= 100:
        max_nc = int(n_points / ppc_le_100)
    elif n_points <= 150:
        max_nc = int(n_points / ppc_le_150)
    elif n_points <= 200:
        max_nc = int(n_points / ppc_le_200)
    elif n_points <= 250:
        max_nc = int(n_points / ppc_le_250)
    elif n_points <= 400:
        max_nc = int(n_points / ppc_le_400)
    else:
        max_nc = int(n_points / ppc_le_750)
        
    return max_nc

def make_cleaning_pipe(remove_dupe=False, 
                       replace_numbers=False, 
                       remove_virtual=True,
                       vocab_size='all',
                       attributes='NAME',
                       seperator='.',
                       heirarchial_weight_word_pattern=True):
    """Make a cleaning pipeline. This document will namespace the cleaning pipe
    so it can be re-used to prepare raw input data. This function returns a 
    function that can be used to run the cleaning pipeline
    
    inputs
    -------
    remove_dupe : see JVDBPipe.cleaning_pipeline
    replace_numbers : see JVDBPipe.cleaning_pipeline
    remove_virtual : see JVDBPipe.cleaning_pipeline
    vocab_size : see JVDBPipe.text_pipeline
    attributes : see JVDBPipe.text_pipeline
    seperator : see JVDBPipe.text_pipeline
    heirarchial_weight_word_pattern : see JVDBPipe.text_pipeline
    
    outputs
    -------
    database : (pd.DataFrame) dataframe of raw input data
    df_clean : (pd.DataFrame) cleaned raw input data
    X : (np.array) specified text features transformed as specified by cleaning
        pipeline
    
    Example usage
    # Make a total pipeline
    my_pipeline = make_cleaning_pipe(remove_dupe=False, [...])
    # process input data with the pipeline
    database, df_clean, X = my_pipe(document)
    """
    
    from JVWork_WholeDBPipeline import JVDBPipe
    myDBPipe = JVDBPipe()
    
    # Create 'clean' data processing pipeline
    clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=remove_dupe, 
                                          replace_numbers=replace_numbers, 
                                          remove_virtual=remove_virtual)
    
    # Create pipeline specifically for clustering text features
    text_pipe = myDBPipe.text_pipeline(vocab_size=vocab_size, 
                                       attributes=attributes,
                                       seperator=seperator,
                                       heirarchial_weight_word_pattern=heirarchial_weight_word_pattern)
    
    def _cleaning_pipe_output(document, input_type='mongodb'):
        
        if input_type == 'mongodb':
            database = pd.DataFrame.from_dict(document['points'], 
                                              orient='columns')
        elif input_type == 'DataFrame':
            database = document
        else:
            raise ValueError("Invalid parameter input_type. Try 'mongodb' or\
                             DataFrame")
        
        # pass data through cleaning and text pipeline
        df_clean = clean_pipe.fit_transform(database)
        X = text_pipe.fit_transform(df_clean).toarray()
    
        return database, df_clean, X

    return _cleaning_pipe_output

def split_database_on_panel(document):
    """Divide a whole database up on NETDEVID. This will divide a whole database
    into single-controller databases
    inputs
    -------
    document : (dict) mongodb document
    output
    -------
    database : (pd.DataFrame) individual based on the set of all unique controller
    names in the input document
    
    database_iterator = split_database_on_panel(document)
    sub_database = next(database_iterator)"""
    
    database = pd.DataFrame.from_dict(document['points'], 
                                      orient='columns')

    unique_controller_names = set(database['NETDEVID'])
    # Remove nan values
    unique_controller_names = {x for x in unique_controller_names if x==x}
    
    for name in unique_controller_names:
        
        yield database[database['NETDEVID'] == name]


def parse_hyperparameter_dictionary(hyper_dict):
    
    # by_size controller
    if isinstance(hyper_dict['by_size'], bool):
        by_size = hyper_dict['by_size']
    elif hyper_dict['by_size'] == 'True':
        by_size = True
    elif hyper_dict['by_size'] == 'False':
        by_size = False
    else:
        raise(ValueError('Fuck me UnsupervisedCluster.py'))
    
    # Index controller
    optimalk_indicies = {'optk_TSNE_gap_Tib':'gap_tib',
                         'optk_X_gap_Tib':'gap_tib',
                         'optk_MDS_gap_Tib':'gap_tib',
                         'optk_MDS_gap_max':'gap_max',
                         'optk_TSNE_gap_max':'gap_max',
                         'optk_X_gap_max':'gap_max',
                         'optk_TSNE_gap*_max':'gap_star_max',
                         'optk_X_gap*_max':'gap_star_max',
                         'optk_MDS_gap*_max':'gap_star_max'}
    
    nbclust_indicies = ['mcclain',
                        'hartigan',
                        'ratkowsky',
                        'beale',
                        'friedman',
                        'tracew',
                        'dunn',
                        'silhouette',
                        'scott',
                        'pseudot2',
                        'ccc',
                        'trcovw',
                        'db',
                        'marriot',
                        'sdbw',
                        'cindex',
                        'ptbiserial',
                        'ball',
                        'duda',
                        'rubin',
                        'ch',
                        'kl',
                        'frey',
                        'sdindex']
    
    if hyper_dict['index'].lower() in nbclust_indicies:
        # Index is stored with some upper case for nbclust indicies
        index = hyper_dict['index'].lower()
        
    elif hyper_dict['index'] in optimalk_indicies:
        # I did a bad job naming optimalk indicies initially
        # map the original names to the reduced set. The reduced set should be
        # one of ['gap_tib','gap_max','gap_star_max']
        index = optimalk_indicies[hyper_dict['index']]
    
    # Clusterer controller
    clusterer = hyper_dict['clusterer']
#    if hyper_dict['clusterer'] == 'Ward.D':
#        clusterer = 'ward.D'
#    elif hyper_dict['clusterer'] == 'Ward.D2':
#        clusterer = 'ward.D2'
#    else:
#        clusterer = hyper_dict['clusterer']
    
    # n_components
    n_components = int(hyper_dict['n_components'])
    
    # Method
    dimensionality_reduction_method = hyper_dict['reduce']
    
    # Distance
    distance = 'euclidean' # Static
    
    parsed_clustering_params =  {
            'by_size': by_size,
            'distance': distance,
            'clusterer': clusterer,
            'n_components': n_components,
            'reduce': dimensionality_reduction_method,
            'index': index}

    return parsed_clustering_params

def dimensionality_reduction(X, method, n_components):
    """Reduce dimensionality of encoded text
    inputs
    -------
    method : (str) one of ['MDS','TSNE']
    n_components : (int) feature space to reduce towards
    ouput
    ------
    X : (np.array) reduced feature space"""
    
    # Dimensionality reduction if called for
    if method=='MDS':
        mds = MDS(n_components = n_components)
        X_dim_reduce = mds.fit_transform(X_partial)
    elif method=='TSNE':
        params = {'method':'barnes_hut', 
                  'n_components':n_components,
                  'metric':'euclidean',
                  'perplexity':12}
        if n_components >= 4:
            params['method'] = 'exact' #Slower
        tsne = TSNE(method=params['method'],n_components=params['n_components'],
                    metric=params['metric'],perplexity=params['perplexity'])
        X_dim_reduce = tsne.fit_transform(X_partial)
    else:
        # Do not apply dimensionality reduction
        X_dim_reduce = X
            
    return X_dim_reduce

def divide_indicies_by_size(X, by_size):
    
    # Handling by_size hyperparameter
    words_bool = X > 0
    lengths = np.sum(words_bool, axis=1)
    unique_lengths = list(set(lengths))
    
    global_indicies = []
    
    if by_size:
        # Divide database into segments based on word length
        # Cluster on each of these partial datasets
        for length in unique_lengths:
            indicies = np.where(lengths == length)
            global_indicies.append(indicies)
    else:
        # Use all indicies in X to cluster - do not cluster partial datasets
        indicies = np.arange(0,X.shape[0],1)
        global_indicies.append(indicies)
        
    return global_indicies



# Clustering with NbClust
def _nbclust_calc(X,
                  index, 
                  clusterer, 
                  distance,
                  min_nc,
                  max_nc):
    """Returns the calculated optimal number of clusters with the NbClust
    R Package.
    parameters
    -------
    X : word encoded array (n,p) n=# instances, p=#features/words
    method : dimensionality reduction method. Must be either 'TSNE' or 'MDS'
    n_components : number of dimensions of reduced dataset (p->p_reduced)
    index : Index to use with NbClust package. See NbClust documentation.
    Typical values can be 'all' or 'alllong'
    by_size : (True/False) cluster by word size (True), or the whole database
    at once (False). Recommended False with the NbClust implementation
    output
    -------
    new_vals (dictionary) : A dictionary that can be used to construct a dataframe.
    The keys returned are variable based on 'index'"""
    
    if X.shape[0] >= 1200:
        print('{} Skipped | size {}'.format(db_name, X_reduced.shape[0]))
        return None
    
    # Calculate with NbClust method
    # max_nc = max_clusters-1 # R Indexing lol
    
    # print(f'min {min_nc} max {max_nc}')
    # print('X.shape {}'.format(X.shape))
    Nb_result = nbclust_calc(X, 
                 min_nc=min_nc, 
                 max_nc=max_nc,
                 distance=distance, 
                 clusterer=clusterer,
                 index=index)

    df_bestnc = Nb_result.best_nc_df
    
    return df_bestnc

# Clustering with kmeans optimalk

def _optimalk_calc(X,
                  index, 
                  clusterer, 
                  distance,
                  min_nc,
                  max_nc):
    """Returns the calculated number of clusters with different methods
    parameters
    -------
    X : word encoded array
    index : (str) one of optk_gap, optk_gap_star, optk_gap_tib
    distance : distance metreic for NbClust. See NbClust documents. 
        only 'euclidean' is currently supported
    clusterer : (str) must be 'kmeans'

    output
    -------
    new_vals (pd.DataFrame) : best_nc_df"""
    
    optimalk = OptimalK(parallel_backend='multiprocessing')
    X = X.astype(np.float32)
    n_clusters = optimalk(X, cluster_array=np.arange(min_nc,max_nc,1))
    gapdf = optimalk.gap_df
    
    df_bestnc = pd.DataFrame(index=['Number_clusters','Value_Index'], 
                         columns=['gap_tib','gap_max','gap_star_max'],
                         data=[[None,None,None],[None,None,None]])
    
    if 'gap_tib' in index:
        # add value to df_bestnc
        gap_value = gapdf['gap_value'].values
        reference_dispersion_std = gapdf['ref_dispersion_std'].values
        diffs = gap_value[:-1] - gap_value[1:] + reference_dispersion_std[:-1]
        
        for idx, diff in enumerate(diffs):
            if diff > 0:
                n_clusters = gapdf.loc[idx, 'n_clusters']
                value_index = gapdf.loc[idx, 'gap_value']
                df_bestnc['gap_tib'] = [n_clusters, value_index]
                break
        
    if 'gap_max' in index:
        max_index = gapdf['gap_value'].idxmax()
        n_clusters = gapdf.loc[max_index, 'n_clusters']
        value_index = gapdf.loc[max_index, 'gap_value']
        df_bestnc['gap_max'] = [n_clusters, value_index]
    
    if 'gap_star_max' in index:
        max_index = gapdf['gap*'].idxmax()
        n_clusters = gapdf.loc[max_index, 'n_clusters']
        value_index = gapdf.loc[max_index, 'gap*']
        df_bestnc['gap_star_max'] = [n_clusters, value_index]
    
    if not any(('gap_star_max' in index, 'gap_tib' in index,'gap_max' in index)):
        raise ValueError('Invalid index passed to _optimalk_calc')
    
    return df_bestnc

#%% 




# Create list of 5 best hyperparam dicts
best_hyperparam_list = [{'by_size': False,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'Ratkowsky'},
 {'by_size': True,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'Cindex'},
 {'by_size': True,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'CCC'},
 {'by_size': True,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'Silhouette'},
 {'by_size': True,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'Hartigan'}]


_cursor = collection.find()

#for document in _cursor:
document = next(_cursor)

database_iterator = split_database_on_panel(document)
#for sub_database in database_iterator:
sub_database = next(database_iterator)

# TODO make pipeline before so its not re-instantialized
my_pipeline = make_cleaning_pipe(remove_dupe=False, 
                       replace_numbers=False, 
                       remove_virtual=True,
                       vocab_size='all',
                       attributes='NAME',
                       seperator='.',
                       heirarchial_weight_word_pattern=True)

database, df_clean, X = my_pipeline(sub_database, input_type='DataFrame')
print(df_clean['NAME'])




# Controller for clustering
for hyper_dict in best_hyperparam_list:
    
    hyper_dict = parse_hyperparameter_dictionary(hyper_dict)
    
    # Iterate over each word size; cluster on each partial dataset
    # If by_size is True 
    indicies_list = divide_indicies_by_size(X, hyper_dict['by_size'])
    for indicie_group in indicies_list: 
        X_partial = X[indicie_group]
        
        #Problem with R indexing and NbClust requirements
        if X_partial.shape[0] <= 4:
            continue

        X_dim_reduced = dimensionality_reduction(X_partial, 
                                     method=hyper_dict['reduce'],
                                     n_components=hyper_dict['n_components'])
        
        # Conditionally call nbclust package or optimalk package 
        # based on input clustering hyperparameters
        nbclust_clusterer = ['kmeans','ward.D', 'ward.D2', 'average']
        
        optimalk_indicies = ['gap_Tib',
                             'gap_max',
                             'gap_star_max']
    
        nbclust_indicies = ['mcclain',
                            'hartigan',
                            'ratkowsky',
                            'beale',
                            'friedman',
                            'tracew',
                            'dunn',
                            'silhouette',
                            'scott',
                            'pseudot2',
                            'ccc',
                            'trcovw',
                            'db',
                            'marriot',
                            'sdbw',
                            'cindex',
                            'ptbiserial',
                            'ball',
                            'duda',
                            'rubin',
                            'ch',
                            'kl',
                            'frey',
                            'sdindex']
    
        if hyper_dict['index'] in nbclust_indicies:
            # Cluster with nbclust and clustering algorithm
            min_nc = 2 # Static
            max_nc = get_max_nc(X) # Based on actual data
            
            best_nc_df = _nbclust_calc(X_dim_reduced,
                                       index=hyper_dict['index'],
                                       clusterer=hyper_dict['clusterer'],
                                       distance=hyper_dict['distance'],
                                       min_nc=min_nc,
                                       max_nc=max_nc)
        
        
        elif index in optimalk_indicies:
            # Cluster with optimalk and kmeans
            assert hyper_dict['clusterer'] == 'kmeans', ('Clusterer and index\
                             are not compatable')
            
            min_nc = 2 # Static
            max_nc = get_max_nc(X) # Based on actual data
            
            best_nc_df = _optimalk_calc(X_dim_reduced,
                                       index=hyper_dict['index'],
                                       clusterer=hyper_dict['clusterer'],
                                       distance=hyper_dict['distance'],
                                       min_nc=min_nc,
                                       max_nc=max_nc)
            
        else:
            error_msg = ('Unknown clustering indicy passed. Please double-' + 
                         'check index in in one of those listed')
            raise ValueError(error_msg)
        














#%% Use this class for unsupevised clustering of databases

"""This class will use known or predicted number of clusters for a database
to partition the database into their partitions.
For example a database with point names :
BANC.RTU0202.CCT
BANC.EF0101.PRF
BANC.EF0101.RMT
BANC.EF0101.SS
BANC.EF0201.PRF
BANC.EF0201.SS
should be separated into BANC.RTU0202*, BANC.EF101*, BANC.EF0201* groups"""

def insert_document_to_document(document, field, insert_item, collection):
    """Inserts a field into an existing document
    inputs
    -------
    document : (dict) document from mongodb to which you want to insert
    field : (str) string of field name
    insert_dict : (dict) dict to insert under field
    collection : (pymongo.Collection.collection) collection that owns document"""
    
    document_id = document['_id']
    result = collection.update_one({'_id':document_id}, {'$set':{field:insert_item}})
    
    return result.modified_count


# Third party imports
from pymongo import MongoClient
import pymongo
from kmodes.kmodes import KModes
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np

# Local imports
from JVWork_WholeDBPipeline import JVDBPipe

# Instantiate local classes
myDBPipe = JVDBPipe()
# Create 'clean' data processing pipeline
clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=False, 
                                      replace_numbers=False, 
                                      remove_virtual=True)

# Create pipeline specifically for clustering text features
text_pipe = myDBPipe.text_pipeline(vocab_size='all', 
                                   attributes='NAME',
                                   seperator='.',
                                   heirarchial_weight_word_pattern=True)

# Retrieve information from Mongo
client = MongoClient('localhost', 27017)
db = client['master_points']
collection_raw = db['raw_databases']
collection_clustered = db['clustered_points']

_cursor = collection_raw.find()
for document in _cursor:
    
    # pass data through cleaning and text pipeline
    database = pd.DataFrame.from_dict(document['points'], 
                                      orient='columns')
    
    # Test if document already exists
    exists_cursor = collection_clustered.find({'database_tag':document['database_tag']}, {'_id':1})
    if exists_cursor.alive:
        # Dont recalculate
        continue
    
    df_clean = clean_pipe.fit_transform(database)
    X = text_pipe.fit_transform(df_clean).toarray()
    #_word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
    #df_text = pd.DataFrame(X, columns=_word_vocab)
    
    
    # perform clustering
    actual_n_clusters = document['correct_k']
    if actual_n_clusters == 1:
        # Dont cluster - why would you? 
        prediction_agglo = np.ones((X.shape[0]))
        
    if X.shape[0] <= 3:
        # Dont cluster - just pass 1 cluster total
        prediction_agglo = np.ones((X.shape[0]))
        
    else:
        # Cluster
        agglomerative = AgglomerativeClustering(n_clusters=actual_n_clusters, 
                                                affinity='euclidean',
                                                linkage='ward')
        prediction_agglo = agglomerative.fit_predict(X)
    
    # Format of clustered_points_dict is {setid:}
    clustered_points_list = []
    for setid in set(prediction_agglo):
        cluster_dict = {}
        indicies = (prediction_agglo==setid)
        cluster_names = df_clean.iloc[indicies]
        points_dict = cluster_names.to_dict(orient='list')
        cluster_dict['points'] = points_dict
        
        # Add entries into clustered_points_dict
        clustered_points_list.append(cluster_dict)

    try:
        database_tag = document['database_tag']
        collection_clustered.insert_one({'clustered_points':clustered_points_list,
                                         'database_tag':database_tag,
                                         'correct_k':actual_n_clusters})
        
    except pymongo.errors.WriteError:
        continue



if __name__ == '__main__':
    km = KModes(n_clusters=actual_n_clusters, 
            init='Huang', 
            n_init=5)
    prediction_kmode = km.fit_predict(X)
    
    print('KMode clustering : ')
    for _setid in set(prediction_kmode):
        _indicies = (prediction_kmode==_setid)
        _cluster_names = df_clean.iloc[_indicies]['NAME'].tolist()
        print(_cluster_names)
    
    print('\n\nAgglomerative clustering : ')
    for _setid in set(prediction_agglo):
        _indicies = (prediction_agglo==_setid)
        _cluster_names = df_clean.iloc[_indicies]['NAME'].tolist()
        print(_cluster_names)
        insert_list_document
