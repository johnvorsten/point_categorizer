# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:43:28 2019

Example usage of this module

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

@author: z003vrzk
"""

# Python imports
import os
import sys
from collections import namedtuple

# Third party imports
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from gap_statistic import OptimalK
import matplotlib.pyplot as plt

# Local imports
"""Add modules in lateral packages to python path. This is necessary because
relative imports are not possible when running this module as a top level module
AKA with a name == __main__"""
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

    from clustering.nbclust_rpy2 import nbclust_calc
else:
    from clustering.nbclust_rpy2 import nbclust_calc

# Local declarations
class DepreciationError(Exception):
    pass


#%%

class OptimalKCluster():

    def __init__(self):
        pass

    @staticmethod
    def optimalK(data, nrefs=3, max_clusters=15):
        """
        Calculates KMeans optimal K using Gap Statistic from Tibshirani,
        Walther, Hastie
        Params:
            data: ndarry of shape (n_samples, n_features)
            nrefs: number of sample reference datasets to create
            max_clusters: Maximum number of clusters to test for
        Returns: (gaps, optimalK)
        """
        gaps = np.zeros((len(range(1, max_clusters)),))
        resultsdf = pd.DataFrame({'n_clusters':[],
                                  'gap_value':[],  'gap*':[],
                                  'obs_dispersion':[],
                                  'ref_dispersion':[] })
        for gap_index, k in enumerate(range(1, max_clusters)):

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


#%%


class UnsupervisedClusterPoints():
    """TODO NbClust cannot handle individual indicies with a number of clusters
    greater than (20).  This means cindex with a number of clusters >20 will
    raise an exception. The fix this, I am converting all indicies that
    would normall go to NbClust to 'all'. 'all' is a code which calculates
    indicies for all indicies available in NbClust, and can accept a max
    number of clusters >20. see _parse_hyperparameter"""

    def __init__(self):

        # All possible nbclust clustering algorithms
        self.nbclust_clusterer = ['kmeans','ward.D', 'ward.D2', 'average']

        # Possible optimalk indicies
        self.optimalk_indicies = ['gap_Tib',
                             'gap_max',
                             'gap_star_max']

        # Index controller for optimalk calculation and parsing
        self.optimalk_indicy_mapper = {'optk_TSNE_gap_Tib':'gap_tib',
                             'optk_X_gap_Tib':'gap_tib',
                             'optk_MDS_gap_Tib':'gap_tib',
                             'optk_MDS_gap_max':'gap_max',
                             'optk_TSNE_gap_max':'gap_max',
                             'optk_X_gap_max':'gap_max',
                             'optk_TSNE_gap*_max':'gap_star_max',
                             'optk_X_gap*_max':'gap_star_max',
                             'optk_MDS_gap*_max':'gap_star_max'}

        self.nbclust_indicies = ['mcclain',
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
                            'sdindex',
                            'all',
                            'alllong']
        return None


    def cluster_with_hyperparameters(self, hyperparam_dict, X):
        """Clusteres an input points_dataframe with the clustering
        hyperparameters in hyperparam_list.  The output is a list of results
        for each hyperparameter specification in hyperparam_list
        It is important to note the form of hyperparam_list and its contents
        Inputs
        -------
        hyperparam_dict : (dict) The dictionary should contain
            specifications for how points_dataframe should be clustered. The
            required keys are {'by_size','clusterer','n_components',
                               'index','reduce', 'distance'}
            by_size : (bool) cluster by size (Depreciated)
            clusterer : (str) one of [ward.D, ward.D2, kmeans, single, complete,
                                      average, mcquitty, median, centroid]
            n_components : (int) number of features to reduce to, usually 5 or 8
            index : (str) see NbClust, or one of ['all','gap','ccc','cindex']
            reduce : (bool) if True perform dimensionality reduction
            distance : (str) distance measure, one of [euclidean, maximum,
                                                       binary, minkowski]
        X : (np.array) of encoded point names. Extract these from document
            in mongodb or other. This should be an array of encoded points
            which has been passed through a cleaning pipeline.  See
            make_cleaning_pipeline for creating and using a cleaning pipeline.
        outputs
        -------
        cluster_results : (tuple) defining your results. There
            are two keys ['hyperparameters','best_nc_dataframe'].
            'hyperparameters' are the hyperparameters used to cluster

        Example usage :

        hyperparameters = {'by_size':False,
                           'distance': 'euclidean',
                           'clusterer': 'ward.D',
                           'n_components': 8,
                           'reduce': 'MDS',
                           'index': 'Hartigan'}

        raw_database = pd.DataFrame(data)
        clean_database = cleaning_pipeline.fit_transform(raw_database)
        text_features = text_pipeline.fit_transform()
        X = text_features.toarray()
        # Cluster X
        result = myClusterer.cluster_with_hyperparameter_list(hyperparameters,
                                                              X)"""
        # Save the results of each item in the list
        results_tuple = namedtuple('cluster_results',
                                   ['hyperparameters','best_nc_dataframe'])


        # Parse the hyperparameter dictionary
        hyper_dict = self._parse_hyperparameter_dictionary(hyperparam_dict)
        index = hyper_dict['index']
        reduce = hyper_dict['reduce']
        clusterer = hyper_dict['clusterer']
        n_components = hyper_dict['n_components']
        distance = hyper_dict['distance']

        # Build results
        cluster_result = results_tuple
        cluster_result.hyperparameters = hyper_dict

        # Cluster
        best_nc_df = self._cluster(X, index,
                                   clusterer, n_components,
                                   distance, reduce)
        cluster_result.best_nc_dataframe = best_nc_df

        return cluster_result


    def _cluster_by_size(self, X,
                         by_size, index,
                         clusterer, n_components,
                         distance, reduce):
        """Cluster by_size (Depreciated)"""

        # Iterate over each word size; cluster on each partial dataset
        # If by_size is True
        indicies_list = self._divide_indicies_by_size(X, by_size)
        for indicie_group in indicies_list:
            X_partial = X[indicie_group]

            #Problem with R indexing and NbClust requirements, dont cluster
            if X_partial.shape[0] <= 4:
                return None

            X_dim_reduced = self._dimensionality_reduction(X_partial,
                                                           method=reduce,
                                                           n_components=n_components)

            # Conditionally call nbclust package or optimalk package
            # based on input clustering hyperparameters
            if index in self.nbclust_indicies:
                # Cluster with nbclust and clustering algorithm
                min_nc = 2 # Static
                max_nc = self._get_max_nc(X) # Based on actual data

                best_nc_df = self._nbclust_calc(X_dim_reduced,
                                            index=index,
                                            clusterer=clusterer,
                                            distance=distance,
                                            min_nc=min_nc,
                                            max_nc=max_nc)

                if best_nc_df is None:
                    # The best number of clusters has not yet been predicted
                    # And best_nc_dataframe is None
                    result = best_nc_df
                else:
                    # Add predicted number of clusters to existing dataframe
                    # This is used when by_size is True, and there are
                    # Multiple best_nc predictions for an input X
                    result = self._df_add_on_columns(result,
                                                    best_nc_df)

            elif index in self.optimalk_indicies:
                # Cluster with optimalk and kmeans
                assert clusterer == 'kmeans', ('Clusterer and index\
                                  are not compatable')

                min_nc = 2 # Static
                max_nc = self._get_max_nc(X) # Based on actual data

                best_nc_df = self._optimalk_calc(X_dim_reduced,
                                            index=index,
                                            clusterer=clusterer,
                                            distance=distance,
                                            min_nc=min_nc,
                                            max_nc=max_nc)

                if best_nc_df is None:
                    # The best number of clusters has not yet been predicted
                    # And best_nc_dataframe is None
                    result = best_nc_df
                else:
                    # Add predicted number of clusters to existing dataframe
                    # This is used when by_size is True, and there are
                    # Multiple best_nc predictions for an input X
                    result = self._df_add_on_columns(result,
                                                    best_nc_df)

            else:
                error_msg = ('Unknown clustering indicy passed. Please double-' +
                              'check index in in one of those listed')
                raise ValueError(error_msg)

        return result


    def _cluster(self, X, index, clusterer, n_components, distance, reduce):
        """Cluster inputs based on parameters
        inputs
        -------
        X : Raw data to be clustered

        outputs
        -------

        """

        # If there are less than 4 indicies there are problems with R indexing
        # And NbClust requirements. Dont cluster on less than 4 instances
        if X.shape[0] <= 4:
            return None

        # Perform dimensionality reduction on data
        X_dim_reduced = self._dimensionality_reduction(X,
                                                       method=reduce,
                                                       n_components=n_components)

        # Conditionally call nbclust package or optimalk package
        # based on input clustering hyperparameters
        if index in self.nbclust_indicies:
            # Cluster with nbclust and clustering algorithm
            min_nc = 2 # Static
            max_nc = self._get_max_nc(X) # Based on actual data

            best_nc_df = self._nbclust_calc(X_dim_reduced,
                                       index=index,
                                       clusterer=clusterer,
                                       distance=distance,
                                       min_nc=min_nc,
                                       max_nc=max_nc)

        elif index in self.optimalk_indicies:
            # Cluster with optimalk and kmeans
            assert clusterer == 'kmeans', ('Clusterer and index\
                             are not compatable')

            min_nc = 1 # Static
            max_nc = self._get_max_nc(X) # Based on actual data

            best_nc_df = self._optimalk_calc(X_dim_reduced,
                                       index=index,
                                       clusterer=clusterer,
                                       distance=distance,
                                       min_nc=min_nc,
                                       max_nc=max_nc)

        else:
            error_msg = ('Unknown clustering indicy passed. Please double-' +
                         'check index in in one of those listed')
            raise ValueError(error_msg)

        return best_nc_df


    def _df_add_on_columns(self, df1, df2):
        """Adds two datframes on their common columns. It assumes all dataframes
        have a common index
        inputs
        -------
        df1 : (dataframe)
        df2 : (dataframe)"""
        df1_cols = set(df1.columns)
        df2_cols = set(df2.columns)
        column_intersection = df1_cols.intersection(df2_cols)
        column_difference = df1_cols.difference(df2_cols)

        for col in column_difference:
            if col in df1_cols:
                df2 = df2.join(df1[col], how='left')
            elif col in df2_cols:
                df1 = df1.join(df2[col], how='left')

        for col in column_intersection:
            df1[col] = df1[col] + df2[col]

        return df1


    def _get_max_nc(self, X):
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


    def split_database_on_panel(self, document):
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


    def _parse_hyperparameter_dictionary(self, hyper_dict):
        """Parse hyperparameters into acceptable inputs for use in clusterer
        controllers. Hyperparameters might be input in incorrect fors (ex.
        CIndex instead of cindex).  This function will correct datatypes and
        capitalization of strings
        inputs
        -------
        hyper_dict : (dict) raw hyperparameter dictionary
        outputs
        -------
        parsed_clustering_params : (dict) of hyperparameters with correct
        dtypes and capitalization"""

        # by_size controller
        if isinstance(hyper_dict['by_size'], bool):
            by_size = hyper_dict['by_size']
        elif hyper_dict['by_size'] == 'True':
            by_size = True
        elif hyper_dict['by_size'] == 'False':
            by_size = False
        else:
            raise(ValueError('Fuck me unsupervised_cluster.py'))

        # index
        if hyper_dict['index'].lower() in self.nbclust_indicies:
            # just calculate all of them. Not worth it to do individual calcs
            index = 'all'

        elif hyper_dict['index'] in self.optimalk_indicies:
            # I did a bad job naming optimalk indicies initially
            # map the original names to the reduced set. The reduced set should be
            # one of ['gap_tib','gap_max','gap_star_max']
            index = ['gap_max','gap_max_star','gap_tib']

        # Clusterer controller
        if hyper_dict['clusterer'] == 'Ward.D':
            clusterer = 'ward.D'
        elif hyper_dict['clusterer'] == 'Ward.D2':
            clusterer = 'ward.D2'
        else:
            clusterer = hyper_dict['clusterer']

        # n_components
        n_components = int(hyper_dict['n_components'])

        # Method
        dimensionality_reduction_method = hyper_dict['reduce']

        # Distance
        distance = hyper_dict['distance'] # Static

        parsed_clustering_params =  {
                'by_size': by_size,
                'distance': distance,
                'clusterer': clusterer,
                'n_components': n_components,
                'reduce': dimensionality_reduction_method,
                'index': index}

        return parsed_clustering_params


    def _dimensionality_reduction(self, X, method, n_components):
        """Reduce dimensionality of encoded text
        inputs
        -------
        X : (np.array) array of encoded point names [n_ponits, vocab_size]
        method : (str) one of ['MDS','TSNE']
        n_components : (int) feature space to reduce towards
        ouput
        ------
        X : (np.array) reduced feature space array"""

        # Dimensionality reduction if called for
        if method=='MDS':
            mds = MDS(n_components = n_components)
            X_dim_reduce = mds.fit_transform(X)
        elif method=='TSNE':
            params = {'method':'barnes_hut',
                      'n_components':n_components,
                      'metric':'euclidean',
                      'perplexity':12}
            if n_components >= 4:
                params['method'] = 'exact' #Slower
            tsne = TSNE(method=params['method'],n_components=params['n_components'],
                        metric=params['metric'],perplexity=params['perplexity'])
            X_dim_reduce = tsne.fit_transform(X)
        else:
            # Do not apply dimensionality reduction
            X_dim_reduce = X

        return X_dim_reduce


    def _divide_indicies_by_size(self, X, by_size):
        """This divides an array of one-hot encoded words, X, into groups
        related to how many 'words' are in the point name. For example,
        two points, TJC.AHU01.SAT and TCJ.L01.EF01.SS, would be encoded like
        [[1,1,1,0,0,0],[1,0,0,1,1,1]]. They are different lengths, so the
        global_indicies returned would be [[0],[1]] where 0 is the indicy
        of the (3) lenght point, and 1 is the indicy of the (4) length point
        inputs
        -------
        X : (np.array) one-hot encoded points array. [n_points, vocab_size]
        shape
        by_size : (bool) whether to group by_size. If True, a list of indicies
        are returned that represent indicies of the first dimension of X where
        words are the same length
        output
        -------
        global_indicies : (list) list of array indicies where point name lengths
        are common"""
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


    def _nbclust_calc(self, X,
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
        # Calculate with NbClust method

        max_instances = 1200
        if X.shape[0] >= max_instances:
            # More than 1200 instances causes unacceptably slow calculations
            # (greater than 3 hours)
            msg = ('More than {} instances will caluse unacceptably slow '\
                   .format(max_instances) + 'calculation times')
            raise ValueError(msg)

        Nb_result = nbclust_calc(X,
                     min_nc=min_nc,
                     max_nc=max_nc,
                     distance=distance,
                     clusterer=clusterer,
                     index=index)

        df_bestnc = Nb_result.best_nc_df

        return df_bestnc


    def _optimalk_calc(self, X,
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


    @staticmethod
    def plt_gap(n_cluster_index, gap_values, optimal_k,
                artist, label='Gap1', correct_k=None):

        artist.plot(k_vals, gap_vals, linewidth=2, label=label)
        artist.scatter(optimal_k, gap_vals[optimal_k - 1], s=200, c='r')
        if correct_k:
            artist.axvline(x=correct_k, ymin=0.05, ymax=0.95, c='g', label='Correct k', linestyle='--')

        artist.grid(True)
        artist.set_xlabel('Cluster Count')
        artist.set_ylabel('Gap Values (mean(log(refDisps)) - log(origDisp))')
        artist.set_title('Gap Value v. Cluster Count')
        artist.legend()

        return None


    @staticmethod
    def plot_reduced_dataset(self, X):
        """Given a non-reduced dataset, perform dimensionality reduction and
        plot the dimensionality reduced dataset
        inputs
        ------
        X : (np.array) data to plot"""
        # Perform dimensionality reduction
        n_components=2
        mds = MDS(n_components = n_components)
        X_dim_reduce = mds.fit_transform(X)

        # Create a figure and artist (axis)
        figure = plt.figure(1)
        ax = figure.subplots(1,1)

        # Define data
        x = X_dim_reduce[:,0]
        y = X_dim_reduce[:,1]

        # Plot data
        artist.scatter(x, y)
        artist.set_title('MDS Reduction')
        artist.legend()
        artist.set_xlabel('$z_1$')
        artist.set_ylabel('$z_2$')
        artist.grid(True)

        return None

