# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:54:21 2019

@author: z003vrzk
"""
import matplotlib.pyplot as plt
from JVWork_UnClusterAccuracy import AccuracyTest
from JVWork_UnsupervisedCluster import JVClusterTools
from JVWork_WholeDBPipeline import JVDBPipe
import numpy as np
import pandas as pd
from sklearn.manifold import MDS, TSNE
from JVrpy2 import nbclust_calc
import time
import matplotlib.pyplot as plt
class CustomProcessError(Exception):
    pass
from concurrent.futures.process import BrokenProcessPool
from rpy2.rinterface import RRuntimeError

myTest = AccuracyTest()
myClustering = JVClusterTools()
_master_pts_db = r"D:\Z - Saved SQL Databases\master_pts_db.csv"
my_iter = myClustering.read_database_set(_master_pts_db)


#%%
"""For hand testing"""
myTest = AccuracyTest()
myClustering = JVClusterTools()
myDBPipe = JVDBPipe()

_master_pts_db = r"D:\Z - Saved SQL Databases\master_pts_db.csv"
my_iter = myClustering.read_database_set(_master_pts_db)

sequence_tag = 'DBPath'
_, database = next(my_iter)
error_df = myTest.error_df

col = np.where(database.columns == sequence_tag)[0][0]
db_name = database.iloc[0, col]
print('\n{}'.format(db_name))
    
clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=False, 
                                      replace_numbers=False, 
                                      remove_virtual=True)
df_clean = clean_pipe.fit_transform(database)
text_pipe = myDBPipe.text_pipeline(vocab_size='all', attributes='NAME',
                                   seperator='.')
X = text_pipe.fit_transform(df_clean).toarray()
_word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
df_text = pd.DataFrame(X, columns=_word_vocab)



#%%
"""Testing the new NbClust implementation"""
if not 'my_iter2' in locals().keys():
    print('Classes instantiated')
    myTest2 = AccuracyTest()
    myClustering2 = JVClusterTools()
    myDBPipe2 = JVDBPipe()
    _master_pts_db = r"D:\Z - Saved SQL Databases\master_pts_db.csv"
    my_iter2 = myClustering.read_database_set(_master_pts_db)
    time_list = []
    time_list_reduce = []

#Constants
sequence_tag = 'DBPath'
n_components=5
method = 'MDS'
clusterer = 'ward.D'
distance = 'euclidean'
index = 'all'
min_nc = 2

for _i in range(0,20):
    _, database = next(my_iter2)
    
    col = np.where(database.columns == sequence_tag)[0][0]
    db_name = database.iloc[0, col]
    clean_pipe = myDBPipe2.cleaning_pipeline(remove_dupe=False, 
                                          replace_numbers=False, 
                                          remove_virtual=True)
    df_clean = clean_pipe.fit_transform(database)
    text_pipe = myDBPipe2.text_pipeline(vocab_size='all', attributes='NAME',
                                       seperator='.')
    X = text_pipe.fit_transform(df_clean).toarray()
    _word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
    df_text = pd.DataFrame(X, columns=_word_vocab)
    correct_k = myTest.get_correct_k(db_name, df_clean, manual=True)
    max_nc = myTest.get_max_iterations(X, X, correct_k)
    
    _t0 = time.time()
    
    if method=='MDS':
        mds = MDS(n_components = n_components)
        X_reduced = mds.fit_transform(X)
    elif method=='TSNE':
        params = {'method':'barnes_hut', 
                  'n_components':n_components,
                  'metric':'euclidean',
                  'perplexity':12}
        if n_components >= 4:
            params['method'] = 'exact' #Slower
        tsne = TSNE(method=params['method'],n_components=params['n_components'],
                    metric=params['metric'],perplexity=params['perplexity'])
        X_reduced = tsne.fit_transform(X)
    
    print('{}\nDB Size : {}'.format(db_name, X_reduced.shape))
    
    if X_reduced.shape[0] >= 800:
        print('SKIPPED')
        continue
    
    print('Starting NbClust')
    
    _t01 = time.time()
    time_list_reduce.append((_t01-_t0, X.shape))
    
    try:
        Nb_result = nbclust_calc(X_reduced, 
                                 min_nc=min_nc, max_nc=max_nc-1,
                                 distance=distance, method=clusterer,
                                 index=index)
    except RRuntimeError as ex:
        print(f'{ex} An error happened')
        continue
    
    _t1 = time.time()
    #df_index = Nb_result.index_df
    #df_bestnc = Nb_result.best_nc_df
    
    print('Elapsed time : {} on database size {}\n'.format(_t1-_t0, X_reduced.shape))
    time_list.append((_t1-_t0, X_reduced.shape))


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

xs = [shape[0] for (_time, shape) in time_list]
ys = [_time for (_time, shape) in time_list]
xs_reduced = [shape[0] for (_time, shape) in time_list_reduce]
ys_reduced = [_time for (_time, shape) in time_list_reduce]

#Get Data
Xs = np.array(xs)
Ys = np.array(ys)
Xs_reduced = np.array(xs_reduced)
Ys_reduced = np.array(ys_reduced)

#Fit Data
poly_features = PolynomialFeatures(degree=4, include_bias=True)
x_poly = poly_features.fit_transform(Xs.reshape(-1,1))
lin_reg = LinearRegression()
lin_reg.fit(x_poly, Ys)
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(x_poly, Ys)

x_poly_reduce = poly_features.fit_transform(Xs_reduced.reshape(-1,1))
lin_reg_reduce = LinearRegression()
lin_reg_reduce.fit(x_poly_reduce, Ys_reduced)
lasso_reg_reduce = Lasso(alpha=0.3)
lasso_reg_reduce.fit(x_poly_reduce, Ys_reduced)

#Predict instances
_x = np.arange(10,4000,100)
x_test = poly_features.fit_transform(_x.reshape(-1,1))
_time_poly = lin_reg.predict(x_test)
_time_lasso = lasso_reg.predict(x_test)
_time_poly_reduce = lin_reg_reduce.predict(x_test)
_time_lasso_reduce = lasso_reg_reduce.predict(x_test)

#Plot data
fig, ax = plt.subplots(1)
ax.plot(_x, _time_poly, label='poly_Nb')
ax.plot(_x, _time_lasso, label='lasso_Nb')
ax.plot(_x, _time_poly_reduce, label='poly reduce')
ax.plot(_x, _time_lasso_reduce, label='lasso reduce')
ax.scatter(xs, ys, c='r', label='Nb Compute')
ax.scatter(Xs_reduced, Ys_reduced, label='Reduce')

ax.set_title('Predicted and measure computation time')
ax.set_xlabel('Database Size (indstances)')
ax.set_ylabel('Time (seconds)')
ax.set_xlim(0, max(_x) + 100)
ax.set_ylim(ax.get_xlim())
ax.legend()


#%%
"""iterate() method and iterate_recalc() method"""
#help(myTest.iterate)
#help(myTest.iterate_recalc)


#for _, database in my_iter:
#    try:
#        error_df = myTest.iterate(database, manual=True, skip=False, 
#                                  by_size=True, method='MDS', 
#                                  n_components=2, plot=False)
#    except BrokenProcessPool as ex:
#        raise CustomProcessError(f'{ex} There was an unknown error, possibly'
#                 ' because of limitied system resources.')


hyper_dict = {'by_size':[True, False],
              'clusterer':['kmeans','ward.D','ward.D2','single','average'],
              'distance':['eucledian', 'minkowski'],
              'reduce':['MDS', 'TSNE', False],
              'n_components':[2,5,10]}

by_size = True
clusterer = 'ward.D'
distance = 'euclidean'
n_components = 8
method = 'MDS'
index = 'alllong'
save_path = r'.\error_df 8-17 no1.csv'
hyper_info = pd.DataFrame({'by_size':[by_size],
              'clusterer':[clusterer],
              'distance':[distance],
              'reduce':[method],
              'n_components':[n_components]})
hyper_info.to_csv(save_path)

for _, database in my_iter:
    try:
        error_df = myTest.iterate_recalc(database, by_size=by_size, standard=True,
                             reduce=True, method=method, n_components=n_components,
                             nbclust=True, index_nb=index, clusterer=clusterer,
                             distance=distance)
                                              
    except BrokenProcessPool as ex:
        raise CustomProcessError(f'{ex} There was an unknown error, possibly'
                 ' because of limitied system resources.')
        
    except RRuntimeError as ex:
        print(r'{ex} There was an unknown error')
        print(str(ex))
        try:
            if str(ex).__contains__('singular'):
                    error_df = myTest.iterate_recalc(database, by_size=by_size, standard=True,
                                 reduce=True, method=method, n_components=int(n_components/2),
                                 nbclust=True, index_nb=index, clusterer=clusterer,
                                 distance=distance)
            else:
                pass
        except RRuntimeError:
            pass
        
error_df.to_csv(save_path, mode='a')
            


#%% Testing 8-17
by_size = True
clusterer = 'ward.D'
distance = 'euclidean'
n_components = 8
method = 'MDS'
index = 'alllong'
save_path = r'.\error_df 8-17 no1.csv'


_, database = next(my_iter)

error_df = myTest.iterate_recalc(database, by_size=by_size, standard=True,
                     reduce=True, method=method, n_components=n_components,
                     nbclust=True, index_nb=index, clusterer=clusterer,
                     distance=distance)


error_df.to_csv(save_path, mode='a')


