# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 20:30:34 2019

@author: z003vrzk
"""

from rpy2 import robjects
print('This is how you access an R object : "pi_object = rpy2.robjects.r.pi" ')
print('This is how you access the objects value : \n{} or \n{}'.format(
        'pi_val = pi_object[0]',
        'pi_val = pi_object.__getitem__(0)'))


pi_object = robjects.r.pi #Float vector object
pi_val = pi_object[0]
pi_val = pi_object.__getitem__(0)

#How to call R functions in python
print('How to call R functions in python')
from rpy2.robjects.packages import importr
base = importr('base')
stats = importr('stats')
graphics = importr('graphics')

plot = graphics.plot
rnorm = stats.rnorm
plot(rnorm(100), ylab="random")

#The module im interested in
print('How to import r packages in python')

#Check to see if module is installed
import rpy2
_module_name = 'NbClust'
if rpy2.robjects.packages.isinstalled(_module_name):
    print('{} is already Installed'.format(_module_name))
nbclust = importr('NbClust') #Import the NbClust package?
_module_path = rpy2.robjects.packages.get_packagepath('NbClust')
nbclust2 = importr('NbClust', lib_loc=_module_path)

#Test Data
import numpy as np
print('Create data and use the r2py & NbClust module')
data = np.random.rand(32).reshape(-1,2)

#Print the modules directory
nbclust.__dict__
nbclust.__dir__()

print('OOPs! It looks like the above wont work because we cant pass a \
      numpy array to an robject module/function without some help')
#nbclust.NbClust(data, diss='NULL', distance='euclidean',
#                method='kmeans', index='all')

#BUT WAIT : we need to convert numpy to an R object
print('From rpy2.robjects.numpy2ri import numpy2ri\
      Rpy2.robjects.numpy2ri.activate()')
from rpy2.robjects.numpy2ri import numpy2ri
#from rpy2.robjects import pandas2ri
import pandas as pd
#ro.conversion.py2ri = numpy2ri
rpy2.robjects.numpy2ri.activate()
#pandas2ri.activate()

answer = nbclust.NbClust(data, distance='euclidean',
                min_nc=2, max_nc=10,
                method='kmeans', index='all')
answer.__dict__
answer.__dir__()

for key, item in answer.items():
    print(key, item)
for name in answer.names:
    print(name)
for col in answer[0].colnames:
    print(col)

answer[0].r_repr()
answer[0].named
answer[0].colnames
answer[0].ncol
answer[0].nrow
answer[0].rownames
answer[0].__dict__

for key, value in answer.items():
    print('{} : {}'.format(key, type(value)))

"""Extracting NbClsut returned information"""
import rpy2
from rpy2.robjects.numpy2ri import numpy2ri
rpy2.robjects.numpy2ri.activate()

def rmatrix_2df(r_matrix):
    assert type(r_matrix) is rpy2.robjects.Matrix, 'r_matrix argument is not\
    type rpy2.robjects.Matrix'
    values = np.array(r_matrix)
    row_names = list(r_matrix.rownames)
    col_names = list(r_matrix.colnames)
    df = pd.DataFrame(data=values, 
                      index=row_names, 
                      columns=col_names)
    return df

all_index = rmatrix_2df(answer[0])
all_crit = rmatrix_2df(answer[1])
best_nc = rmatrix_2df(answer[2])
best_partition = np.array(answer[3])





