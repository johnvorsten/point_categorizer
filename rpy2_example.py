# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 20:30:34 2019

@author: z003vrzk
"""

import rpy2
print('This is how you access an R object : "pi_object = rpy2.robjects.r.pi" ')
print('This is how you access the objects value : \n{} or \n{}'.format(
        'pi_val = pi_object[0]',
        'pi_val = pi_object.__getitem__(0)'))


pi_object = rpy2.robjects.r.pi #Float vector object
pi_val = pi_object[0]
pi_val = pi_object.__getitem__(0)

#How to call R functions in python
from rpy2.robjects.packages import importr
base = importr('base')
stats = importr('stats')
graphics = importr('graphics')

plot = graphics.plot
rnorm = stats.rnorm
plot(rnorm(100), ylab="random")

#The module im interested in
from rpy2.robjects.packages import importr

#Check to see if module is installed
_module_name = 'NbClust'
if rpy2.robjects.packages.isinstalled(_module_name):
    print('{} Installed'.format(_module_name))
nbclust = importr('NbClust') #Import the NbClust package?
_module_path = rpy2.robjects.packages.get_packagepath('NbClust')
nbclust2 = importr('NbClust', lib_loc=_module_path)

#Test Data
import numpy as np
data = np.random.rand(32).reshape(-1,2)
nbclust(data)

#Print the modules directory
nbclust.__dict__
nbclust.__dir__()
nbclust.__doc__
nbclust_sub = nbclust.__init_subclass__()
nbclust.NbClust(data)
