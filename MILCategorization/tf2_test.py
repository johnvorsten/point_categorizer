# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:54:16 2020

@author: z003vrzk
"""
# Python imports
import os, sys
import pdb

# Third party imports
import tensorflow as tf

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

#%%

"""Notice the function wrapper - it compiles a python function into a callable
tensorflow graph"""
@tf.function
def f(x):
  if any(x > 0):
    pdb.set_trace()
    x = x + 1
  return x

@tf.function
def f2(x, y):
    return x ** 2 + y

tf.config.experimental_run_functions_eagerly(True)
x = tf.constant([1])
f(x)
x = tf.constant([2,34,5,6])
f(x)

x = tf.constant([2,3])
y = tf.constant([4,4])
f2(x,y)
