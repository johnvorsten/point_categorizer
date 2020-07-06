# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:17:17 2020

@author: z003vrzk
"""
# Python imports
import sys
import os


# Third party imports


# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

from MILCategorization import mil_load

LoadMIL = mil_load.LoadMIL()


#%%

if __name__ == '__main__':
    # Pipeline w/ numeric features
    bags, labels = LoadMIL.gather_mil_dataset(pipeline='whole')

    # Save
    file_name=r'../data/MIL_dataset.dat'
    LoadMIL.save_mil_dataset(bags, labels, file_name)

    # Retrieve
    dataset = LoadMIL.load_mil_dataset(file_name)

    # Pipeline w/o numeric features
    bags, labels = LoadMIL.gather_mil_dataset(pipeline='categorical')

    # Save
    file_name=r'../data/MIL_cat_dataset.dat'
    LoadMIL.save_mil_dataset(bags, labels, file_name)

