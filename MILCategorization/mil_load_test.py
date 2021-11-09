# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:28:03 2020

@author: z003vrzk
"""

# Python imports
import sys
import os

# Third party imports

# Local imports
import mil_load

if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)


# Declarations
LoadMIL = mil_load.LoadMIL()

#%%

def test_bag_data_generator():

    bag_generator = LoadMIL.bag_data_generator(verbose=False)
    bag = next(bag_generator)

    return None


def test_validate_bag():

    bag_generator = LoadMIL.bag_data_generator(verbose=False)
    bag = next(bag_generator)

    valid_bool = LoadMIL.validate_bag(bag)
    if valid_bool:
        print('Bag is Valid :)')
    else:
        print('Bag is NOT Valid :(')

    return None



def test_mil_transform_pipeline():

    mil_transform_pipeline = LoadMIL.mil_transform_pipeline()

    return None


def test_gather_mil_dataset():
    return None


def test_load_mil_dataset():

    file_name = r'../data/MIL_dataset.dat'
    dataset =LoadMIL.load_mil_dataset(file_name)

    return None