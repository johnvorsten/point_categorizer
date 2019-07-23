# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:00:16 2019

@author: z003vrzk
"""

import JVWork_ExtractDatabases as JVExtract
import os



"""Test"""
#base_directory = r'D:\SQLTest'
#destination_base = r'D:\Z - Saved SQL Databases'
#JVExtract.search_databases(base_directory, destination_base=destination_base)
#
#
#base_directory = r"R:\JOBS\44OP-148606_BU_RANEY_SMITH"
#destination_base = r'D:\Z - Saved SQL Databases'
#JVExtract.search_databases(base_directory, destination_base=destination_base)
#
#
#base_directory = r"R:\JOBS\44OP-235077 DCMC Cath Lab Fan Coils"
#destination_base = r'D:\Z - Saved SQL Databases'
#JVExtract.search_databases(base_directory, destination_base=destination_base)


base_directory = r"R:\JOBS"
base_dir_list = os.listdir(base_directory)
destination_base = r'D:\Z - Saved SQL Databases'
JVExtract.search_databases(base_directory, destination_base=destination_base)