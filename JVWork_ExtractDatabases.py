# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:19:51 2019

@author: z003vrzk
"""

import os
import subprocess
from pathlib import Path, WindowsPath
import shutil

def get_UNC():
    """Return a users mapped network drives. UNC path will be used for 
    connecting to networked database"""
    
    output = subprocess.run(['net', 'use'], stdout = subprocess.PIPE).stdout 
    output = output.decode() 
    alphabet = [chr(i) for i in range(65,91)]
    drives = []
    
    for letter in alphabet:
        if output.__contains__(letter + ':'):
            drives.append(letter)

    output = subprocess.run(['net', 'use'], stdout = subprocess.PIPE).stdout 
    output = output.decode() 

    alphabet = [chr(i) for i in range(65,91)]
    drives = []
    
    for letter in alphabet:
        if output.__contains__(letter + ':'):
            drives.append(letter)
    
    #get UNC server names
    output = output.splitlines()
    serverUNC = []
    
    for lines in output:
        if lines.__contains__('\\'):
            serverUNC.append(lines[lines.index('\\'):len(lines)-1])
    myOutput = {}
    
    for index, letter in enumerate(drives):
        myOutput[letter] = serverUNC[index]
        
    return myOutput

def search_databases(base_directory, destination_base, idx=1, print_flag=False):
    """Base function for module. Recursively looks through base_directory
    and copies any instances of SQL databases named JobDB.mdf and JobLog.ldf
    to a default directory on the D drive
    parameters
    -------
    base_directory : base directory to look in; recursive from here
    destination_base : where to save databases when found
    idx : for printing (optional)
    print_flag : enable print (optional)
    output
    -------
    (None) Saves databases to default directory"""
    
    for _dir in os.listdir(base_directory):
        current_dir = Path(os.path.join(base_directory, _dir))
        
        if current_dir.is_dir():
            databases = sorted(current_dir.glob('*.mdf'))
            if print_flag:
                print('{}Current directory is directory, name : {}'
                          .format(chr(62)*idx,current_dir))
    
            
            if len(databases) == 0: #Nothing Found
                search_databases(current_dir, destination_base=destination_base, idx=idx+1)
            
            else: #databases found
                logs = sorted(current_dir.glob('*.ldf'))
                
                for database_path, log_path in zip(databases, logs):
                
                    if print_flag:
                        print('{}Database found : {}'.format(chr(62)*idx,str(str(database_path))))
                        print('{}Log found : {}'.format(chr(62)*idx,str(str(log_path))))
                        
                    save_databases(database_path, log_path, destination_base=destination_base)
                    
                search_databases(current_dir, destination_base=destination_base, idx=idx+1)
            
        else:
            
            if print_flag:
                print('{}File Name : {}'.format(chr(62)*idx,current_dir))
            pass
    

def save_databases(source_path, log_path, destination_base=r'D:\Z - Saved SQL Databases'):
    """Saves a database source_path to destination_base. Handles new 
    naming structure
    Inputs
    -------
    source_path : database path
    log_path : log of database path
    destination_base : destination folder to save to (default D:\Z - Saved SQL Databases)
    Outputs
    -------
    True : database in source_path was successfully saved
    False : Database was not successfully saved"""
    
    assert type(source_path) is Path() or WindowsPath(), 'TypeError : Source path input should be Path object'
    
    folder_name = get_database_name(str(source_path), destination_base=destination_base)
    
    if not folder_name:
        print('\nDatabase for {} already exists in {}. Folder skipped'.format(
                source_path,destination_base))
        return
    
    destination_folder = os.path.join(destination_base, folder_name)
    try:
        os.mkdir(destination_folder)
    except FileExistsError:
        print('Folder {} Already exists. Repeat database skipped'.format(
                destination_folder))
        return
    
    try:
        shutil.copy(source_path, destination_folder)
        shutil.copy(log_path, destination_folder)
    except PermissionError:
        print('Permission Denied on {}. Folder not copied'.format(source_path))
        return


def get_database_name(source_path, destination_base):
    """Decide on the name of the folder holding the database
    #Check to see if that name already exists
    #Return appropriate name, or an error if the database exists"""
    
    assert type(source_path) is str, 'Source path must be string'
    
    split_path = source_path.split('\\')
    match_string = ['44OP','44op']
    
    def list_intersect_contain(split_path, _match):
        """Find the job name that contains _match in the split path
        If not found return false"""
        
        for path in split_path:
            
            for _x in _match:
                
                if path.__contains__(_x):
                    return path
                
        return False
    
    def check_name_exist(folder_name, destination_base):
        """Checks if folder_name already exists in destination_base
        Returns True if folder exists
        False if does not exist"""
        
        base_dir_list = os.listdir(destination_base)
        
        for base_dir in base_dir_list:
            if base_dir.__contains__(folder_name):
                return True
            else:
                return False
    
    job_name = list_intersect_contain(split_path, match_string)
    
    if type(job_name) is str:
        folder_name = job_name
        
        if check_name_exist(folder_name, destination_base):
            return False #Do not create a new folder b/c it already exist
        
        else:
            return folder_name
    
    elif job_name is False: 
        #return generic_name
        base_dir_list = os.listdir(destination_base)
        matching_dir = []
        
        for base_dir in base_dir_list:
            
            if base_dir.__contains__('No_Name_'):
                matching_dir.append(base_dir)
        idx = []
        
        for name in matching_dir:
            _idx = int(name[name.rindex('_') + 1:len(name)])
            idx.append(_idx)
            
        generic_dir = r'No_Name_'
        try:
            _maxidx = max(idx)
        except:
            _maxidx = 0
        index = int(_maxidx + 1)
        folder_name = generic_dir + str(index)
        
        return folder_name





