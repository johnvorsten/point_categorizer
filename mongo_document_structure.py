# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 20:50:35 2019

@author: z003vrzk
"""
database_tag = r'D:\path\to\database'
columns = ['POINTID', 'NETDEVID', 'NAME', 'CTSYSNAME', 'DESCRIPTOR', 'TYPE',
       'INITVALUE', 'TMEMBER', 'ALARMTYPE', 'ALARMHIGH', 'ALARMLOW', 'COMBOID',
       'FUNCTION', 'VIRTUAL', 'PROOFPRSNT', 'PROOFDELAY', 'NORMCLOSE',
       'INVERTED', 'LAN', 'DROP', 'POINT', 'ADDRESSEXT', 'SYSTEM', 'CS',
       'DEVNUMBER', 'SENSORTYPE', 'CTSENSTYPE', 'CONTRLTYPE', 'UNITSTYPE',
       'DEVICEHI', 'DEVICELO', 'DEVUNITS', 'SIGNALHI', 'SIGNALLO', 'SIGUNITS',
       'NUMBERWIRE', 'POWER', 'WIRESIZE', 'WIRELENGTH', 'S1000TYPE', 'SLOPE',
       'INTERCEPT', 'DBPath']
values = ['example', 'example', 'example', 'example', 'example', 'example', 'example',
 'example', 'example', 'example', 'example', 'example', 'example', 'example', 
 'example', 'example', 'example', 'example', 'example', 'example', 'example',
 'example', 'example', 'example', 'example', 'example', 'example', 'example', 
 'example', 'example', 'example', 'example', 'example', 'example', 'example', 
 'example', 'example', 'example', 'example', 'example', 'example', 'example', 'example']

mongo_structure = {'_id':ObjectID,
                     'database_tag':database_tag,
                     'points':{columns:values}, #Example
                     'db_features':{<key>:<Array>},
                     'hyper_labels':{<string_key>:{'by_size':boolean, 
                                                   'distance':<string>,
                                                   'clusterer':<string>, 
                                                   'n_components':<integer>,
                                                   'reduce':<string>, 
                                                   'index':<string>,
                                                   'loss':<float>},
                                    }
                    'best_hyper':{'by_size':<boolean>,
                                  'n_components':<integer>,
                                  'reduce':<string>,
                                  'clusterer':<Array>,
                                  'index':<Array>}
                     }