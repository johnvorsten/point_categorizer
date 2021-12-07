# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:14:38 2021

@author: jvorsten
"""

# Python imports
from dataclasses import dataclass
from typing import Union, List

# Third party imports
from pydantic import BaseModel
import pandas as pd

# Local imports

# Declarations
"""There are a total of 10 types of predictions that a predictor can choose
from: ['ahu', 'alarm', 'boiler', 'chiller', 'exhaust_fan', 'misc', 'room',
       'rtu', 'skip', 'unknown'].
In the event that a predictor has an unknown label, this class will restrict
the possible outputs from our API.
It is unfortunate that these preidction labels were chosen. 'skip' and
'unknown' are very similar class labels, and I'd like to combine them into
one result. 'room','misc' are also unimportant, and I'd like to lump them 
under the unknown class label"""
prediction_map = {
    'ahu':'Air Handler',
    'alarm':'Alarm',
    'boiler':'Boiler or Hot Water System',
    'chiller':'Chiller, CHW pump, or Condenser System',
    'exhaust_fan':'Exhaust System',
    'misc':'Unknown System',
    'room':'Unknown System',
    'rtu':'Rooftop Unit',
    'skip':'Unknown System',
    'unknown':'Unknown System',
    }

#%% Classes

class RawInputDataPydantic(BaseModel):
    """Raw input data from HTTP Web form
    Note, any default values are not required by the estimator, and are 
    removed by the data cleaning pipeline
    
    Required numeric attributes
    ['DEVICEHI', 'DEVICELO', 'SIGNALHI', 'SIGNALLO', 'SLOPE', 'INTERCEPT']
    
    Required categorical attributes
    ['TYPE', 'ALARMTYPE', 'FUNCTION', 'VIRTUAL', 'CS','SENSORTYPE', 'DEVUNITS']
    
    Requried text attributes
    ['NAME', 'DESCRIPTOR']
    """
    # Required numeric attributes
    DEVICEHI: float
    DEVICELO: float
    SIGNALHI: float
    SIGNALLO: float
    SLOPE: float
    INTERCEPT: float
    # Required categorical attributes
    TYPE: str
    ALARMTYPE: str
    FUNCTION: str
    VIRTUAL: bool
    CS: str
    SENSORTYPE: str
    DEVUNITS: str
    # Requried text attributes
    NAME: str
    DESCRIPTOR: str
    NETDEVID: str
    SYSTEM: str


class PredictorOutput(BaseModel):
    prediction: str

@dataclass
class RawInputData:
    """Raw input data from HTTP Web form
    Note, any default values are not required by the estimator, and are 
    removed by the data cleaning pipeline
    
    Required numeric attributes
    ['DEVICEHI', 'DEVICELO', 'SIGNALHI', 'SIGNALLO', 'SLOPE', 'INTERCEPT']
    
    Required categorical attributes
    ['TYPE', 'ALARMTYPE', 'FUNCTION', 'VIRTUAL', 'CS','SENSORTYPE', 'DEVUNITS']
    
    Requried text attributes
    ['NAME', 'DESCRIPTOR']
    """
    # Required numeric attributes
    DEVICEHI: float
    DEVICELO: float
    SIGNALHI: float
    SIGNALLO: float
    SLOPE: float
    INTERCEPT: float
    # Required categorical attributes
    TYPE: str
    ALARMTYPE: str
    FUNCTION: str
    VIRTUAL: bool
    CS: str
    SENSORTYPE: str
    DEVUNITS: str
    # Requried text attributes
    NAME: str
    DESCRIPTOR: str
    NETDEVID: str
    SYSTEM: str


def transform_data_pydantic(data:Union[List[RawInputDataPydantic],
                                        RawInputDataPydantic]) -> pd.DataFrame:
    if isinstance(data, list):
        df_raw = pd.DataFrame([x.__dict__ for x in data])
    elif isinstance(data, RawInputDataPydantic):
        df_raw = pd.DataFrame([data.__dict__])
    else:
        msg=("Expected type of List[RawInputDataPydantic] or "+
             "RawInputDataPydatic, instead got {}")
        raise ValueError(msg.format(type(data)))
        
    return df_raw

def transform_data_dataclass(data:Union[List[RawInputData],
                                         RawInputData]) -> pd.DataFrame:
    if isinstance(data, list):
        df_raw = pd.DataFrame(data)
    elif isinstance(data, RawInputData):
        df_raw = pd.DataFrame([data])
    else:
        msg=("Expected type of List[RawInputData] or "+
             "RawInputData, instead got {}")
        raise ValueError(msg.format(type(data)))
        
    return df_raw
