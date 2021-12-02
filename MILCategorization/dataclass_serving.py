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
