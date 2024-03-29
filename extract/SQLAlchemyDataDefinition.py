# -*- coding: utf-8 -*-
"""
Created on Sun May 10 10:51:04 2020

SQLAlchemy Table Definitions

@author: z003vrzk
"""

# Python imports
import os
import sys

# Third party imports
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.mssql import NVARCHAR, NUMERIC, BIT, VARBINARY
from sqlalchemy import create_engine
from sqlalchemy import CheckConstraint
import numpy as np

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
from extract.sql_tools import SQLBase

# Table schema
Base = declarative_base()

#%%


# Define tables
class Points(Base):
    __tablename__ = "points"

    id = Column(Integer, primary_key=True)
    POINTID = Column('POINTID', NUMERIC(19,5), nullable=False) #
    DEVICEHI = Column('DEVICEHI', NUMERIC(19,5)) #
    DEVICELO = Column('DEVICELO', NUMERIC(19,5)) #
    SIGNALHI = Column('SIGNALHI', NUMERIC(19,5)) #
    SIGNALLO = Column('SIGNALLO', NUMERIC(19,5)) #
    NUMBERWIRE = Column('NUMBERWIRE', NUMERIC(19,5)) #
    WIRELENGTH = Column('WIRELENGTH', NUMERIC(19,5)) #
    WIRESIZE = Column('WIRESIZE', NUMERIC(19,5)) #
    PROOFDELAY = Column('PROOFDELAY', NUMERIC(19,5)) #

    NORMCLOSE = Column('NORMCLOSE', BIT()) #
    INVERTED = Column('INVERTED', BIT()) #
    PROOFPRSNT = Column('PROOFPRSNT', BIT()) #
    VIRTUAL = Column('VIRTUAL', BIT()) #
    TMEMBER = Column('TMEMBER', BIT()) #

    ADDRESSEXT = Column('ADDRESSEXT', NVARCHAR(20)) #
    ALARMHIGH = Column('ALARMHIGH', NVARCHAR(30)) #
    ALARMLOW = Column('ALARMLOW', NVARCHAR(30)) #
    ALARMTYPE = Column('ALARMTYPE', NVARCHAR(30)) #
    COMBOID = Column('COMBOID', NVARCHAR(30)) #
    CONTRLTYPE = Column('CONTRLTYPE', NVARCHAR(10)) #
    CS = Column('CS', NVARCHAR(30)) #
    CTSENSTYPE = Column('CTSENSTYPE', NVARCHAR(15)) #
    CTSYSNAME = Column('CTSYSNAME', NVARCHAR(30)) #
    DESCRIPTOR = Column('DESCRIPTOR', NVARCHAR(30))
    DEVNUMBER = Column('DEVNUMBER', NVARCHAR(30)) #
    DEVUNITS = Column('DEVUNITS', NVARCHAR(10)) #
    DROP = Column('DROP', NVARCHAR(3)) #
    FUNCTION = Column('FUNCTION', NVARCHAR(10)) #
    INITVALUE = Column('INITVALUE', NVARCHAR(16)) #
    INTERCEPT = Column('INTERCEPT', NVARCHAR(30)) #
    LAN = Column('LAN', NVARCHAR(3)) #
    NAME = Column('NAME', NVARCHAR(30), nullable=False) #
    NETDEVID = Column('NETDEVID', NVARCHAR(30)) #
    POINT = Column('POINT', NVARCHAR(30)) #
    POINTACTUAL = Column('POINTACTUAL', NVARCHAR(30)) #
    POWER = Column('POWER', NVARCHAR(8)) #
    S1000TYPE = Column('S1000TYPE', NVARCHAR(10)) #
    SENSORTYPE = Column('SENSORTYPE', NVARCHAR(30)) #
    SIGUNITS = Column('SIGUNITS', NVARCHAR(6)) #
    SLOPE = Column('SLOPE', NVARCHAR(16)) #
    SYSTEM = Column('SYSTEM', NVARCHAR(30)) #

    TYPE = Column('TYPE', NVARCHAR(5)) #
    UNITSTYPE = Column('UNITSTYPE', NVARCHAR(3))

    # Relationships
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False)
    customer = relationship('Customers',back_populates='')
    # Many points are grouped under one label
    group_id = Column(Integer, ForeignKey('labeling.id'), nullable=True)
    group = relationship('Labeling', back_populates='points')

    def __repr__(self):
        return "<points(name='%s', type='%s', 'id='%s')>"\
            % (self.NAME, self.TYPE, self.POINTID)



class Netdev(Base):
    __tablename__ = "netdev"

    id = Column(Integer, primary_key=True)
    ADDRESS1 = Column('ADDRESS1', NVARCHAR(40))
    ADDRESS2 = Column('ADDRESS2', NVARCHAR(40))
    ADDRESS3 = Column('ADDRESS3', NVARCHAR(40))
    ADDRNAME1 = Column('ADDRNAME1', NVARCHAR(40))
    ADDRNAME2 = Column('ADDRNAME2', NVARCHAR(40))
    ADDRNAME3 = Column('ADDRNAME3', NVARCHAR(40))
    ADDRSTYLE = Column('ADDRSTYLE', NVARCHAR(40))
    BAUDRATE = Column('BAUDRATE', NVARCHAR(40))
    CTSYSNAME = Column('CTSYSNAME', NVARCHAR(40))
    DESCRIPTOR = Column('DESCRIPTOR', NVARCHAR(40))
    DNS = Column('DNS', NVARCHAR(40))
    DWG_NAME = Column('DWG_NAME', NVARCHAR(40))
    FLNTYPE = Column('FLNTYPE', NVARCHAR(40))
    INSTANCE = Column('INSTANCE', NVARCHAR(40))
    MACADDRESS = Column('MACADDRESS', NVARCHAR(40))
    NAME = Column('NAME', NVARCHAR(40))
    NETDEVID = Column('NETDEVID', NVARCHAR(40))
    NODEADDR = Column('NODEADDR', NVARCHAR(40))
    ONETOFOUR = Column('ONETOFOUR', NVARCHAR(40))
    PARENTID = Column('PARENTID', NVARCHAR(40))
    PARTNO = Column('PARTNO', NVARCHAR(40))
    REFERNAME = Column('REFERNAME', NVARCHAR(40))
    SITENAME = Column('SITENAME', NVARCHAR(40))
    STARTADDR = Column('STARTADDR', NUMERIC(19,5))
    SUBTYPE = Column('SUBTYPE', NVARCHAR(40))
    TBLOCKID = Column('TBLOCKID', NUMERIC(19,5))
    TYPE = Column('TYPE', NVARCHAR(40))

    # Relationships
    customer_id = Column(Integer, ForeignKey('customers.id'))
    customer = relationship('Customers', back_populates='netdev')

    def __repr__(self):
        return "<netdev(name='%s', 'id='%s')>"\
            % (self.NETDEVID, self.id)


class Customers(Base):
    __tablename__ = 'customers'

    id = Column(Integer, primary_key=True)
    name = Column(String(150), nullable=False, unique=True)
    correct_k = Column(Integer, nullable=True)

    # Relationships
    points = relationship('Points', back_populates='customer', order_by=Points.id)
    netdev = relationship('Netdev', back_populates='customer', order_by=Netdev.id)

    def __repr__(self):
        return"<customer(name='%s')>" % (self.name)


class Clustering(Base):
    __tablename__ = 'clustering'

    id = Column(Integer, primary_key=True)

    # Clustering results in pickled form
    best_nc = Column('best_nc', VARBINARY(), nullable=True) # Pickled dataframe
    indicies = Column('indicies', VARBINARY(), nullable=True) # Pickled dataframe

    # Database features
    correct_k = Column('correct_k', Integer, nullable=False)
    n_points = Column('n_points', Integer, nullable=False)
    n_len1 = Column('n_len1', Integer, nullable=False)
    n_len2 = Column('n_len2', Integer, nullable=False)
    n_len3 = Column('n_len3', Integer, nullable=False)
    n_len4 = Column('n_len4', Integer, nullable=False)
    n_len5 = Column('n_len5', Integer, nullable=False)
    n_len6 = Column('n_len6', Integer, nullable=False)
    n_len7 = Column('n_len7', Integer, nullable=False)

    # Clustering results in explicit form (best number of clusters predicted)
    # By index
    KL = Column('KL', Integer, nullable=True)
    CH = Column('CH', Integer, nullable=True)
    Hartigan = Column('Hartigan', Integer, nullable=True)
    CCC = Column('CCC', Integer, nullable=True)
    Scott = Column('Scott', Integer, nullable=True)
    Marriot = Column('Marriot', Integer, nullable=True)
    TrCovW = Column('TrCovW', Integer, nullable=True)
    TraceW = Column('TraceW', Integer, nullable=True)
    Friedman = Column('Friedman', Integer, nullable=True)
    Rubin = Column('Rubin', Integer, nullable=True)
    Cindex = Column('Cindex', Integer, nullable=True)
    DB = Column('DB', Integer, nullable=True)
    Silhouette = Column('Silhouette', Integer, nullable=True)
    Duda = Column('Duda', Integer, nullable=True)
    PseudoT2 = Column('PseudoT2', Integer, nullable=True)
    Beale = Column('Beale', Integer, nullable=True)
    Ratkowsky = Column('Ratkowsky', Integer, nullable=True)
    Ball = Column('Ball', Integer, nullable=True)
    PtBiserial = Column('PtBiserial', Integer, nullable=True)
    Frey = Column('Frey', Integer, nullable=True)
    McClain = Column('McClain', Integer, nullable=True)
    Dunn = Column('Dunn', Integer, nullable=True)
    Hubert = Column('Hubert', Integer, nullable=True)
    SDindex = Column('SDindex', Integer, nullable=True)
    Dindex = Column('Dindex', Integer, nullable=True)
    SDbw = Column('SDbw', Integer, nullable=True)
    gap_tib = Column('gap_tib', Integer, nullable=True)
    gap_star = Column('gap_star', Integer, nullable=True)
    gap_max = Column('gap_max', Integer, nullable=True)

    # Relationships
    customer_id = Column(Integer, ForeignKey('customers.id'))
    hyperparameter_id = Column(Integer, ForeignKey('hyperparameter.id'))

    def __repr__(self):
        return "<clustering(customer_id='%s', 'id='%s')>"\
            % (self.customer_id, self.id)

    @staticmethod
    def get_n_len_features(X):
        """Calculate n_len1, n_len2, [...], n_len7 from a given dataset. n_len1 is
        calculated by finding the number of instances that have at least
        1 feature. A dataset with n_len3 value of 53 has 53 instances with
        3 features. n_len represents the number of instances that are
        at least n words long (used for text clustering)

        inputs
        -------
        X : (np.array) input features.
        outputs
        -------
        n_len_features : (dict) with keys n_len1, n_len2, [...] for X"""

        assert X.ndim == 2, "X Must have 2 dimensions. X passed has {}".format(X.ndim)
        assert np.sum(np.isnan(X), axis=(0,1)) == 0, 'X Must not have nan values'
        n_len_features = {'n_len1':None,'n_len2':None,'n_len3':None,
                          'n_len4':None,'n_len5':None,'n_len6':None,
                          'n_len7':None}
        lengths = np.sum(X > 0, axis=1)
        n_len_features['n_points'] = X.shape[0]
        n_len_features['n_len1'] = np.sum(lengths == 1)
        n_len_features['n_len2'] = np.sum(lengths == 2)
        n_len_features['n_len3'] = np.sum(lengths == 3)
        n_len_features['n_len4'] = np.sum(lengths == 4)
        n_len_features['n_len5'] = np.sum(lengths == 5)
        n_len_features['n_len6'] = np.sum(lengths == 6)
        n_len_features['n_len7'] = np.sum(lengths == 7)

        return n_len_features


class ClusteringHyperparameter(Base):
    __tablename__ = 'hyperparameter'

    id = Column(Integer, primary_key=True)
    by_size = Column('by_size', BIT(), nullable=False)
    clusterer = Column('clusterer', NVARCHAR(50), nullable=False)
    distance = Column('distance', NVARCHAR(50), nullable=False)
    reduce = Column('reduce', NVARCHAR(50), nullable=False)
    n_components = Column('n_components', NVARCHAR(50), nullable=False)

    def __repr__(self):
        return "<clusteringhyperparameter(by_size='%s', 'clusterer='%s')>"\
            % (self.by_size, self.clusterer)

class TypesCorrection(Base):
    __tablename__ = 'typescorrection'

    id = Column(Integer, primary_key=True)
    depreciated_type = Column('depreciated_type', NVARCHAR(50), nullable=False, unique=True)
    new_type = Column('new_type', NVARCHAR(50), nullable=True)

    def __repr__(self):
        return "<typescorrection(depreciated_type='%s', 'new_type='%s')>"\
            % (self.depreciated_type, self.new_type)


class Labeling(Base):
    __tablename__ = 'labeling'

    valid_labels = ['ahu','rtu','exhaust_fan','misc','chiller','boiler',
                    'alarm','room','unknown','skip']
    bag_label_constraint = "bag_label in ('ahu','rtu','exhaust_fan','misc','chiller',\
                'boiler', 'alarm','room','unknown','skip')"

    id = Column(Integer, primary_key=True)
    bag_label = Column(NVARCHAR(100), CheckConstraint(bag_label_constraint), nullable=True)

    # Relationships - reverse query Points on lableing.id (each label id is a group)
    points = relationship('Points', back_populates='group', order_by=Points.id)

    def __repr__(self):
        return "<labeling(id={}, label={})>".format(self.id, self.bag_label)



def create_tables(server_name='.\DT_SQLEXPR2008',
                  driver_name='SQL Server Native Client 10.0',
                  database_name='Clustering'):
    """Create tables defined in this module"""
    """
    Creating tables
    The declarative base class uses a metaclass to perform additional things
    Including creating a Table object and constructing a Mapper object
    MetaData → Table
    MetaData has info to emit schema generation commands (CREATE TABLE)

    A Table that represents a table in a database.
    A mapper that maps a Python class to a table in a database.
    A class object that defines how a database record maps to a normal Python object.

    Instead of writing code for a Table, mappper, and class object sqlalchemy's declarative
    allows Table, mapper, and the class to be defined in one class definition
    """

    # Create an engine
    SQLHelper = SQLHandling(server_name=server_name, driver_name=driver_name)
    conn_str = SQLHelper.get_sqlalchemy_connection_str(database_name, driver_name=driver_name)
    engine = create_engine(conn_str)

    # Inspect Table object (part of metadata)
    customer_table = Customers.__table__
    points_table = Points.__table__
    netdev_table = Netdev.__table__

    # Creates tables that we define...
    Base.metadata.create_all(engine)

    return (customer_table, points_table, netdev_table)

"""
# How to remove a table from metadata
table_object = BaseClass.__table__
Base.metadata.remove(tale_object)

# Create a single table
engine = sqlalchemy.create_engine('connection_string')
table_object = BaseClass.__table__
table_object.create(engine)

# Allow altering of metadata
Base."""