# -*- coding: utf-8 -*-
"""
Created on Sun May 10 10:51:04 2020

SQLAlchemy Table Definitions

@author: z003vrzk
"""

# Python imports

# Third party imports
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.mssql import NVARCHAR, NUMERIC, BIT
from sqlalchemy import create_engine

# Local imports
from DT_sql_tools_v6 import SQLHandling

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

    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False)

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

    customer_id = Column(Integer, ForeignKey('customers.id'))

    def __repr__(self):
        return "<netdev(name='%s', 'id='%s')>"\
            % (self.NETDEVID, self.id)


class Customers(Base):
    __tablename__ = 'customers'

    id = Column(Integer, primary_key=True)
    name = Column(String(150), nullable=False, unique=True)

    points = relationship('Points', order_by=Points.id)
    netdev = relationship('Netdev', order_by=Netdev.id)

    def __repr__(self):
        return"<customer(name='%s')>" % (self.name)






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