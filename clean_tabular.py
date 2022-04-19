# %% 
# IMPORT LIBRARIES
from os import link
import numpy as np
import pandas as pd

import pymysql
from sqlalchemy import create_engine, inspect

import missingno as msno

print('Done')

# %% 
# IMPORT TABULAR DATA
DATABASE_TYPE= 'postgresql'
DBAPI= 'psycopg2' 
ENDPOINT= 'products.c8k7he1p0ynz.us-east-1.rds.amazonaws.com'
DBUSER= 'postgres'
DBPASSWORD= 'aicore2022!' 
PORT= '5432' 
DATABASE= 'postgres' 
AWS_CONFIG_FILE= '/app/.aws/config AWS_SHARED_CREDENTIALS_FILE=/app/.aws/credentials'

engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{DBUSER}:" f"{DBPASSWORD}@{ENDPOINT}:" f"{PORT}/{DATABASE}")

engine.connect()

df = pd.read_sql_table( 'products', engine, columns=["id", "product_name", "category", "product_description", "price", "location", "page_id", "create_time"]) 
#df.replace('N/A', np.nan, inplace=True)
df.to_csv('tabular_data.csv')

# %%
df = pd.read_csv(
    'tabular_data.csv',
    lineterminator='\n',
    index_col=0)

print('Done')
df
# %%
# EXPLORE TABULAR DATA
print(df.info())
msno.matrix(df)

print('Done')

