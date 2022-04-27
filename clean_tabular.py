# %% 
# IMPORT LIBRARIES
from inspect import Attribute
from os import link
from unicodedata import category
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

print('Done')

# %%
# READ TABULAR DATA
df = pd.read_csv(
    'tabular_data.csv',
    lineterminator='\n',
    index_col=0)

print('Done')
# %%
# EXPLORE TABULAR DATA
print(df.info())
msno.matrix(df)

print('Done')

# %% 
# FILL MISSING PRODUCT DESCRIPTION AND LOCATION VALUES USING PRODUCT NAME
df['product_name'].values

df['name_len'] = df['product_name'].apply(lambda x: len(x.split('|')))
df['name_len'].value_counts()

df.loc[(df['name_len'] == 2) & (df['location'].isna()), 'location'] = df.loc[(df['name_len'] == 2) & (df['location'].isna()), 'product_name'].apply(lambda x: x.split('|')[0].split(' for Sale in ')[-1])
df.loc[(df['name_len'] == 2) & (df['product_description'].isna()), 'product_description'] = df.loc[(df['name_len'] == 2) & (df['product_description'].isna()), 'product_name'].apply(lambda x: x.split('|')[0].split(' for Sale in ')[0])

print('Done')

# %% 
# MODIFY CATEGORY TO ONLY INCLUDE THE HIGHEST LEVEL
def cat(x):
    try:
        output = x.split('/')[0]
        return output
    except AttributeError:
        return x

df['category'] = df['category'].apply(lambda x: cat(x))

print('Done')
# %%
