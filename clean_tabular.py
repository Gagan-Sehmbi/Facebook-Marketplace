# %% 
# IMPORT LIBRARIES
import numpy as np
import pandas as pd

import pymysql
from sqlalchemy import create_engine

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

df1 = pd.read_sql_table( 'products', engine)
df2 = pd.read_sql_table( 'products_2', engine)
df3 = pd.read_sql_table( 'images', engine)

print('Done')

# %%
# MERGE TABULAR DATA
df = pd.concat([df1,df2])
df = df.rename(columns={'id': 'product_id'})
df = pd.merge(df, df3, on='product_id', how='outer')
df = df.replace('N/A', np.nan)
df = df.reset_index()
df = df.drop_duplicates(subset=['id'])

print('Done')
# %% 
# IMPORT LINKS DATA
links = pd.read_csv('Links.csv')
links.drop_duplicates(inplace=True)

print('Done')

# %%
# CREATE LINK ID COLUMN AND MERGE DATASETS
links['link_id'] = links['Link'].apply(lambda x: x.split('/')[-1])
df['link_id'] = df['page_id'].apply(str)

df = pd.merge(df, links, on='link_id', how='inner')

print('Done')

# %%
# EXPLORE TABULAR DATA
print(df.info())
msno.matrix(df)

print('Done')

# %% 
# FILL MISSING LOCATIONS USING PRODUCT NAME
df['product_name'].values

df['name_len'] = df['product_name'].apply(lambda x: len(x.split('|')))
df['name_len'].value_counts()

df.loc[(df['name_len'] == 2) & (df['location'].isna()), 'location'] = df.loc[(df['name_len'] == 2) & (df['location'].isna()), 'product_name'].apply(lambda x: x.split('|')[0].split(' for Sale in ')[-1])

df['location'] = df['location'].apply(lambda x: x.split(' ')[-1].strip())

print('Done')

# %%
#  FILL PRODUCT DESCRIPTIONs USING LINK
df['Description'] = df['Link'].apply(lambda x: ' '.join(x.split('/')[-2].split('-')))

print('Done')

# %%
# CONVERT PRICE TO FLOAT

def c2f(x):
    if isinstance(x, float):
        return x
    elif x == 'N/A':
        return np.nan
    else:
        return float(x.replace(',', '')[1:])

df['price'] = df['price'].apply(lambda x: c2f(x))

print('Done')
# %%
# DROP COLUMNS
df.drop(columns=['product_name', 'category', 'product_description', 'page_id', 'create_time_x', 'create_time_y', 'bucket_link', 'name_len', 'link_id', 'Link'], inplace=True)

print('Done')

# %%
# DROP NULL VALUES AND RESET INDEX

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print('Done')

# %%
# DELETE EXTRA IMAGES
import glob
import os

list_img = glob.glob('images/*.jpg')
list_img = [x[7:-4] for x in list_img]
image_set = set(list_img)

image_set_2 = set(df['id'].values.tolist())

images = image_set.symmetric_difference(image_set_2)

for i in images:
    os.remove(f'images/{i}.jpg')

print('Done')

# %%
# SAVE DF AS CSV

df.to_csv('tabular_data.csv')

print('Done')

# %%
