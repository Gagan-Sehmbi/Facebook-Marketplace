# %%
# IMPORT LIBRARIES

from random import shuffle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


# %%
# IMPORT TABULAR DATA

df = pd.read_csv('tabular_data.csv', index_col=0).drop(columns=['index'])

# %%
# CREATE DIRECTORIES

labels = df.loc[:, 'Category'].unique()

for t in ['train', 'test']:
    parent_dir = '/home/gagansehmbi/Documents/AiCore/Facebook-Marketplace/images'
    os.mkdir(os.path.join(parent_dir, t))
    for label in labels:
        os.mkdir(os.path.join(parent_dir, t, label))


# %%
# SPLIT TABULAR DATA

df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

df_train.to_csv('tabular_data_train.csv')
df_test.to_csv('tabular_data_test.csv')

# %%
# SPLIT IMAGE DATA

for idx in df_train.index:
    id = df_train.loc[idx, 'id']
    label = df_train.loc[idx, 'Category']
    os.replace(f'images/{id}.jpg', f'images/train/{label}/{id}.jpg')

for idx in df_test.index:
    id = df_test.loc[idx, 'id']
    label = df_test.loc[idx, 'Category']
    os.replace(f'images/{id}.jpg', f'images/test/{label}/{id}.jpg')
# %%
