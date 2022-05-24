# %%
# IMPORT LIBRARIES

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# %%
# IMPORT TABULAR DATA

df = pd.read_csv('tabular_data.csv', index_col=0).drop(columns=['index'])
df
# %%
# SPLIT TABULAR DATA

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=42)


df_train.to_csv('tabular_data_train.csv')
df_val.to_csv('tabular_data_val.csv')
df_test.to_csv('tabular_data_test.csv')

train_images = df_train.loc[:, 'id'].tolist()
val_images = df_val.loc[:, 'id'].tolist()
test_images = df_test.loc[:, 'id'].tolist()


# %%
# SPLIT IMAGE DATA

for image in train_images:
    os.replace(f'images/{image}.jpg', f'images/train/{image}.jpg')

for image in val_images:
    os.replace(f'images/{image}.jpg', f'images/validation/{image}.jpg')

for image in test_images:
    os.replace(f'images/{image}.jpg', f'images/test/{image}.jpg')
# %%
