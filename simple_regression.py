# %%
# IMPORT LIBRARIES

import pandas as pd
import numpy as np

from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

# %%
# IMPORT DATA

df = pd.read_csv('tabular_data.csv', index_col=0)
df['Text'] = df['location'] + ' ' + df['Description']

# %%
# TEXT PREPROCESSING

# Tokenize
def tokenize(x):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(x)
    
df['Tokens'] = df['Text'].map(tokenize)

def stemmer(x):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in x])
    
def lemmatize(x):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in x])

df['Lemma'] = df['Tokens'].map(lemmatize)
df['Stems'] = df['Tokens'].map(stemmer)

# %%
# TRAIN-TEST SPLIT

X = df['Lemma'].values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 13)

# %%
# SET UP MODEL PIPELINE
model = GradientBoostingRegressor(random_state=0)
pipe = Pipeline(steps = [('tf', TfidfVectorizer()), ('model', model)])

parameters = {
    'tf__max_features' : [1000, 2000, 3000],
    'tf__stop_words' : ['english', None],
    'tf__ngram_range' : [(1,1),(1,2)],
    'tf__use_idf' : [True, False],
    'model__n_estimators':[100,500], 
    'model__learning_rate': [0.1,0.05,0.02],
    'model__max_depth':[4], 
    'model__min_samples_leaf':[3], 
    'model__max_features':[1.0]
    } 

# %%
# FIT MODEL

gs = GridSearchCV(pipe, parameters, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)

# %%
# VALIDATE MODEL

print(gs.score(X_train, y_train))
print(gs.score(X_test, y_test))

# %%
# VISUALISE RESULTS

preds = gs.predict(X)
df['preds'] = preds

mse = mean_squared_error(y, preds)
rmse = mean_squared_error(y, preds, squared=False)
mae = mean_absolute_error(y, preds)


print(f'Mean square error: {mse}')
print(f'Root mean square error: {rmse}')
print(f'Mean absolute error: {mae}')
# %%
df[['price', 'preds']]


# %%
