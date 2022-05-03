# %%
# IMPORT LIBRARIES

import pandas as pd
import numpy as np

from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix

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
y = df['Category'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 13)

# %%
# SET UP MODEL PIPELINE

pipe_mnnb = Pipeline(steps = [('tf', TfidfVectorizer()), ('mnnb', MultinomialNB())])

pgrid_mnnb = {
    'tf__max_features' : [1000, 2000, 3000],
    'tf__stop_words' : ['english', None],
    'tf__ngram_range' : [(1,1),(1,2)],
    'tf__use_idf' : [True, False],
    'mnnb__alpha' : [0.1, 0.5, 1]
    }

# %%
# FIT MODEL

gs_mnnb = GridSearchCV(pipe_mnnb,pgrid_mnnb,cv=5,n_jobs=-1)
gs_mnnb.fit(X_train, y_train)

# %%
# VALIDATE MODEL

print(gs_mnnb.score(X_train, y_train))
print(gs_mnnb.score(X_test, y_test))

# %%
# VISUALISE RESULTS

preds_mnnb = gs_mnnb.predict(X)
df['preds'] = preds_mnnb

conf_mnnb = confusion_matrix(y, preds_mnnb)

# %%
