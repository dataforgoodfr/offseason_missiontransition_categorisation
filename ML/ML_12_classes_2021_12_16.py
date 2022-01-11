# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 17:58:58 2021

@author: emmal
"""
#%% import libraries
import pandas as pd
import numpy as np
from sklearn import model_selection, feature_selection, feature_extraction
import random
import re
import nltk
from nltk.stem.snowball import FrenchStemmer
nltk.download('wordnet')
nltk.download('stopwords')
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier 
from sklearn.metrics import accuracy_score, f1_score



#%% import data
aids_mt = pd.read_csv('C:/Users/emmal/Documents/Data for good/mission_transition_categorisation/data drive/aids_mt.csv',
                      #error_bad_lines=False,
                      sep = '	')

aids_mt_preprocessed = pd.read_csv('C:/Users/emmal/Documents/Data for good/mission_transition_categorisation/data drive/aids_mt_preprocessed.csv',
                      #error_bad_lines=False,
                      sep = '	')

aids_mt_preprocessed_subclasses_50 = pd.read_csv('C:/Users/emmal/Documents/Data for good/mission_transition_categorisation/data drive/aids_mt_preprocessed_subclasses_50.csv',
                      #error_bad_lines=False,
                      sep = '	')

bases_aides = pd.read_csv('C:/Users/emmal/Documents/Data for good/mission_transition_categorisation/data drive/bases_aides.csv',
                      #error_bad_lines=False,
                      sep = '	')

data = pd.read_csv('C:/Users/emmal/Documents/Data for good/mission_transition_categorisation/data drive/data.csv',
                      #error_bad_lines=False,
                      sep = ',')

#%% fct preprocess

#french stopwords
lst_stopwords = nltk.corpus.stopwords.words("french")
lst_stopwords

#fonction preprocess
def preprocess_text(text, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    
    ## Removing Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
        
    ## Stemming
    ps = FrenchStemmer()
    lst_text = [ps.stem(word) for word in lst_text]
                  
    ## back to string from list
    text = " ".join(lst_text)
    return text

#%% mise en forme du df
df = aids_mt_preprocessed
df["text_clean"] = df["text"].apply(lambda x: preprocess_text(x, lst_stopwords = lst_stopwords))

#%% split 
random.seed(123)
train, test = model_selection.train_test_split(df, test_size = 0.1)
y_train = train['list']
y_test = test['list']

#%% TF-IDF
vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))

vectorizer.fit(train['text_clean'])
X_train = vectorizer.transform(train['text_clean'])
dic_vocabulary = vectorizer.vocabulary_

#%% Select features 
X_names = vectorizer.get_feature_names()
p_value_limit = 0.95
features = pd.DataFrame()

for cat in np.unique(y_train):
    chi2, p = feature_selection.chi2(X_train, y_train==cat)
    features = features.append(pd.DataFrame(
                   {"feature":X_names, "score":1-p, "y_train":cat}))
    features = features.sort_values(["y_train","score"], 
                    ascending=[True,False])
    features = features[features["score"]>p_value_limit]

X_names = features["feature"].unique().tolist()

#%% New X_train and X_test 
X_train_old = X_train

vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
vectorizer.fit(train['text_clean'])

X_train = vectorizer.transform(train['text_clean'])
X_test = vectorizer.transform(test['text_clean'])
dic_vocabulary = vectorizer.vocabulary_

#%% Modelisation

models = {'model_xgb' : XGBClassifier(),
          'model_lgbm' : LGBMClassifier(),
          'model_gb' : GradientBoostingClassifier(),
          'model_rf' : RandomForestClassifier(),
          'model_svm' : SVC()}

pred = {}
for i in models:
  models[i] = models[i].fit(X_train, y_train)
  pred[i+'_train'] = models[i].predict(X_train)
  pred[i+'_test'] = models[i].predict(X_test)

#%% Metrics 
metrics = list()

for i in models :
  #accuracy
  acc_train = accuracy_score(y_train, pred[i+'_train'])
  acc_test = accuracy_score(y_test, pred[i+'_test'])

  #f1 score
  f1_train = f1_score(y_train, 
                      pred[i+'_train'],
                      average='micro')
  
  f1_weighted_train = f1_score(y_train, 
                      pred[i+'_train'],
                      average='weighted')
  
  f1_test = f1_score(y_test, 
                     pred[i+'_test'],
                     average='micro')
  
  f1_weighted_test = f1_score(y_test, 
                     pred[i+'_test'],
                     average='weighted')

  metrics.append([i, acc_train, acc_test, f1_train, f1_test, f1_weighted_train, f1_weighted_test])
  
metrics_clean = pd.DataFrame(metrics)
metrics_clean.columns = ['Models','Accuracy train', 'Accuracy test', 'Score F1 train', 'Score F1 test', 'Score F1 weighted train', 'Score F1 weighted test']
metrics_clean.index = metrics_clean['Models']
del metrics_clean['Models']
metrics_clean
