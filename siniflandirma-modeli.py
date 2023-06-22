# Sentiment Analizi ve Siniflandirma Modelleri
from textblob import TextBlob
from textblob import Word

from nltk.corpus import stopwords

from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import decomposition
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import xgboost as xgb
import numpy as np
import textblob
import string
import re

from keras.preprocessing import text
from keras.preprocessing import sequence
from keras import layers
from keras import models
from keras import optimizers

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 150)
pd.set_option("display.expand_frame_repr",  False)

data = pd.read_csv("train.tsv", sep="\t")
data.head()

data["Sentiment"].replace(0, value="Negative", inplace=True)
data["Sentiment"].replace(1, value="Negative", inplace=True)
# data["Sentiment"].replace(2, value="Negative")
data["Sentiment"].replace(3, value="Positive", inplace=True)
data["Sentiment"].replace(4, value="Positive", inplace=True)

data = data[data["Sentiment"] != 2]
data = data.reset_index(drop=True)
data["Sentiment"].value_counts()

df = data[["Phrase", "Sentiment"]]
df.head()

# Kucuk/Buyuk Harf Donusumu
df["Phrase"] = df.loc[:, "Phrase"].str.lower()
df["Phrase"].str.isupper().sum()
# Noktalama Isaretleri
df["Phrase"] = df["Phrase"].apply(lambda x: " ".join(re.sub("[^\w\s]", "", x) for x in x.split()))
# Rakam/Sayilar
df["Phrase"] = df["Phrase"].apply(lambda x: " ".join(re.sub("\d", "", x) for x in x.split()))
# Stopwords
sw = stopwords.words("english")
df["Phrase"] = df["Phrase"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
# Seyrek Kelimeler
rares = pd.Series(" ".join(df["Phrase"]).split()).value_counts()[-1000:]
df["Phrase"] = df["Phrase"].apply(lambda x: " ".join(x for x in x.split() if x not in rares))
# Lemmatization
df["Phrase"] = df["Phrase"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df.head()

# Degisken Muhendisligi
"""
- Count Vectors
- TF-IDF Vectors (words, characters, n-grams)
- Word Embeddings
"""

# Train-Test-Split
X_train, X_test, y_train, y_test = model_selection.train_test_split(df["Phrase"],
                                                                    df["Sentiment"],
                                                                    random_state=42)

LE = preprocessing.LabelEncoder()
y_train = LE.fit_transform(y_train)
y_test = LE.fit_transform(y_test)

# Count Vectors
CV = CountVectorizer()
CV.fit(X_train)
X_train_cv = CV.transform(X_train)
X_test_cv = CV.transform(X_test)

CV.get_feature_names_out()[0:10]
X_train_cv.toarray()

# TF-IDF
# word level
tfidf_Vec_word = TfidfVectorizer()
tfidf_Vec_word.fit(X_train)
X_train_tfidf_Vec_word = tfidf_Vec_word.transform(X_train)
X_test_tfidf_Vec_word = tfidf_Vec_word.transform(X_test)

tfidf_Vec_word.get_feature_names_out()[0:5]
X_train_tfidf_Vec_word.toarray()

# n-gram level
tfidf_Vec_ngrams = TfidfVectorizer(ngram_range=(2, 3))
tfidf_Vec_ngrams.fit(X_train)
X_train_tfidf_Vec_ngrams = tfidf_Vec_ngrams.transform(X_train)
X_test_tfidf_Vec_ngrams = tfidf_Vec_ngrams.transform(X_test)

tfidf_Vec_ngrams.get_feature_names_out()[0:10]

# character level
tfidf_Vec_chars = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
tfidf_Vec_chars.fit(X_train)
X_train_tfidf_Vec_chars = tfidf_Vec_chars.transform(X_train)
X_test_tfidf_Vec_chars = tfidf_Vec_chars.transform(X_test)

# Makine Ogrenmesi ile Sentiment Siniflandirma
# Lojistik Regresyon
# Count Vector
LR = linear_model.LogisticRegression()
LR_model = LR.fit(X_train_cv, y_train)
accuracy = model_selection.cross_val_score(LR_model,
                                           X_test_cv,
                                           y_test,
                                           cv=10).mean()
print("Count Vector dogruluk orani:", accuracy)
# 0.8398012552301255

# TF-IDF Word Level
LR = linear_model.LogisticRegression()
LR_model = LR.fit(X_train_tfidf_Vec_word, y_train)
accuracy = model_selection.cross_val_score(LR_model,
                                           X_test_tfidf_Vec_word,
                                           y_test,
                                           cv=10).mean()
print("TF-IDF Word Level dogruluk orani:", accuracy)
# 0.8353556485355649

# TF-IDF N-Grams Level
LR = linear_model.LogisticRegression()
LR_model = LR.fit(X_train_tfidf_Vec_ngrams, y_train)
accuracy = model_selection.cross_val_score(LR_model,
                                           X_test_tfidf_Vec_ngrams,
                                           y_test,
                                           cv=10).mean()
print("TF-IDF N-Grams Level dogruluk orani:", accuracy)
# 0.7463912133891213

# TF-IDF Character Level
LR = linear_model.LogisticRegression()
LR_model = LR.fit(X_train_tfidf_Vec_chars, y_train)
accuracy = model_selection.cross_val_score(LR_model,
                                           X_test_tfidf_Vec_chars,
                                           y_test,
                                           cv=10).mean()
print("TF-IDF Character Level dogruluk orani:", accuracy)
# 0.7802301255230126

# Naive Bayes
# Count Vector
NB = naive_bayes.MultinomialNB()
NB_model = NB.fit(X_train_cv, y_train)
accuracy = model_selection.cross_val_score(NB_model,
                                           X_test_cv,
                                           y_test,
                                           cv=10).mean()
print("Count Vector dogruluk orani:", accuracy)
# 0.8332112970711296

# TF-IDF Word Level
NB = naive_bayes.MultinomialNB()
NB_model = NB.fit(X_train_tfidf_Vec_word, y_train)
accuracy = model_selection.cross_val_score(NB_model,
                                           X_test_tfidf_Vec_word,
                                           y_test,
                                           cv=10).mean()
print("TF-IDF Word Level dogruluk orani:", accuracy)
# 0.835041841004184

# TF-IDF N-Grams Level
NB = naive_bayes.MultinomialNB()
NB_model = NB.fit(X_train_tfidf_Vec_ngrams, y_train)
accuracy = model_selection.cross_val_score(NB_model,
                                           X_test_tfidf_Vec_ngrams,
                                           y_test,
                                           cv=10).mean()
print("TF-IDF N-Grams Level dogruluk orani:", accuracy)
# 0.7685146443514643

# TF-IDF Word Level
NB = naive_bayes.MultinomialNB()
NB_model = NB.fit(X_train_tfidf_Vec_chars, y_train)
accuracy = model_selection.cross_val_score(NB_model,
                                           X_test_tfidf_Vec_chars,
                                           y_test,
                                           cv=10).mean()
print("TF-IDF Word Level dogruluk orani:", accuracy)
# 0.7557008368200837

# Random Forest
# Count Vector
RF = ensemble.RandomForestClassifier()
RF_model = RF.fit(X_train_cv, y_train)
accuracy = model_selection.cross_val_score(RF_model,
                                           X_test_cv,
                                           y_test,
                                           cv=10).mean()
print("Count Vector dogruluk orani:", accuracy)
# 0.8217573221757322

# Tahmin
CV = CountVectorizer()
CV.fit(X_train)
x = CV.transform(pd.Series("this film is very nice and good i like it"))
LR_model.predict(x)

x = CV.transform(pd.Series("no not good look at that shit very bad"))
LR_model.predict(x)
