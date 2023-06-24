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

def VectorMaker(Vectorizer, X_train, X_test):
    Vectorizer.fit(X_train)
    X_train_Vectorized = Vectorizer.transform(X_train)
    X_test_Vectorized = Vectorizer.transform(X_test)
    return X_train_Vectorized, X_test_Vectorized

# Count Vectors
# CV = CountVectorizer()
# CV.fit(X_train)
# X_train_cv = CV.transform(X_train)
# X_test_cv = CV.transform(X_test)
# CV.get_feature_names_out()[0:10]
# X_train_cv.toarray()
CV = CountVectorizer()
X_train_cv, X_test_cv = VectorMaker(CV, X_train, X_test)

# TF-IDF | Word Level
TFIDF_WORDS = TfidfVectorizer()
X_train_tfidf_Vec_word, X_test_tfidf_Vec_word = VectorMaker(TFIDF_WORDS, X_train, X_test)

# TF-IDF | N-Gram Level
TFIDF_NGRAMS = TfidfVectorizer(ngram_range=(2, 3))
X_train_tfidf_Vec_ngrams, X_test_tfidf_Vec_ngrams = VectorMaker(TFIDF_NGRAMS, X_train, X_test)

# TF-IDF | Character Level
TFIDF_CHARS = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
X_train_tfidf_Vec_chars, X_test_tfidf_Vec_chars = VectorMaker(TFIDF_CHARS, X_train, X_test)

# Makine Ogrenmesi ile Sentiment Siniflandirma

def FitAndCrossVal(Estimator, X_train, X_test, y_train, y_test):
    Model = Estimator.fit(X_train, y_train)
    accuracy = model_selection.cross_val_score(Model, X_test, y_test, cv=10).mean()
    print("Dogruluk orani:", accuracy)

# Lojistik Regresyon
# Count Vector
LR = linear_model.LogisticRegression()
FitAndCrossVal(LR, X_train_cv, X_test_cv, y_train, y_test)
# 0.8398012552301255

# TF-IDF Word Level
LR = linear_model.LogisticRegression()
FitAndCrossVal(LR, X_train_tfidf_Vec_word, X_test_tfidf_Vec_word, y_train, y_test)
# 0.8353556485355649

# TF-IDF N-Grams Level
LR = linear_model.LogisticRegression()
FitAndCrossVal(LR, X_train_tfidf_Vec_ngrams, X_test_tfidf_Vec_ngrams, y_train, y_test)
# 0.7463912133891213

# TF-IDF Character Level
LR = linear_model.LogisticRegression()
FitAndCrossVal(LR, X_train_tfidf_Vec_chars, X_test_tfidf_Vec_chars, y_train, y_test)
# 0.7802301255230126

# Naive Bayes
# Count Vector
NB = naive_bayes.MultinomialNB()
FitAndCrossVal(NB, X_train_cv, X_test_cv, y_train, y_test)
# 0.8332112970711296

# TF-IDF Word Level
NB = naive_bayes.MultinomialNB()
FitAndCrossVal(NB, X_train_tfidf_Vec_word, X_test_tfidf_Vec_word, y_train, y_test)
# 0.835041841004184

# TF-IDF N-Grams Level
NB = naive_bayes.MultinomialNB()
FitAndCrossVal(NB, X_train_tfidf_Vec_ngrams, X_test_tfidf_Vec_ngrams, y_train, y_test)
# 0.7685146443514643

# TF-IDF Word Level
NB = naive_bayes.MultinomialNB()
FitAndCrossVal(NB, X_train_tfidf_Vec_chars, X_test_tfidf_Vec_chars, y_train, y_test)
# 0.7557008368200837

# Random Forest
# Count Vector
RF = ensemble.RandomForestClassifier()
FitAndCrossVal(RF, X_train_cv, X_test_cv, y_train, y_test)
# 0.8217573221757322

# Tahmin
CV = CountVectorizer()
CV.fit(X_train)
x = CV.transform(pd.Series("this film is very nice and good i like it"))
LR = linear_model.LogisticRegression()
LR_model = LR.fit(X_train_cv)
LR_model.predict(x)

x = CV.transform(pd.Series("no not good look at that shit very bad"))
LR_model.predict(x)
