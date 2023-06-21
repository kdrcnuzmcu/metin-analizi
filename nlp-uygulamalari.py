from textblob import TextBlob
import nltk

metin = """Machine learning algorithms dominate applied machine learning. Because algorithms are such
a big part of machine learning you must spend time to get familiar with them and really
understand how they work. I wrote this book to help you start this journey."""
metin = " ".join(metin.split("\n"))

# N-Grams
TextBlob(metin).ngrams(1)
TextBlob(metin).ngrams(2)
TextBlob(metin).ngrams(3)

# Part of speech tagging (POS)
nltk.download("averaged_perceptron_tagger")
pos = TextBlob(metin).tags
# dataframe["column"].apply(lambda x: TextBlob(x).tags)
# series.apply(lambda x: TextBlob(x).tags)

# Chunking (shallow parsing)
reg_ex = "NP: {<DT>?<JJ>*<NN>}"
rp = nltk.RegexpParser(reg_ex)
sonuclar = rp.parse(pos)
sonuclar.draw()

# Named Entity Recognition
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
nltk.download("maxent_ne_chunker")
nltk.download("words")

print(ne_chunk(pos_tag(word_tokenize(metin))))

# Matematiksel Islemler ve Basit Ozellik Cikarimi

import pandas as pd
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 150)
pd.set_option("display.expand_frame_repr",  False)
metin2 = """Why linear regression belongs to both statistics and machine learning.
The many names by which linear regression is known.
The representation and learning algorithms used to create a linear regression model.
How to best prepare your data when modeling using linear regression."""

s = pd.Series(metin2.split("\n"))
s.str.len() # Bosluk karakterini de sayar.
df = pd.DataFrame({"Metin": s, "Harfler": s.str.len()})
df["Kelimeler"] = df["Metin"].apply(lambda x: len(str(x).split(" ")))
df["OzelKelime"] = df["Metin"].apply(lambda x: len([x for x in x.split() if x.startswith("statistics")]))
df["Rakamlar"] = df["Metin"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

# Metin Gorsellestirme
import re
import pandas as pd
import nltk
from textblob import Word
from nltk.corpus import stopwords
data = pd.read_csv("train.tsv", sep="\t")
data.head()
#nltk.download("stopwords")
#nltk.download("punkt")
#nltk.download("wordnet")
sw = stopwords.words("english")
data.shape
data.head(100)
samples = data.sample(5000)
samples = samples.sort_values("SentenceId", ascending=True)
samples = samples.reset_index(drop=True)
samples.head()

samples["Phrase"] = samples["Phrase"].apply(lambda x: " ".join(re.sub("[^\w\s]", "", x) for x in x.split()))
samples["Phrase"] = samples["Phrase"].apply(lambda x: " ".join(re.sub("\d", "", x) for x in x.split()))
samples["Phrase"] = samples["Phrase"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
samples["Phrase"] = samples["Phrase"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

tf = samples["Phrase"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["Words", "Counts"]
tf.head()

tf.info()
tf.nunique()
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
onemliler = tf[tf["Counts"] > 50]
onemliler.plot.bar(x="Words", y="Counts");
plt.show()