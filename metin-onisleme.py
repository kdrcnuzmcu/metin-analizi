import pandas as pd
import re
import nltk

# Bir kitabın Icindekiler bolumunu kullanacagiz.
metin = """
CHAPTER 1
INTRODUCTION TO MACHINE LEARNING 11
Theory
What is machine learning?1
Why machine learning?
When should you use machine learning?
Types of Systems of Machine Learning
Supervised and unsupervised learning22
Supervised Learning
The most important supervised algorithms__
Unsupervised Learning33
The most important unsupervised algorithms
Reinforcement Learning
Batch 4 Learning
Online Learning
Instance based learning
Model-based learning
Bad and Insufficient Quantity of Training Data
Poor-Quality Data
Irrelevant Features 3
Feature Engineering
Testing
Overfitting the Data
Solutions **
Underfitting the Data --
Solutions
EXERCISES--
SUMMARY
REFERENCES
"""

# Metin birden cok satirdan olustugu icin "\n" ifadelerini cikartiyoruz.
metin = metin.split("\n")

# Onisleme boyunca v degiskenini kullanip, meydana gelen degisimi gozlemlemek icin v2 degiskenini olusturuyoruz.
v = pd.Series(metin)
v2 = pd.Series(metin)

# Pandas serisi 0 indeksi ile basliyor. 0 indeksinde ise bir deger bulunmadigi icin bu satiri siliyoruz.
# Yeni seri 1 den basliyor.
v = v[1:len(v)]

# Serinin satirlarindaki butun alfabetik karakterleri kucultuyoruz.
v = v.apply(lambda x: " ".join(x.lower() for x in x.split()))

# Butun noktalama isaretlerini cikartiyoruz.
v = v.apply(lambda x: " ".join(re.sub("[^\w\s]", "", x) for x in x.split()))

# Butun numerik karakterleri cikartiyoruz.
v = v.apply(lambda x: " ".join(re.sub("\d", "", x) for x in x.split()))

# Metinsel ifadeler icinde gereksiz/duraklama kelimelerini nltk kutuphanesi ile cikartiyoruz.
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
sw = stopwords.words("english")
v = v.apply(lambda x: " ".join(x for x in x.split() if x not in sw))

# Serinin butun satirlarinda bulunan kelimelerin frekanslarini bulup en az tekrar eden, dolayisiyla en az anlam tasiyan kelimeleri cikartacagiz.
# value_counts fonksiyonu essiz degerleri saymaya yarar. Satirlardaki kelimelerin her biri bir eleman olacak sekilde ayristirip value_counts fonksiyonu ile frekanslarini saydirabiliriz.
# Series yapisi, value_counts fonksiyonuna sahip olmadigi icin yapiyi DataFrame e cevirecegiz.
silinecekler = pd.DataFrame(" ".join(v).split(), columns=["Konular"]).value_counts()[-5:]
v = v.apply(lambda x: " ".join(x for x in x.split() if x not in silinecekler))

# Buradaki islem ile satirlardaki kelimeleri bir liste elemani haline getirir.
from textblob import TextBlob
v.apply(lambda x: TextBlob(x).words)

# Kelime halindeki ifadeleri cumlelerden cikarttik. Bu kelimeler anlam butunlugunu saglamak icin bazi ekler almis olabilir. Bu ekleri temizlemek icin iki farkli fonksiyon mevcuttur.
# Bu fonksiyon gorece "asiri" ek temizleme islemi yapmaktadir.
from nltk.stem import PorterStemmer
st = PorterStemmer()
v.apply(lambda x: " ".join([st.stem(i) for i in x.split()]))

# Bu fonksiyon gorece "tutarli" ek temizleme islemi yapmaktadir.
nltk.download("wordnet")
from textblob import Word
v = v.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Asagidaki DataFrame yapisi ile islemlerimiz sonucunda olusan farkliligi gorebiliyoruz.
pd.DataFrame({"Once": v2, "Sonra": v})