import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
sw = stopwords.words("english")

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

metin.split()
yeni_metin = metin.split("\n")
v = pd.Series(yeni_metin)
v = v[1:len(v)]
v = v.apply(lambda x: " ".join(x.lower() for x in x.split()))
v = v.apply(lambda x: " ".join(re.sub("[^\w\s]", "", x) for x in x.split()))
v = v.apply(lambda x: " ".join(re.sub("\d", "", x) for x in x.split()))
df = pd.DataFrame(v, columns=["Konular"])
df = df["Konular"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))












