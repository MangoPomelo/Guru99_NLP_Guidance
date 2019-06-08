from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
data_corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?'
]
matrix = vectorizer.fit_transform(data_corpus)
print(matrix.toarray())
print(vectorizer.get_feature_names())

import nltk
import gensim
from nltk.corpus import abc

model = gensim.models.Word2Vec(abc.sents())
vocab = list(model.wv.vocab)
data = model.most_similar('science')
print(data)