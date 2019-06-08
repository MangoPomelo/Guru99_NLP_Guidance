from collections import Counter
import nltk
text = " Guru99 is one of the best sites to learn WEB, SAP, Ethical Hacking and much more online."
lower_case = text.lower()
tokens = nltk.word_tokenize(lower_case)
tags = nltk.pos_tag(tokens)
counts = Counter(tag for word, tag in tags)
print(counts)

import nltk
a = "Guru99 is the site where you can find the best tutorials for Software Testing     Tutorial, SAP Course for Beginners. Java Tutorial for Beginners and much more. Please     visit the site guru99.com and much more."
words = nltk.tokenize.word_tokenize(a)
fd = nltk.FreqDist(words)
fd.plot()