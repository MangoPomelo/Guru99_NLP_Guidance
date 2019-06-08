from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
filteredText = tokenizer.tokenize("There is absolutely no cat on the mat with a hat!")
print(filteredText)

from nltk.tokenize import word_tokenize
text = "God is Great!! I won a lottery."
print(word_tokenize(text))

from nltk.tokenize import sent_tokenize
text = "God is Great!! I won a lottery."
print(sent_tokenize(text))