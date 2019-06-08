import json
import pandas as pd
from gensim.models import Word2Vec

json_file = 'intents.json'
with open('intents.json','r') as f:
	data = json.load(f)

df = pd.DataFrame(data)
print(df)
df['patterns'] = df['patterns'].apply(', '.join) 
print(df)

from nltk.corpus import stopwords
from textblob import Word
import string
stop = stopwords.words('english')
df['patterns'] = df['patterns'].apply(lambda x:' '.join(x.lower() for x in x.split())) # lower every word
df['patterns']= df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if x not in string.punctuation)) # exclude punctuation
df['patterns']= df['patterns'].str.replace('[^\w\s]','') # replace non-word
df['patterns']= df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit())) # exclude numbers
df['patterns'] = df['patterns'].apply(lambda x:' '.join(x for x in x.split() if not x in stop)) # exclude stopwords
df['patterns'] = df['patterns'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) # textblob is another library like nltk

bigger_list=[]
for i in df['patterns']:
    li = list(i.split(" "))
    bigger_list.append(li)	
model = Word2Vec(bigger_list,min_count=1,size=300,workers=4)
print("Data format for the overall list:",bigger_list)
#custom data is fed to machine for further processing
model = Word2Vec(bigger_list, min_count=1,size=300,workers=4)

model.save("model.bin")
del model
model = Word2Vec.load('model.bin')
print(model)

# Most Similar words checking
similar_words = model.most_similar('thanks')	
print(similar_words)

# Does not match word from words supplied
dissimlar_words = model.doesnt_match('See you later, thanks for visiting'.split())
print(dissimlar_words)

# Finding the similarity between two words
similarity_two_words = model.similarity('please','see')
print("Please provide the similarity between these two words:")
print(similarity_two_words)

similar = model.similar_by_word('kind')
print(similar)