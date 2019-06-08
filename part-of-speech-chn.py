import jieba.posseg
from nltk import RegexpParser
text = "学习英语在教室"
tags = [(word, flag) for word, flag in list(jieba.posseg.cut(text))]
patterns = "NP: {(<v><nz>)|(<n>)}"
chunker = RegexpParser(patterns)
output = chunker.parse(tags)
print(output)
output.draw()