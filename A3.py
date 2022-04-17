import pandas as pd
import json
import gensim
from gensim.models import Word2Vec, TfidfModel
import nltk
from nltk.corpus import brown

from collections import defaultdict
import pytrec_eval

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords

stop_words = pd.read_csv('C:\\Users\\jamal\\Documents\\MS\\Comp8730\\Assignment\\3MT\\Stopword_Master.csv',encoding = 'latin1')
stopWordSet = [x for x in stop_words['stop']]
stopWordSet = list(set(stopWordSet))
len(stopWordSet)


 message = nltk.Text(brown.words(categories="news"))
 print(message)
 print()
 print("Concordance:")
 message.concordance("news")
 print()
 print("Distributionally similar words:")
 message.similar("news")
 print()
 print("Collocations:")
 message.collocations()
 print()
  print("Automatically generated text:")
  message.generate()
  print()
 print("Dispersion plot:")
 message.dispersion_plot(["news", "report", "said", "announced"])
 print()
 print("Vocab plot:")
 message.plot(50)
 print()
 print("Indexing:")
 print("text[3]:", message[3])
 print("text[3:5]:", message[3:5])
 print("text.vocab()['news']:", message.vocab()["news"])

for sent in brown.sents(categories='news'):
     munged_sentence = ' '.join(sent).replace('``', '"').replace("''", '"').replace('`', "'")
     sentence = TreebankWordDetokenizer().detokenize(
         munged_sentence.split(), ).lower()
     splits = sentence.split()
     for split in splits:
         listofwords.append(split)
     listofsents.append(sentence)
 print('***********TF-IDF based preprocessing is now completed***********')

 def word2vectrainer(category):
    message = brown.sents(categories=category)
    w2vt = Word2Vec(message, window=3, min_count=1, size=110, iter=110)
    return w2vt
