import numpy as np
import pandas as pd
import scipy
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys

l=sys.argv


df=pd.read_csv(l[2],sep="\t")

w1=df['word1'].tolist()
w2=df['word2'].tolist()
score=df['SimLex999']


#------------------creating the corpus--------------------------#
from nltk.corpus import reuters
t=reuters.fileids()
from nltk.corpus import stopwords
#nltk.download('stopwords')
rem = set(stopwords.words('english'))
from nltk.corpus import stopwords
#nltk.download('stopwords')
#rem = set(stopwords.words('english'))
train=[]
test=[]
for a in t:
    if a.startswith('training'):
        train.append(a)
    else:
        test.append(a)
corpus=reuters.words(train)
filtered_corpus = []
for w in corpus:
      if not (w.isdigit() or w in rem or len(w)<3):
            filtered_corpus.append(w.lower())
vocab = nltk.FreqDist(filtered_corpus)
vocab_size=len(vocab)
#-------------------corpus created------------------------------#
word_idx = dict()
idx_word=dict()
dict_count = dict()
i=1
for word, count in vocab.items():
    word_idx[word]=i
    idx_word[i]=word
    dict_count[i]=count
    i=i+1


cfiltered=[]
for w in filtered_corpus:
    if(dict_count[word_idx[w]]>10):
        cfiltered.append(w)



w1_=[]
w2_=[]
score_=[]
for i in range(len(w1)):
    if w1[i] in cfiltered and w2[i] in cfiltered:
        w1_.append(word_idx[w1[i]])
        w2_.append(word_idx[w2[i]])
        score_.append(score[i])

file=open(l[1],'rb')
data = pickle.load(file)
embeddings = data[0] 

myscore=[]
for i in range(len(w1_)):
    k=embeddings[w1_[i]]
    vec2=embeddings[w2_[i]]
    s=cosine_similarity(np.reshape(embeddings[w1_[i]],(1,50)),np.reshape(embeddings[w2_[i]],(1,50)))[0][0]
    myscore.append(s)

    
    

from scipy.stats import spearmanr
s= spearmanr(score_,myscore)
print(str(s))
