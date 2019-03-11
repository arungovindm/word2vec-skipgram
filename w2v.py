# -*- coding: utf-8 -*-
"""Copy of word2vec-skipgram.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fNlyPh8nUAdGA3OpajzhwFCB1DNHpGQG
"""


import numpy as np
import sys
import nltk
import tensorflow as tf
import keras
import pickle

args=sys.argv

def unigram_sampling(dct):
    rand_val = np.random.random()
    total = 0
    for k, v in dct.items():
        total += v
        if rand_val <= total:
            return k
    assert False, 'unreachable'
    
def create_train_instance(batch_id,batch_size,neg):  #target word, context word and neg negative context samples
  train_instance=[]
  for j in range(batch_size):
    train_instance.append(target[(batch_id-1)*batch_size+j])
    train_instance.append(context[(batch_id-1)*batch_size+j])
    i=0
    for i in range(neg):
      train_instance.append(unigram_sampling(dict_prob))
  return np.reshape(train_instance,[batch_size*(neg+2),1])

def get_idx(w):
  w_idx=[]
  c_idx=[]
  for i in range(batch_size*(neg+2)):
    if i % (neg+2) ==0:
      for j in range(neg+1):
        w_idx.append(w[i])
    else:
      c_idx.append(w[i])
  return w_idx,c_idx

#------------------parameters-----------------------------------#
window_size = 5                              #context window size
neg = int(args[2])                               #number of negative samples
lr = 0.1                                           #learning rate
d = 50                                            #dimensionality
power = 0.75            #negative sampling distribution parameter
batch_size = int(args[1])                                       #batch size



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


Z=0
for a,b in vocab.items():
  Z=Z+b**power

word_idx = dict()
idx_word=dict()
dict_count = dict()
i=1
for word, count in vocab.items():
  word_idx[word]=i
  idx_word[i]=word
  dict_count[i]=count
  i=i+1
  
#---------------create corpus of indices------------#
labelled_corpus=[]
for word in filtered_corpus:
  labelled_corpus.append(word_idx[word])
del corpus, filtered_corpus
#---------------------------------------------------#

target=[]
context=[]
for i in range(len(labelled_corpus)):
  for j in range(window_size):
    if i+j+1 < len(labelled_corpus):
      context.append(labelled_corpus[i+j+1])
      target.append(labelled_corpus[i])
    if i-j-1 >=0:
      context.append(labelled_corpus[i-j-1])
      target.append(labelled_corpus[i])
dict_prob=dict()
for idx,count in dict_count.items():
  dict_prob[idx]=(dict_count[idx]**power)/Z


# building the neural net with tensorflow

#model parameteres
b=tf.Variable(tf.zeros(vocab_size,1))
W=tf.Variable(tf.random_uniform((vocab_size+1,d),-1,1),dtype=tf.float32)
C=tf.Variable(tf.random_uniform((vocab_size+1,d),-1,1),dtype=tf.float32)


temp=-1*np.eye(neg+1)
temp[0,0]=1

#input
w=tf.placeholder(tf.int32, (batch_size*(neg+2),1))
w_idx,c_idx = get_idx(w)
#hidden layer
H1=tf.reshape(tf.nn.embedding_lookup(C,c_idx),[batch_size,neg+1,d])
v_w=tf.reshape(tf.nn.embedding_lookup(W,w_idx), [batch_size,neg+1,d])
dot_prod=tf.reduce_sum(H1*v_w,axis=2)
mult=tf.constant(temp,dtype=tf.float32)
loss=-1*tf.reduce_sum(tf.log_sigmoid(tf.matmul(dot_prod,mult)))
train_step=tf.train.GradientDescentOptimizer(lr).minimize(loss)

print(d,"-d vectors","\nnegative samples:",neg,"\nbatch size:",batch_size,"\noptimser loop: ",5,"\nlearning rate:",lr)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
n_batches=int(len(target)/batch_size)
epo=3
for ep in range(epo):
  for j in range(n_batches):
    #init=tf.global_variables_initializer()
    #sess.run(init)
    w_batch=create_train_instance(j,batch_size,neg)
    for i in range(10):
      loss_value, _ =sess.run([loss,train_step],feed_dict={w:w_batch})
      print("batch:",j+1,'/',n_batches,"loss @ ",i,":",loss_value)
  file=open('f_'+str(ep)+'_'+str(batch_size)+'_'+str(window_size)+'.dat','wb')
  [W_,C_]=sess.run([W,C],feed_dict={w:w_batch})
  pickle.dump([W_,C_],file)
