# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:25:10 2019

@author: nEW u
"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def sentiment_score(l):
    p=0
    n=0
    #print(l)
    for each in l:
       # print(each)
        if each in positive_words:
            
            p=p+1
            #print(p)
        elif each in negative_words:
            n=n+1
            #print(n)
       
    if((p-n)>=1):
        return(1)
    elif((p-n)<=-1):
        return(-1)
    else:
        return(0)


dataset=pd.read_csv("D:/Program Files/netflix_tweets.csv")

dataset['text'] = dataset['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
dataset['text'] = dataset['text'].str.replace('[^\w\s]','')
dataset['text']=dataset['text'].str.lstrip('b')
corpus=[]
for i in range(0,3235):
    tweets=re.sub('[^a-zA-Z]',' ',dataset['text'][i])
    corpus.append(tweets)
dataset['text']=corpus  
stop = stopwords.words('english')
dataset['text'] = dataset['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#dataset['text'] = dataset['text'].apply(lambda x: x.split())
stemmer=PorterStemmer()
dataset['text'] = dataset['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in x.split()]))


f=open("D:/project/opinion-lexicon-English/positive-words.txt","r")
positive_words=[]
for x in f:
    positive_words.append(x.strip('\n'))
positive_words=positive_words[30:len(positive_words)]


f1=open("D:/project/opinion-lexicon-English/negative-words.txt","r")
negative_words=[]
for y in f1:
    negative_words.append(y.strip('\n'))
negative_words=negative_words[31:len(negative_words)]

score=[]
for j in dataset['text'].str.split():
    z=sentiment_score(j)
    score.append(z)

dataset['label']=score
print(dataset['label'].unique())


x_train=dataset.iloc[:,2]
y_train=dataset.iloc[:,3].values

from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=4000)
train_bow = bow.fit_transform(x_train).toarray()



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_bow,y_train,test_size=0.2,random_state=0)
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
ac=accuracy_score(y_test,y_pred)
print(cm)
print(ac)
