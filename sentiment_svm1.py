# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:03:21 2019

@author: nEW u
"""


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score

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
        

netflix_tweets=pd.read_csv("C:/Users/nEW u/Documents/GitHub/project_data/cleaned_netflix_tweets.csv")
prime_tweets=pd.read_csv("C:/Users/nEW u/Documents/GitHub/project_data/cleaned_primevideo_tweets.csv")

positive=open("C:/Users/nEW u/Documents/GitHub/project_data/opinion-lexicon-English/positive-words.txt")
negative=open("C:/Users/nEW u/Documents/GitHub/project_data/opinion-lexicon-English/negative-words.txt")

positive_words=[]
for pos_word in positive:
    positive_words.append(pos_word.strip('\n'))
positive_words=positive_words[30:len(positive_words)]

negative_words=[]
for neg_word in negative:
    negative_words.append(neg_word.strip('\n'))
negative_words=negative_words[31:len(negative_words)]

netflix_tweets=netflix_tweets.dropna()
prime_tweets=prime_tweets.dropna()

netflix_tweets=netflix_tweets.iloc[:,-1]
netflix_tweets=netflix_tweets.to_frame(name='text')
prime_tweets=prime_tweets.iloc[:,-1]
prime_tweets=prime_tweets.to_frame(name='text')

netflix_score=[]
for n_tweet in netflix_tweets.text.str.split():
    n_tweet_score=sentiment_score(n_tweet)
    netflix_score.append(n_tweet_score)


prime_score=[]
for p_tweet in prime_tweets.text.str.split():
    p_tweet_score=sentiment_score(p_tweet)
    prime_score.append(p_tweet_score)

    
netflix_tweets['label']=netflix_score

prime_tweets['label']=prime_score

netflix_x=netflix_tweets.iloc[:,0]
netflix_y=netflix_tweets.iloc[:,-1]

prime_x=prime_tweets.iloc[:,0]
prime_y=prime_tweets.iloc[:,-1]

bow=CountVectorizer(max_features=3500)
netflix_bow=bow.fit_transform(netflix_x).toarray()
prime_bow=bow.fit_transform(prime_x).toarray()


netflix_x_train,netflix_x_test,netflix_y_train,netflix_y_test=train_test_split(netflix_bow,netflix_y,test_size=0.2,random_state=0)

prime_x_train,prime_x_test,prime_y_train,prime_y_test=train_test_split(prime_bow,prime_y,test_size=0.2,random_state=0)

x_train=np.concatenate((netflix_x_train,prime_x_train),axis=0)
y_train=np.concatenate((netflix_y_train,prime_y_train),axis=None)


classifier=SVC(kernel='rbf',C=10,gamma=0.1,random_state=0)
classifier.fit(x_train,y_train)

netflix_pred=classifier.predict(netflix_x_test)
prime_pred=classifier.predict(prime_x_test)

netflix_cm=confusion_matrix(netflix_y_test,netflix_pred)
prime_cm=confusion_matrix(prime_y_test,prime_pred)

netflix_ac=accuracy_score(netflix_y_test,netflix_pred)
prime_ac=accuracy_score(prime_y_test,prime_pred)

print(netflix_cm)
print(prime_cm)
print(netflix_ac)
print(prime_ac)


accuracies=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10)
accuracies.mean()

from sklearn. model_selection import GridSearchCV
parameters=[{'C':[1,10,100],'kernel':['linear']},
            {'C':[1,10,100],'kernel':['rbf'],'gamma':[0.5,0.1,0.01]}]
grid_search=GridSearchCV(estimator=classifier,
                        param_grid=parameters,
                        scoring='accuracy',
                        cv=10,
                        n_jobs=-1)
grid_search=grid_search.fit(x_train,y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_

