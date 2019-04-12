# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:12:30 2019

@author: nEW u
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import seaborn as sns

prime_tweets=pd.read_csv("C:/Users/nEW u/Documents/GitHub/project_data/primevideo_extracted_csv.csv")

def prime_hashtag(tweets):
    hashtags=[]
    for hashtag in tweets:
        ht = re.findall(r"#(\w+)", hashtag)
        hashtags.append(ht)
        
    return(hashtags)
    
n_hashtag=prime_hashtag(prime_tweets['text'])

n_hashtag=sum(n_hashtag,[])
for i in range(0,len(n_hashtag)):
    n_hashtag[i]=n_hashtag[i].lower()


frequence=nltk.FreqDist(n_hashtag)
df=pd.DataFrame({'hashtag':list(frequence.keys()),
                 'count':list(frequence.values())})
    
df = df.nlargest(columns="count", n = 10) 
plt.figure(figsize=(16,5))
graph = sns.barplot(data=df, x= "hashtag", y = "count")
graph.set(ylabel = 'count')
plt.show()