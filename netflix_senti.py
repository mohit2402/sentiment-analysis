# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:53:36 2019

@author: nEW u
"""

import pandas as pd
from nltk.tokenize import WordPunctTokenizer
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup  
import string 
import emoji
import numpy as np
 


dataset=pd.read_csv("D:/Program Files/netflix_extracted_file.csv",encoding='utf-8')

token=WordPunctTokenizer()
part1=r'@[A-Za-z0-9]+'
part2=r'https?://[_A-Za-z0-9./]+'
part3=r'http?://[_A-Za-z0-9./]+'
combine=r'|'.join((part1,part2,part3))

def clean_tweets(tweet):
    #soup=BeautifulSoup(tweet,'html.parser')
    #souped=soup.get_text()
    clean_emoji=emoji.demojize(tweet)
    stripped=re.sub(combine,'',clean_emoji)
   # text= "".join([char for char in stripped if char not in string.punctuation])
    lower_case=stripped.lower()
    tweets=re.sub('[^a-zA-Z]',' ',lower_case)
    tweets=tweets.strip('rt')
    #print(tweets)
    tweets=tweets.split()
    lem=WordNetLemmatizer()
    #print(tweets)
    lemma=[lem.lemmatize(word) for word in tweets if not word in set(stopwords.words('english'))]
    lemma=" ".join(lemma)
    #print(root)
    return(lemma)
    
twitter_data=[]
for tweet in dataset.text:
    twitter_data.append(clean_tweets(tweet))
    

    