# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 18:59:28 2019

@author: nEW u
"""

import tweepy #https://github.com/tweepy/tweepy
import csv
import simplejson
from __future__ import unicode_literals
from io import open
import sys
import numpy as np

key=[]
f=open("C:/Users/nEW u/Documents/GitHub/sentiment-analysis/twitter_key.txt","r")
for x in f:
    key.append(x)

consumer_key=key[0]
consumer_secret=key[1] 
access_key=key[2] 
access_secret=key[3] 

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)
    
alltweets = []
new_tweets = api.user_timeline(screen_name = 'netflix',count=200)
alltweets.extend(new_tweets)
oldest = alltweets[-1].id - 1

while len(new_tweets) > 0:
    print ("getting tweets before %s" % (oldest))
    new_tweets = api.user_timeline(screen_name = 'netflix',count=200,max_id=oldest)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
    print( "...%s tweets downloaded so far" % (len(alltweets)))
    
#outtweets = [[tweet.id, tweet.created_at, tweet.text.encode('utf-8')] for tweet in alltweets]

#np.savetxt("filename.txt",outtweets,)

with open('filename.txt', 'a') as f:
    for tweet in alltweets:
        simplejson.dump(tweet._json, f)
        f.write('\n')
    
