# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:27:58 2019

@author: nEW u
"""

import tweepy
import simplejson
from io import open



consumer_key="xxxx"
consumer_secret="xxxx"
access_key="xxxx"
access_secret="xxxx"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)
    
alltweets = []
new_tweets = api.user_timeline(screen_name = 'PrimeVideo',count=200)
alltweets.extend(new_tweets)
oldest = alltweets[-1].id - 1

while len(new_tweets) > 0:
    print ("getting tweets before %s" % (oldest))
    new_tweets = api.user_timeline(screen_name = 'PrimeVideo',count=200,max_id=oldest)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
    print( "...%s tweets downloaded so far" % (len(alltweets)))
    


with open('primevideo.txt', 'a') as f:
    for tweet in alltweets:
        simplejson.dump(tweet._json, f)
        f.write('\n')
    
