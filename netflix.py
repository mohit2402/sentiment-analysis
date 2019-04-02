# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:44:51 2019

@author: nEW u
"""
import tweepy #https://github.com/tweepy/tweepy
import csv
import scipy as sp
import simplejson
from __future__ import unicode_literals
from io import open
import sys

key=[]
f=open("C:/Users/nEW u/Documents/GitHub/sentiment-analysis/twitter_key.txt","r")
for x in f:
    key.append(x)

consumer_key=key[0]
consumer_secret=key[1] 
access_key=key[2] 
access_secret=key[3] 

def get_all_tweets(screen_name):
    
	#Twitter only allows access to a users most recent 3240 tweets with this method
	
	#authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	api = tweepy.API(auth)
    
	
	#initialize a list to hold all the tweepy Tweets
	alltweets = []
        
	#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = screen_name,count=200)
	
	#save most recent tweets
	alltweets.extend(new_tweets)
	
	#save the id of the oldest tweet less one
	#oldest = alltweets[-1].id - 1
	
	#keep grabbing tweets until there are no tweets left to grab
	#while len(new_tweets) > 0:
		#print ("getting tweets before %s" % (oldest))
		
		#all subsiquent requests use the max_id param to prevent duplicates
		#new_tweets = api.user_timeline(screen_name = screen_name,count=5,max_id=oldest)
		
		#save most recent tweets
		#alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
		#oldest = alltweets[-1].id - 1
		
		#print( "...%s tweets downloaded so far" % (len(alltweets)))
	
	#transform the tweepy tweets into a 2D array that will populate the csv	
	outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode('utf-8')] for tweet in alltweets]
    
	#write the csv	
	with open('%s_tweets.csv' % screen_name, 'wt',encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(["id","created_at","text"])
      
		writer.writerows(outtweets)
	
	pass



if __name__ == '__main__':
	#pass in the username of the account you want to download
	get_all_tweets("netflix")
