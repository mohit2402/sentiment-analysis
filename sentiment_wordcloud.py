# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:37:31 2019

@author: nEW u
"""

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

netflix_tweets=pd.read_csv("C:/Users/nEW u/Documents/GitHub/project_data/cleaned_netflix_tweets.csv")
netflix_tweets=netflix_tweets.dropna()
all_words_netflix = ' '.join([text for text in netflix_tweets['0']])
wordcloud_netflix = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_netflix)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_netflix, interpolation="bilinear")
plt.axis('off')
plt.show()


prime_tweets=pd.read_csv("C:/Users/nEW u/Documents/GitHub/project_data/cleaned_primevideo_tweets.csv")
prime_tweets=prime_tweets.dropna()
all_words_prime = ' '.join([text for text in prime_tweets['0']])
wordcloud_prime = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_prime)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_prime, interpolation="bilinear")
plt.axis('off')
plt.show()