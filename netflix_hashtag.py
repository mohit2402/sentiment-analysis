

import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import seaborn as sns

netflix_tweets=pd.read_csv("C:/Users/nEW u/Documents/GitHub/project_data/netflix_extracted_csv.csv")

def netflix_hashtag(tweets):
    hashtags=[]
    for hashtag in tweets:
        ht = re.findall(r"#(\w+)", hashtag)
        hashtags.append(ht)
        
    return(hashtags)
    
n_hashtag=netflix_hashtag(netflix_tweets['text'])

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
plt.savefig('netflix_hashtag.png')