

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
plt.figure(figsize=(20,8))
graph = sns.barplot(data=df, x= "hashtag", y = "count")
graph.set(ylabel = 'count')

for i in graph.patches:
    graph.text(i.get_x()+.03,i.get_height()+.5,str(i.get_height()),fontsize=15)
plt.title("prime_hashtags",font_size=15)
plt.savefig('prime_hashtag.png')