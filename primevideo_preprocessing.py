

import pandas as pd
from nltk.tokenize import WordPunctTokenizer
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import emoji
from nltk.stem.porter import PorterStemmer
 


dataset=pd.read_csv("C:/Users/nEW u/Documents/GitHub/project_data/primevideo_extracted_csv.csv",encoding='utf-8')

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
    ps=PorterStemmer()
    porter=[ps.stem(word) for word in tweets if not word in set(stopwords.words('english'))]
    porter=" ".join(porter)
    lem=WordNetLemmatizer()
    #print(tweets)
    lemma=[lem.lemmatize(word) for word in tweets if not word in set(stopwords.words('english'))]
    lemma=" ".join(lemma)
    #print(root)
    return(lemma)
    
twitter_data=[]
for tweet in dataset.text:
    twitter_data.append(clean_tweets(tweet))
    
cleaned_primevideo_tweets=pd.DataFrame(twitter_data)
cleaned_primevideo_tweets.to_csv("C:/Users/nEW u/Documents/GitHub/project_data/cleaned_primevideo_tweets.csv")
    