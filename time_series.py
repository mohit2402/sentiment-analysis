
import pandas as pd
import datetime
import matplotlib.pyplot as plt


netflix_data=pd.read_csv("C:/Users/nEW u/Documents/GitHub/project_data/netflix_twitter_csv.csv")
prime_data=pd.read_csv("C:/Users/nEW u/Documents/GitHub/project_data/primevideo_twitter_csv.csv")



ex_netflix_data=netflix_data.iloc[:,[2,26]]
ex_prime_data=prime_data.iloc[:,[2,26]]


netflix_dates=[]
for i in range(0,len(ex_netflix_data['created_at'])):
    ex_netflix_data['created_at'][i]=datetime.datetime.strptime(ex_netflix_data['created_at'][i],'%a %b %d %H:%M:%S %z %Y')
    netflix_dates.append(ex_netflix_data['created_at'][i].date())
    
prime_dates=[]
for i in range(0,len(ex_prime_data['created_at'])):
    ex_prime_data['created_at'][i]=datetime.datetime.strptime(ex_prime_data['created_at'][i],'%a %b %d %H:%M:%S %z %Y')
    prime_dates.append(ex_prime_data['created_at'][i].date())


def frequency(dates):
    d='2018-10-08'
    date_freq={}
    for date in dates:
        if(date > datetime.datetime.strptime(d,'%Y-%m-%d').date()):
            if(date in date_freq):
                date_freq[date]+=1
            else:
                date_freq[date]=1
            
    return(date_freq)
    
netflix_freq=frequency(netflix_dates)
netflix_freq=pd.DataFrame.from_dict(netflix_freq,orient='index')
netflix_freq.rename(columns={0:'netflix_tweets'},inplace=True)

prime_freq=frequency(prime_dates)
prime_freq=pd.DataFrame.from_dict(prime_freq,orient='index')
prime_freq.rename(columns={0:'prime_tweets'},inplace=True)


ax=netflix_freq.plot(figsize=(10,7),linewidth=2,fontsize=10)
prime_freq.plot(ax=ax,figsize=(10,7),linewidth=2,fontsize=10)
plt.xlabel('year_month',fontsize=15)
plt.ylabel('tweets',fontsize=15)
plt.title("tweets timeseries",fontsize=15)
plt.savefig('tweets_timeseries.png')
