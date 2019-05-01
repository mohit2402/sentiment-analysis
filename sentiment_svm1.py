

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score

def sentiment_score(l):
    p=0
    n=0
    #print(l)
    for each in l:
       # print(each)
        if each in positive_words:
            
            p=p+1
            #print(p)
        elif each in negative_words:
            n=n+1
            #print(n)
       
    if((p-n)>=1):
        return(1)
    elif((p-n)<=-1):
        return(-1)
    else:
        return(0)
        

netflix_tweets=pd.read_csv("C:/Users/nEW u/Documents/GitHub/project_data/cleaned_netflix_tweets.csv")
prime_tweets=pd.read_csv("C:/Users/nEW u/Documents/GitHub/project_data/cleaned_primevideo_tweets.csv")

positive=open("C:/Users/nEW u/Documents/GitHub/project_data/opinion-lexicon-English/positive-words.txt")
negative=open("C:/Users/nEW u/Documents/GitHub/project_data/opinion-lexicon-English/negative-words.txt")

positive_words=[]
for pos_word in positive:
    positive_words.append(pos_word.strip('\n'))
positive_words=positive_words[30:len(positive_words)]

negative_words=[]
for neg_word in negative:
    negative_words.append(neg_word.strip('\n'))
negative_words=negative_words[31:len(negative_words)]

netflix_tweets=netflix_tweets.dropna()
prime_tweets=prime_tweets.dropna()

netflix_tweets=netflix_tweets.iloc[:,-1]
netflix_tweets=netflix_tweets.to_frame(name='text')
prime_tweets=prime_tweets.iloc[:,-1]
prime_tweets=prime_tweets.to_frame(name='text')

netflix_score=[]
for n_tweet in netflix_tweets.text.str.split():
    n_tweet_score=sentiment_score(n_tweet)
    netflix_score.append(n_tweet_score)


prime_score=[]
for p_tweet in prime_tweets.text.str.split():
    p_tweet_score=sentiment_score(p_tweet)
    prime_score.append(p_tweet_score)

    
netflix_tweets['label']=netflix_score

prime_tweets['label']=prime_score

netflix_x=netflix_tweets.iloc[:,0]
netflix_y=netflix_tweets.iloc[:,-1]

prime_x=prime_tweets.iloc[:,0]
prime_y=prime_tweets.iloc[:,-1]

bow=CountVectorizer(max_features=3500)
netflix_bow=bow.fit_transform(netflix_x).toarray()
prime_bow=bow.fit_transform(prime_x).toarray()


netflix_x_train,netflix_x_test,netflix_y_train,netflix_y_test=train_test_split(netflix_bow,netflix_y,test_size=0.2,random_state=0)

prime_x_train,prime_x_test,prime_y_train,prime_y_test=train_test_split(prime_bow,prime_y,test_size=0.2,random_state=0)

x_train=np.concatenate((netflix_x_train,prime_x_train),axis=0)
y_train=np.concatenate((netflix_y_train,prime_y_train),axis=None)

total_x_test=np.concatenate((netflix_x_test,prime_x_test),axis=0)
total_y_test=np.concatenate((netflix_y_test,prime_y_test),axis=None)



classifier=SVC(kernel='rbf',C=10,gamma=0.1,random_state=0)
classifier.fit(x_train,y_train)

netflix_pred=classifier.predict(netflix_x_test)
prime_pred=classifier.predict(prime_x_test)
total_pred=classifier.predict(total_x_test)

def frequency(labels):
    label_freq={}
    for l in labels:
        if(l in label_freq):
            label_freq[l]+=1
        else:
            label_freq[l]=1
            
    return(label_freq)
    

netflix_label=frequency(netflix_pred)
prime_label=frequency(prime_pred)

pred_labels=pd.DataFrame({'netflix':netflix_label,'prime':prime_label})
a=pred_labels.plot.bar(figsize=(10,8),fontsize=20,rot=360)
for i, each in enumerate(pred_labels.index):
    z=0
    for col in pred_labels.columns:
        y = pred_labels.ix[each][col]
        if(z==0):
            a.text(i-0.2,y/2,y,fontsize=15)
            z=1
        else:
            a.text(i+0.05,y/2,y,fontsize=15)
 
plt.title("opinion tweets",fontsize=20)
plt.xlabel('tweet opinion',fontsize=20)
plt.ylabel("no. of tweets")
plt.savefig('opinion tweets.png')


netflix_cm=confusion_matrix(netflix_y_test,netflix_pred)
prime_cm=confusion_matrix(prime_y_test,prime_pred)
total_cm=confusion_matrix(total_y_test,total_pred)


netflix_ac=accuracy_score(netflix_y_test,netflix_pred)
prime_ac=accuracy_score(prime_y_test,prime_pred)
total_ac=accuracy_score(total_y_test,total_pred)


print(netflix_cm)
print(prime_cm)
print(netflix_ac)
print(prime_ac)

print(total_cm)
print(total_ac)

from sklearn.metrics import precision_recall_fscore_support
n_fmeasure=precision_recall_fscore_support(netflix_y_test, netflix_pred, average=None,labels=[-1, 0, 1])
n_df=pd.DataFrame.from_records(n_fmeasure)
n_df=n_df.drop([3])
n_df.rename(columns={0:'precision',1:'recall',2:'f1-score'},inplace=True)
n_df=n_df.rename({0:'negative',1:'neutral',2:'positive'})

#n_df.plot.bar(figsize=(10,8),fontsize=20,rot=360)
#plt.title("fmeasure",fontsize=20)
#plt.xlabel('netflix tweet opinion',fontsize=20)
#plt.savefig('netflix_fmeasure.png')


p_fmeasure=precision_recall_fscore_support(prime_y_test, prime_pred, average=None,labels=[-1, 0, 1])
p_df=pd.DataFrame.from_records(p_fmeasure)
p_df=p_df.drop([3])
p_df.rename(columns={0:'precision',1:'recall',2:'f1-score'},inplace=True)
p_df=p_df.rename({0:'negative',1:'neutral',2:'positive'})

#p_df.plot.bar(figsize=(10,8),fontsize=20,rot=360)
#plt.title("fmeasure",fontsize=20)
#plt.xlabel('prime tweet opinion',fontsize=20)
#plt.savefig('prime_fmeasure.png')

precision_df=pd.DataFrame({'netflix_precision':n_df['precision'],'prime_precision':p_df['precision']},index=['negative','neutral','positive'])
ax=precision_df.plot.bar(figsize=(10,8),fontsize=20,rot=360)

for i, each in enumerate(precision_df.index):
    z=0
    for col in precision_df.columns:
        y = precision_df.ix[each][col]
        if(z==0):
            ax.text(i-0.2,y/2,'{0:.2f}'.format(y),fontsize=15)
            z=1
        else:
            ax.text(i+0.05,y/2,'{0:.2f}'.format(y),fontsize=15)
    
plt.title("precision",fontsize=20)
plt.xlabel('tweet opinion',fontsize=20)
plt.savefig('precision.png')


recall_df=pd.DataFrame({'netflix_recall':n_df['recall'],'prime_recall':p_df['recall']},index=['negative','neutral','positive'])
ax1=recall_df.plot.bar(figsize=(10,8),fontsize=20,rot=360)
for i, each in enumerate(recall_df.index):
    z=0
    for col in recall_df.columns:
        y = recall_df.ix[each][col]
        if(z==0):
            ax1.text(i-0.2,y/2,'{0:.2f}'.format(y),fontsize=15)
            z=1
        else:
            ax1.text(i+0.05,y/2,'{0:.2f}'.format(y),fontsize=15)
    
plt.title("recall",fontsize=20)
plt.xlabel('tweet opinion',fontsize=20)
plt.savefig('recall.png')





avg_n_fmeasure=precision_recall_fscore_support(netflix_y_test, netflix_pred, average='macro',labels=[-1, 0, 1])
avg_p_fmeasure=precision_recall_fscore_support(prime_y_test, prime_pred, average='macro',labels=[-1, 0, 1])



avg_fmeasure=pd.DataFrame({'netflix':avg_n_fmeasure,'prime':avg_p_fmeasure})
avg_fmeasure=avg_fmeasure.drop([3],axis=0)
avg_fmeasure=avg_fmeasure.rename({0:'precision',1:'recall',2:'f1-score'})


ax3=avg_fmeasure.plot.bar(figsize=(10,8),fontsize=20,rot=360)
for i, each in enumerate(avg_fmeasure.index):
    z=0
    for col in avg_fmeasure.columns:
        y = avg_fmeasure.ix[each][col]
        if(z==0):
            ax3.text(i-0.2,y/2,'{0:.2f}'.format(y),fontsize=15)
            z=1
       
        else:
            ax3.text(i+0.05,y/2,'{0:.2f}'.format(y),fontsize=15)
        
    
plt.title("evaluation metrics",fontsize=20)
plt.xlabel('online streaming',fontsize=20)
plt.savefig('evaluation_metrics.png')



import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(netflix_y_test, netflix_pred, normalize=True,title='netflix confusion matrix')
plt.savefig('netflix_confusion_matrix.png')

skplt.metrics.plot_confusion_matrix(prime_y_test, prime_pred, normalize=True,title='prime confusion matrix')
plt.savefig('prime_confusion_matrix.png')

skplt.metrics.plot_confusion_matrix(total_y_test, total_pred, normalize=True,title='confusion matrix')
plt.savefig('confusion_matrix.png')



accuracies=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10)
accuracies.mean()

from sklearn. model_selection import GridSearchCV
parameters=[{'C':[1,10,100],'kernel':['linear']},
            {'C':[1,10,100],'kernel':['rbf'],'gamma':[0.5,0.1,0.01]}]
grid_search=GridSearchCV(estimator=classifier,
                        param_grid=parameters,
                        scoring='accuracy',
                        cv=10,
                        n_jobs=-1)
grid_search=grid_search.fit(x_train,y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_

