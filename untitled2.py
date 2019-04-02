# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:32:31 2019

@author: nEW u
"""

import json
import pandas as pd



data=[]
file1=open("D:/Program Files/filename.txt",'r')

for line in file1:
    data.append(json.loads(line))

twitter_dataset=pd.DataFrame(data)

twitter_dataset.to_csv("D:/Program Files/netflix_twitter_file.csv",index=False,encoding='utf-8')

extracted_twitter_dataset=twitter_dataset.iloc[:,[8,2,26]]

extracted_twitter_dataset.to_csv("D:/Program Files/netflix_extracted_file.csv",index=False,encoding='utf-8')



