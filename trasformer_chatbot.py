
import pandas as pd
import json
import numpy as np
from urllib import request 
import os
request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
train_data = pd.read_csv('ChatBotData.csv')
res_data=pd.read_excel('/Users/bagjeongho/Desktop/chatbot1-main/A 음식점(15,726).xlsx')
train_data=train_data.iloc[:,0:2]
base_dir='/Users/bagjeongho/Desktop/chatbot1-main/'
file_name=['A 음식점(15,726).xlsx','B 의류(15,826).xlsx','C 학원(4,773).xlsx','D 소매점(14,949).xlsx','D 소매점(14,949).xlsx','F 카페(7,859).xlsx','G 숙박업(7,113).xlsx','H 관광여가오락(4,949).xlsx','I 부동산(8,131).xlsx']
for i in range(len(file_name)):
    if i==0:
        data=pd.read_excel(base_dir+file_name[i])
    else:
        data1=pd.read_excel(base_dir+file_name[i])
        data=pd.concat((data,data1),axis=0)
        
data['idx']=np.arange(data.shape[0])
data.set_index('idx',inplace=True)
for i in range(data.shape[0]-2):
    if i%2==0 :
        if i==0:
            data__=pd.DataFrame({"Q":[data.iloc[i:i+2,1][0]],"A":[data.iloc[i:i+2,1][1]]})
        else:
       
            q=data.iloc[i:i+2,1][i]
            l=data.iloc[i:i+2,1][i+1]
            data__.loc[i]=[q,l]
    else:
        continue
real_data=pd.concat((train_data,data__),axis=0)
real_data.to_csv("/Users/bagjeongho/Desktop/chatbot1-main/chatbot_data.csv")