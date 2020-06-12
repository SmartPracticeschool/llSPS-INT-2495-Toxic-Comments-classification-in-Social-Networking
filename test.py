# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:57:07 2020

@author: Sanjay
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
# For the first time, if stopwords is not installed
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Importing the dataset into data variable of type DataFrame
data=pd.read_csv('train.csv')

# Checking for null data
data.isnull().any()

# Dropping the first column
data.drop(['id'],axis=1,inplace=True)
#Dropping the row 115606 
data.drop(115606,axis=0,inplace=True)

# Counting the number of comments classified into class labels
vis=list()
for i in range(1,len(data.columns)):
    vis.append(sum(data.iloc[:,i].values))

# Data Visualisation for the number of comments classified into class labels
plt.rcParams['figure.figsize']=(14,8)
plt.bar(['Toxic','Severe_Toxic','Obscene','Threat','Insult','Identity Hate'],vis)
plt.xlabel('Number of Comments')
plt.ylabel('Class Labels',)
plt.show()

c=[]
for i in range(159570):
    comment=re.sub('[^a-zA-Z]',' ', data.iloc[i,0])
    comment=comment.lower()
    comment=comment.split()
    # removing stopwords
    # Stemming
    ps=PorterStemmer()
    comment=[ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment=" ".join(comment)
    c.append(comment)

cv=CountVectorizer(max_features=2000) # top 1500 repeated words will be taken
x=cv.fit_transform(c).toarray() # a sparse matrix will be obtained
y=data.iloc[:,1:].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
 
#sving the model by pickling
pickle.dump(cv.vocabulary_,open("toxic_cmnts.pkl","wb"))

from keras.models import Sequential
from keras.layers import Dense
model=Sequential()

model.add(Dense(input_dim=2000,init="random_uniform",activation="sigmoid",output_dim=1000))
model.add(Dense(init="random_uniform",activation="sigmoid",output_dim=100))
model.add(Dense(output_dim=6,init='random_uniform',activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.save('toxic.h5')

model.fit(x_train,y_train,epochs=50,batch_size=32)
y_pred=model.predict(x_test)
y_pred=(y_pred>0.5)

#accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

