#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[23]:


dataframe = pd.read_csv('news.csv')
dataframe.head()


# In[19]:


x = dataframe['text']
y = dataframe['label']


# In[4]:


x


# In[5]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[6]:


tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)


# In[7]:


classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfid_x_train,y_train)


# In[8]:


y_pred = classifier.predict(tfid_x_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[9]:


cf = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(cf)


# In[10]:


def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = classifier.predict(vectorized_input_data)
    print(prediction)


# In[13]:


fake_news_det("""Go to Article President Barack Obama has been 
campaigning hard for the woman who is supposedly going to extend his legacy 
four more years. The only problem with stumping for Hillary Clinton, however, 
is sheâ€™s not exactly a candidate easy to get too enthused about.  """)


# In[14]:


import pickle
pickle.dump(classifier,open('model.pkl', 'wb'))


# In[ ]:




