#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# # Exploring the Dataset

# In[2]:


df=pd.read_csv("spam.csv",encoding='latin-1')


# In[3]:


df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)


# In[4]:


df.head()


# In[5]:


df.rename(columns={'v1':'Class','v2':'sms'},inplace=True)


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.groupby('Class').describe()


# In[10]:


sns.countplot(data=df,x='Class')


# In[11]:


df['label']=df.Class.map({'ham':0,'spam':1})


# In[12]:


df.head()


# In[13]:


X=df.sms
y=df.label


# In[14]:


print(X.shape)
print(y.shape)


# # Split Datset Training and Test 

# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer


# In[19]:


vec=CountVectorizer(stop_words='english')


# In[20]:


vec.fit(X_train)


# In[21]:


vec.vocabulary_


# In[22]:


print(vec.get_feature_names)
print(len(vec.get_feature_names()))


# In[23]:


X_train_transfrom = vec.transform(X_train)
X_test_transfrom = vec.transform(X_test)


# In[24]:


print(type(X_train_transfrom))
print(X_train_transfrom)


# # Classifier Using Multinomial NaiveBayes

# In[25]:


from sklearn.naive_bayes import MultinomialNB


# In[26]:


mnb= MultinomialNB()


# In[27]:


mnb.fit(X_train_transfrom,y_train)


# In[30]:


predict = mnb.predict(X_test_transfrom)
predict_prob=mnb.predict_proba(X_test_transfrom)


# # Classification Report and Confusion Matrix

# In[31]:


from sklearn.metrics import confusion_matrix,classification_report


# In[38]:


print(confusion_matrix(y_test,predict))
sns.heatmap(confusion_matrix(y_test,predict),annot=True)


# In[39]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: {}'.format(accuracy_score(y_test, predict)))
print('Precision score: {}'.format(precision_score(y_test, predict)))
print('Recall score: {}'.format(recall_score(y_test, predict)))
print('F1 score: {}'.format(f1_score(y_test, predict)))


# In[40]:


print(classification_report(y_test,predict))


# In[41]:


predict


# In[42]:


predict_prob


# In[43]:


from sklearn.metrics import roc_curve, auc


# In[44]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predict_prob[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[45]:


print(roc_auc)


# In[46]:


print(true_positive_rate)


# In[47]:


print(false_positive_rate)


# In[48]:


print(thresholds)


# In[49]:


New_df=pd.DataFrame({'Threshold': thresholds, 
              'TPR': true_positive_rate, 
              'FPR':false_positive_rate
             })


# In[50]:


New_df


# In[51]:


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate)


# # Classifier Using Bernoulli NaiveBayes

# In[52]:


from sklearn.naive_bayes import BernoulliNB


# In[53]:


bnb = BernoulliNB()


# In[54]:


bnb.fit(X_train_transfrom,y_train)


# In[55]:


b_predict=bnb.predict(X_test_transfrom)
b_predict_prob=bnb.predict_proba(X_test_transfrom)


# # Classification Report and Confusion Matrix

# In[58]:


print(confusion_matrix(y_test,b_predict))
sns.heatmap(confusion_matrix(y_test,b_predict),annot=True)


# In[59]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: {}'.format(accuracy_score(y_test, b_predict)))
print('Precision score: {}'.format(precision_score(y_test, b_predict)))
print('Recall score: {}'.format(recall_score(y_test, b_predict)))
print('F1 score: {}'.format(f1_score(y_test, b_predict)))


# In[60]:


print(classification_report(y_test,b_predict))


# In[61]:


lse_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predict_prob[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[62]:


print(roc_auc)


# In[63]:


print(true_positive_rate)


# In[64]:


print(false_positive_rate)


# In[65]:


print(thresholds)


# In[66]:


New_dff=pd.DataFrame({'Threshold': thresholds, 
              'TPR': true_positive_rate, 
              'FPR':false_positive_rate
             })


# In[67]:


print(New_dff)


# In[68]:


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate)


# In[ ]:




