#!/usr/bin/env python
# coding: utf-8

# ***

# # Libraries required for the Project

# In[1]:


import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix
from collections import Counter
from nltk.corpus import stopwords
from sklearn.svm import SVC
import nltk
plt.style.use("ggplot")
nltk.download('names')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')


# ***

# ## Data Reading

# In[2]:


data=fetch_20newsgroups()


# In[3]:


#Getting the keys from given dataset
data.keys()


# In[4]:


#Data Description
print(data["DESCR"])


# In[5]:


data["target"]


# In[6]:


#type of target categories 
data["target_names"]


# In[7]:


#Actual text data
data["data"]


# ***

# # So, the basic workflow of our project will go like
# 
# # Data reading-->Data cleaning(In this case, text cleaning using NLP techniques)-->Data Transformation(To model understandable type)--> Model Training and Testing

# ## Dividing our dataset into training and testing...
# 
# ## We will be working with only 3 categories to predict...

# In[8]:


#Our target variables
target=['sci.electronics','rec.sport.hockey','talk.politics.guns']
#Training data
df_train= fetch_20newsgroups(subset='train', categories=target, random_state=101)
#Testing data
df_test= fetch_20newsgroups(subset='test', categories=target, random_state=101)


# In[9]:


df_train


# In[10]:


df_train["data"]


# ***

# ## So, now we will be cleaning our datasets by removing unwanted characters like punctuations, stopwords, and lemmatizing different words

# In[11]:


#We will be creating a function to clean our textual data


# In[12]:


# NLTK- natural language toolkit
stop_words= stopwords.words('english')# stopwords- a, the, an ,for, is...etc
#all_names
all_names=set(names.words())
# preprocessing
#- lower case
#- root form run, running- run (lemmatization)

lemma= WordNetLemmatizer()
def is_letter_only(word):
    return word.isalpha()


# In[13]:


#Definign a fucntion to clean our text data for both training and testing dataset...by lemmatizing and removing stopwords...
def clean_text(doc):
    doc_clean=[]
    for i in doc:
        i=i.lower()
        i_clean=' '.join(lemma.lemmatize(word) for word in i.split() if is_letter_only(word) and word not in all_names and word not in stop_words)
        doc_clean.append(i_clean)
    return doc_clean


# In[14]:


#We will be cleaning our data
#clean train data
df_train_clean= clean_text(df_train.data)
df_train_label= df_train.target

#clean test data
df_test_clean= clean_text(df_test.data)
df_test_label= df_test.target


# In[15]:


#Counting our target values
print("Training label",Counter(df_train_label))
print("Testing label",Counter(df_test_label))


# In[16]:


#Now our text data is clean and ready to be converted to numeric form by using tf-idf vectorizer


# In[17]:


tfidf= TfidfVectorizer(stop_words='english') #basically stopping all the english basic words
df_train_conv= tfidf.fit_transform(df_train_clean)
df_test_conv= tfidf.transform(df_test_clean)


# ***

# ## Now our data is clean and ready to perform predictive analysis using SVC

# In[18]:


#Our SVC model
model= SVC() #default model
model.fit(df_train_conv, df_train_label) #model fitting with train data
acc= model.score(df_test_conv,df_test_label)
print("The accuracy of binary classification is : {}%".format(round(acc*100,2))) #Model accuracy score

#Model predictions
y_pred=model.predict(df_test_conv)

#Model Evaluation
print(classification_report(df_test_label, y_pred))
print(confusion_matrix(df_test_label, y_pred))
plot_confusion_matrix(model,df_test_conv,df_test_label)


# ## So, Our model performed with an overall accuracy of 97.3%..Similarly, We can peform with multi-class as well...

# ***
