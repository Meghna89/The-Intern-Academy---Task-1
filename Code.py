#!/usr/bin/env python
# coding: utf-8

# # The Intern Academy #

# ### TASK 1- Titanic EDA (Exploratory Data Analysis)###

# ### NAME: Meghna Chakraborty###

# ### Task Details : Every task has a story. Tell users what this task is all about and why you created it. Analysing Titanic Data Set.###

# ### Dataset : https://www.kaggle.com/shuofxz/titanic-machine-learning-from-disaster/tasks?taskId=2692 ###

# ### Importing required libraries ###

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:



titanic = pd.read_csv("train.csv")
titanic.head(10)


# ### Data Analysis ###

# In[3]:


titanic.shape


# In[4]:


titanic.columns


# In[5]:


titanic.info()


# In[6]:


titanic.isna().sum()


# ### Data Visualization ###

# In[7]:


import matplotlib.pyplot as plt


plt.figure(figsize=(10,10))
sns.heatmap(titanic.corr(), annot=True, linewidths=0.5, fmt= '.3f')


# In[8]:


titanic.corr()


# In[9]:


def woman_or_ch_or_man(passenger):
    age, sex = passenger
    if age < 16:
        return "child"
    else:
        return dict(male="man", female="woman")[sex]


# In[10]:


titanic["who"] = titanic[["Age", "Sex"]].apply(woman_or_ch_or_man, axis=1)
titanic.head()


# In[11]:


titanic["adult_male"] = titanic.who == "man"
titanic.head()


# In[12]:


titanic["deck"] = titanic.Cabin.str[0]
titanic.head()


# In[13]:


titanic["alone"] = ~(titanic. Parch + titanic.SibSp  ).astype(bool)
titanic.head()


# In[15]:


sns.factorplot("Pclass", "Survived", data=titanic).set(ylim=(0, 1))


# In[16]:


sns.factorplot("Pclass", "Survived", data=titanic, hue="Sex")


# In[17]:


sns.factorplot("Pclass", "Survived", data=titanic, hue="who")


# In[18]:


sns.factorplot("alone", "Survived", data=titanic, hue="Sex")


# In[19]:


sns.factorplot("adult_male", "Survived", data=titanic, hue="Sex")


# In[20]:


sns.barplot("deck", "Survived", data=titanic,order=['A','B','C','D','E','F','G'])


# In[21]:


sns.factorplot("alone", "Survived", data=titanic, hue="Sex",col="Pclass")


# ### Data Preprocessing ###

# In[22]:


#encoding deck

dk = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
titanic['deck']=titanic.deck.map(dk)
titanic.head()


# In[23]:


# encoding embarked


titanic['Embarked'].value_counts()


# In[24]:


e = {'S':3,'Q':2, 'C':1}
titanic['Embarked']=titanic.Embarked.map(e)
titanic.head()


# In[25]:


# encoding gender

genders = {"male": 0, "female": 1}
titanic['Sex'] = titanic['Sex'].map(genders)
titanic.head()


# In[26]:


#encoding who

wh = {'child':3,'woman':2, 'man':1}
titanic['who']=titanic.who.map(wh)


# In[27]:



titanic.head()


# In[28]:


#imputing deck
titanic['deck']=titanic['deck'].fillna(0)
titanic.head()


# In[29]:


#imputing embarked

titanic['Embarked'].value_counts()


# In[30]:


titanic['Embarked']=titanic['Embarked'].fillna('3.0')
titanic.head()


# In[31]:


#imputing age

m=titanic['Age'].mean()
m


# In[32]:


titanic['Age']=titanic['Age'].fillna(m)
titanic.head()


# ### Adding New Features ###

# In[33]:


def process_family(parameters):
     
    x,y=parameters
    
    
    family_size = x+ y + 1
    
    if (family_size==1):
      return 1 # for singleton
    elif(2<= family_size <= 4 ):
      return 2 #for small family
    else:
      return 3 #for big family


# In[34]:


titanic['FAM_SIZE']= titanic[['Parch','SibSp']].apply(process_family, axis=1)
titanic.head()


# In[35]:


# to get title from the name.

titles = set()
for name in titanic['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())


# In[36]:


titles


# In[37]:


len(titles)


# In[38]:


title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}


# In[39]:


def get_titles():
    
    titanic['title'] = titanic['Name'].map(lambda Name:Name.split(',')[1].split('.')[0].strip())
    
    
    
    titanic['title'] = titanic.title.map(title_Dictionary)
    return titanic


# In[40]:


titanic = get_titles()
titanic.head()


# In[41]:


titles_dummies = pd.get_dummies(titanic['title'], prefix='title')
titanic = pd.concat([titanic, titles_dummies], axis=1)
titanic.head()


# In[42]:


def new_fe(parameters):
  p,w=parameters
  
  if (p==1):
    if (w==1):
      return 1
    elif (w==2):
      return 2
    elif (w==3):
      return 3
  elif (p==2):
    if (w==1):
      return 4
    elif (w==2):
      return 5
    elif (w==3):
      return 6
  elif (p==3):
    if (w==1):
      return 7
    elif (w==2):
      return 8
    elif (w==3):
      return 9


# In[43]:


titanic['pcl_wh']= titanic[['Pclass','who']].apply(new_fe, axis=1)
titanic.head()


# In[44]:


titanic.columns


# In[45]:


drop_list=['Name','Ticket','Fare', 'Cabin','title']
titanic = titanic.drop(drop_list, axis=1)
titanic.head()


# In[46]:


plt.figure(figsize=(20,20))
sns.heatmap(titanic.corr(), annot=True, linewidths=0.5, fmt= '.3f')


# ### Build the Models ### 

# In[47]:


X_train = titanic.drop("Survived", axis=1)
Y_train = titanic["Survived"]


# In[48]:


from sklearn.model_selection import train_test_split

# splitting data in training set(80%) and test set(20%).
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2)


# ### Logistic Regression ### 

# In[49]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression() #create the object of the model
lr = lr.fit(x_train,y_train)


# In[50]:


from sklearn.metrics import accuracy_score, confusion_matrix

act = accuracy_score(y_train,lr.predict(x_train))
print('Training Accuracy is: ',(act*100))


# In[51]:


act = accuracy_score(y_test,lr.predict(x_test))
print('Test Accuracy is: ',(act*100))


# ### Decision Tree Classifier ###

# In[52]:


from sklearn.tree import DecisionTreeClassifier


dt = DecisionTreeClassifier()
dt=dt.fit(x_train, y_train)


# In[53]:


act = accuracy_score(y_train,dt.predict(x_train))
print('Training Accuracy is: ',(act*100))


# In[54]:


act = accuracy_score(y_test,dt.predict(x_test))
print('Test Accuracy is: ',(act*100))


# ### Random Forest Classifier ###

# In[55]:


from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 3, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features=0.5, 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)
rf = rf.fit(x_train,y_train)


# In[56]:


act = accuracy_score(y_train,rf.predict(x_train))
print('Training Accuracy is: ',(act*100))


# In[57]:


act = accuracy_score(y_test,rf.predict(x_test))
print('Test Accuracy is: ',(act*100))


# The test accuracy of the Random Forest Classifier is 85.47486033519553, thus 
# it's better and will be chosen as the final model.
