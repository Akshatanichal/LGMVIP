#!/usr/bin/env python
# coding: utf-8

# # LetsGrowMore: Data Science
# ## Task02 (Intermediate Level task): Prediction using Decision tree Algorithm
# ### Name of Intern:Akshata Naganath Nichal

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report,confusion_matrix


# In[2]:


data=pd.read_csv("C:/Users/91986/Downloads/IRIS.csv")
data.head(10)


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


data['Species'].unique()


# In[8]:


data['Species'].value_counts()


# ### Data Visualization

# In[9]:


sns.pairplot(data,hue='Species')


# In[10]:


sns.violinplot(y='Species', x='SepalLengthCm', data=data, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='SepalWidthCm', data=data, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='PetalLengthCm', data=data, inner='quartile')
plt.show()
sns.violinplot(y='Species', x='PetalWidthCm', data=data, inner='quartile')
plt.show()


# In[11]:


fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(16,5))
sns.scatterplot(x='SepalLengthCm',y='PetalLengthCm',data=data,hue='Species',ax=ax1,s=300,marker='o')
sns.scatterplot(x='SepalWidthCm',y='PetalWidthCm',data=data,hue='Species',ax=ax2,s=300,marker='o')


# ### Spliting the dataset into train and test dataset

# In[12]:


#Defining independent and dependent variables
features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = data.loc[:, features].values   #defining the feature matrix
y = data.Species


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=0)

#Defining the decision tree classifier and fitting the training set
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# ### Visualizing the decision tree

# In[14]:


from sklearn import tree
feature_name =  ['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)']
class_name= data.Species.unique()
plt.figure(figsize=(15,10))
tree.plot_tree(dtree, filled = True, feature_names = feature_name, class_names= class_name)


# ### Prediction on test data

# In[15]:


y_pred = dtree.predict(X_test)
y_pred
score=accuracy_score(y_test,y_pred)
print("Accuracy:",score)


# ### Plotting Confusion matrix

# In[16]:


def report(model):
    preds=model.predict(X_test)
    print(classification_report(preds,y_test))
    plot_confusion_matrix(model,X_test,y_test,cmap='nipy_spectral',colorbar=True)


# In[17]:


print('Decision Tree Classifier')
report(dtree)
print(f'Accuracy: {round(score*100,2)}%')


# In[18]:


confusion_matrix(y_test, y_pred)


# In[19]:


dtree.predict([[5, 3.6, 1.4 , 0.2]])


# In[20]:


dtree.predict([[9, 3.1, 5, 1.5]])


# In[21]:


dtree.predict([[4.1, 3.0, 5.1, 1.8]])


# In[ ]:




