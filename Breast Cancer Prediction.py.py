#!/usr/bin/env python
# coding: utf-8

# # ***Importing packages and libraries :***

# In[54]:


# import libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_columns = 100


# # ***Reading the dataset :***

# ### Load the dataset :

# In[55]:


data= pd.read_csv("data.csv")


# In[56]:


data  # ****** Printing the dataset *****


# ### Printing first Five Records/Rows :

# In[57]:


data.head()   # ****** Printing the first five rows/records in the dataset ******


# ### Printing last five Records/Rows :

# In[58]:


data.tail()   # ****** Printing the last five rows/records in the dataset *******


# # ***Exploratory Data Anaysis :***

# ### Analyzing Number of rows and columns :

# In[59]:


# ***** Basic information of number of rows and columns *****

data.shape


# ### More Information about the dataframes :

# In[60]:


# ***** Info on DataFrame *****

data.info()  


# ### Statistical Summary of the Dataframe :

# In[61]:


# ***** Statistical summary of data frame.*****

data.describe()


# ### Checking the datatypes :
# 

# In[62]:


data.dtypes    #  Note : Except 'diagnosis' all the columns are numeric


# ### Check for null values in dataset :

# In[63]:


data.isnull().values.any()


# In[64]:


data.isna().any()


# In[65]:


data.isna().sum() 


# ### Number of Non-NA values :
# 

# In[66]:


data.count()


# ### Accessing the columns names :

# In[67]:


data.columns


# In[68]:


pd.set_option("display.float", "{:.2f}".format)
data.describe()


# ### Count class labels :

# In[69]:


data['diagnosis'].value_counts()


# In[70]:


diagnosis_unique = data.diagnosis.unique()


# In[71]:


diagnosis_unique 


# In[72]:


data = data.dropna(axis='columns')


# In[73]:


data.describe(include="O")


# # ***Data Visualization :***
# 

# ### Importing data visualizing packages and libraries

# In[74]:


pip install plotly


# In[79]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[76]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis,color='red')
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")
plt.subplot(1, 2, 2)
sns.countplot('diagnosis', data=data);


# In[92]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# # ***Data Filtering :***

# In[82]:


from sklearn.preprocessing import LabelEncoder


# In[83]:


data.head(2)


# ### Normalizing the labels :

# In[86]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)


# In[87]:


data.head(2)


# ### Conversion of categorical values to 0 and 1 :

# In[88]:


print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


# # ***Correlation Matrix :***
# 

# In[89]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[90]:


plt.figure(figsize=(12, 12))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap='inferno');


# # ***Model Implementation :***

# ### Train Test Splitting :

# ### _Preprocessing and model selection:_

# In[37]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# ### Importing machine learning models :

# In[38]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# ### Check the Model Accuracy, Errors and it's Validations :

# In[39]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# ### Select feature for predictions :
# 

# In[40]:


data.columns


# ### Take the dependent and independent feature for prediction :

# In[41]:


prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[42]:


X = data[prediction_feature]
X

# print(X.shape)
# print(X.values)


# In[43]:


y = data.diagnosis
y

# print(y.values)


# ### Splite the dataset into TrainingSet and TestingSet by 33% and set the 15 fixed records :

# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=15)

print(X_train)
# print(X_test)


# ### Perform Feature Standard Scalling :

# In[45]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ### ML Model Selecting and Model Prediction

# In[46]:


from sklearn.metrics import accuracy_score,confusion_matrix


# ### Random Forest :

# In[47]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=50)
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)
print(accuracy_score(y_test,y_pred_rf))
confusion_matrix(y_test,y_pred_rf)


# ### Logistic Regression :
# 

# In[48]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=1,solver='liblinear')
lr.fit(X_train,y_train)
y_pred_lr=lr.predict(X_test)
print(accuracy_score(y_test,y_pred_lr))
confusion_matrix(y_test,y_pred_lr)


# ### KNN :

# In[49]:


from sklearn.neighbors import KNeighborsClassifier
Ks=20
accuracy=[]
for k in range(1,Ks):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    y_pred_knn = knn_classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred_knn))
plt.plot(accuracy)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()


# In[50]:


knn_classifier = KNeighborsClassifier(n_neighbors = 6)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_score(y_test, y_pred_knn)


# ## Thus we can see that _Logistic Regression_  is giving best accuracy of 90.95%

# In[51]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


# In[52]:


input_data=(4,3,2,1,3,4)

input_data_narray= np.asarray(input_data)

input_data_reshape = input_data_narray.reshape(1,-1)

prediction = model.predict(input_data_reshape)
print(prediction)

if(prediction[0] == 0):
    print("The Person has Malignant Cancer !")
elif(prediction[0]==1):
    print("Person has Benign Cancer !")
else:
    print('Person has No breast Cancer!')


# ### Credits :
# #### Created by -> Hozefa  Arielwala
