#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ### Data Collection and Analysis

# In[2]:


# Loading the dataset
diabetes_dataset = pd.read_csv('data(Diabetes_prediction_project_#2)\diabetes.csv')


# In[3]:


diabetes_dataset.head()


# In[4]:


diabetes_dataset.shape


# In[5]:


diabetes_dataset.describe()


# In[6]:


diabetes_dataset['Outcome'].value_counts()


# ## 0-->Non-diabetic
# ## 1-->Diabetic

# In[7]:


diabetes_dataset.groupby('Outcome').mean()


# In[8]:


# Seperating the data and labels.
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[9]:


print(X)


# In[10]:


print(Y)


# ### Data Standardization

# In[11]:


scaler = StandardScaler()


# In[12]:


scaler.fit(X)


# In[13]:


standardized_data = scaler.transform(X)


# In[14]:


print(standardized_data)


# In[15]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# ### Choosing the best model by Cross Validation

# In[16]:


# Importing classification models.
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# In[17]:


# List of models.
models = [LogisticRegression(), svm.SVC(kernel='linear'), KNeighborsClassifier()]


# In[18]:


# Making a function for calculating accuracy of each model using cross validation.
def compare_models_cross_validation():
    
    for model in models:
        
        cv_score = cross_val_score(model, X, Y, cv=5)
        
        mean_accuracy = sum(cv_score)/len(cv_score)
        
        mean_accuracy = mean_accuracy*100
        
        mean_accuracy = round(mean_accuracy, 2)
        
        print('Cross Validation accuracies for', model, ': ', cv_score)
        
        print('Accuracy in percentage of', model, 'is: ', mean_accuracy)
        
        print('-----------------------------------------------------------')


# In[19]:


compare_models_cross_validation()


# ### Inference

# In[20]:


# In this case all the models have almost the same accuracy, but out of these four SVC is performing the best.


# ### Train-Test-Split

# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)


# In[22]:


print(X.shape, X_train.shape, X_test.shape)


# ### Training the model

# In[23]:


model = svm.SVC()

model.fit(X_train, Y_train)


# ### Evaluating the model

# #### Accuracy score

# In[24]:


# Accuracy on training data to check for overfitting
X_train_predictions = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predictions, Y_train)


# In[25]:


training_data_accuracy = round(training_data_accuracy*100)
print('Accuracy on training data in percentage is: ', training_data_accuracy, '%')


# In[26]:


# Accuracy on test data.
X_test_predictions = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predictions, Y_test)


# In[27]:


test_data_accuracy = round(test_data_accuracy*100)
print('Accuracy on test data in percentage is: ', test_data_accuracy, '%')


# ### Making a Predictive system

# In[29]:


# Define the feature names
feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# Function to get input data from the user
def get_input_data():
    print("Enter the values for the following features:")

    input_data = []
    for feature in feature_names:
        value = float(input(f"{feature}: "))
        input_data.append(value)
    
    # Convert the input into a numpy array and reshape it to (1, 8) for prediction
    return np.array(input_data).reshape(1, -1)

# Get user input
input_data = get_input_data()

# Standardizing the input data using the pre-fitted scaler
scaler.fit(input_data)
std_data = scaler.transform(input_data)

# Make a prediction using the model
prediction = model.predict(std_data)

# Display the result
if prediction[0] == 0:
    print('The person is not Diabetic')
else:
    print('The person is Diabetic')

