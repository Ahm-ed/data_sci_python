#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:09:40 2019

@author: amin
"""

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
plt.style.use('ggplot')
import seaborn as sns

# =============================================================================
# The Iris dataset in scikit-learn
# =============================================================================

iris = datasets.load_iris()
type(iris)

print(iris.keys()) 

type(iris.data), type(iris.target)

iris.data.shape

iris.target_names 

# =============================================================================
# Exploratory data analysis (EDA)
# =============================================================================
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)

print(df.head()) 

_ = scatter_matrix(df, c = y, figsize = [8, 8], s=150, marker = 'D') 

# More EDA - This is better for binary data
plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

# Be sure to begin your plotting statements for each figure with plt.figure() 
# so that a new figure will be set up. Otherwise, your plots will be overlayed onto the same figure.


# =============================================================================
# k-Nearest Neighbors
# =============================================================================


# =============================================================================
# Scikit-learn fit and predict
# ● All machine learning models implemented as Python classes
# ● They implement the algorithms for learning and predicting
# ● Store the information learned from the data
# ● Training a model on the data = ‘fi"ing’ a model to the data
# ● .fit() method
# ● To predict the labels of new data: .predict() method
# =============================================================================

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target']) # Data must be numpy array or pandas dataframe and target must be numpy array

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params= None, n_jobs=1,
                     n_neighbors=6, p=2,weights='uniform')

iris['data'].shape

iris['target'].shape

#Predicting on unlabeled data

prediction = knn.predict(X_new) # must be numpy array
X_new.shape

print('Prediction {}’.format(prediction))

# ===========================================================================
names = ['party',
         'handicapped-infants',
         'water-project-cost-sharing',
         'adoption-of-the-budget-resolution',
         'physician-fee-freeze',
         'el-salvador-aid',
         'religious-groups-in-schools',
         'anti-satellite-test-ban',
         'aid-to-nicaraguan-contras',
         'mx-missile',
         'immigration',
         'synfuels-corporation-cutback',
         'education-spending',
         'superfund-right-to-sue',
         'crime',
         'duty-free-exports',
         'export-administration-act-south-africa']

df = pd.read_csv('data/house-votes-84.data', names = names, na_values = '?')

# EDA
plt.figure()
sns.countplot(x='immigration', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors = 6)

# Fit the classifier to the data
knn.fit(X, y)

# =============================================================================
# k-Nearest Neighbors: Predict
# 
# Having fit a k-NN classifier, you can now use it to predict the label of a 
# new data point. However, there is no unlabeled data available since all of it 
# was used to fit the model! You can still use the .predict() method on the X 
# that was used to fit the model, but it is not a good indicator of the model's 
# ability to generalize to new, unseen data.
# 
# In the next video, Hugo will discuss a solution to this problem. For now, a 
# random unlabeled data point has been generated and is available to you as X_new.
#  You will use your classifier to predict the label for this new data point, as 
#  well as on the training data X that the model has already seen. 
# Using .predict() on X_new will generate 1 prediction, while using it on X
#  will generate 435 predictions: 1 for each sample.
# =============================================================================

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis = 1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors = 6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))


# =============================================================================

# =============================================================================
# Measuring model performance
# =============================================================================
#Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=21, stratify=y)

knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

knn.score(X_test, y_test) 

#Model complexity
#● Larger k = smoother decision boundary = less complex model
#● Smaller k = more complex model = can lead to overfitting

# =============================================================================
# The digits recognition dataset
# =============================================================================

# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

#Train/Test Split + Fit/Predict/Accuracy
#
#Now that you have learned about the importance of splitting your data into 
#training and test sets, it's time to practice doing this on the digits dataset! 
#
#After creating arrays for the features and target variable, you will split them
#into training and test sets, fit a k-NN classifier to the training data, and 
#then compute its accuracy using the .score() method.

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify= y)

#Stratify the split according to the labels so that they are distributed in the 
#training and test sets as they are in the original dataset.

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors = 7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

# =============================================================================
# Overfitting and underfitting
# =============================================================================

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


#It looks like the test accuracy is highest when using 3 and 5 neighbors. 
#Using 8 neighbors or more seems to result in a simple model that underfits the data. 

# =============================================================================
# Introduction to regression
# =============================================================================

# Boston housing data

boston = pd.read_csv('data/boston-housing/train.csv')

print(boston.head()) 


# Creating feature and target arrays

X = boston.drop('medv', axis=1).values
y = boston['medv'].values

# Predicting house value from a single feature

X_rooms = X[:,6]
type(X_rooms), type(y)

y = y.reshape(-1, 1)

X_rooms = X_rooms.reshape(-1, 1)

plt.scatter(X_rooms, y)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show();

# Fitting a regression model

import numpy as np
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_rooms, y)
prediction_space = np.linspace(min(X_rooms), 
                               max(X_rooms)).reshape(-1, 1)

plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space),
         color='black', linewidth=3)
plt.show()





















