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











