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


# It looks like the test accuracy is highest when using 3 and 5 neighbors. 
# Using 8 neighbors or more seems to result in a simple model that underfits the data. 

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


# =============================================================================
# Exercise
# 
# In this chapter, you will work with Gapminder data that we have consolidated 
# into one CSV file available in the workspace as 'gapminder.csv'. 
# Specifically, your goal will be to use this data to predict the life 
# expectancy in a given country based on features such as the country's GDP, 
# fertility rate, and population
# =============================================================================

#since you are going to use only one feature to begin with, you need to do some 
#reshaping using NumPy's .reshape()

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('data/gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

#Notice the differences in shape before and after applying the .reshape() method. 
#Getting the feature and target variable arrays into the right format for 
#scikit-learn is an important precursor to model building

# Exploring the Gapminder data

df.head()

df.info()

df.describe()

sns.heatmap(df.corr(), square=True, cmap='RdYlGn')

# Cells that are in green show positive correlation, while cells that are in 
# red show negative correlation

# ----------------------on boston data ----------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3, 
                                                    random_state=42)
reg_all = linear_model.LinearRegression()
reg_all.fit(X_train, y_train)

y_pred = reg_all.predict(X_test)
reg_all.score(X_test, y_test)
# ----------------------on boston data ----------------------------------------


# ----------------------on gapminder data -------------------------------------
# Now, you will fit a linear regression and predict life expectancy using just one feature.
# In this exercise, you will use the 'fertility' feature of the Gapminder dataset. 
# Since the goal is to predict life expectancy, the target variable here is 'life'

df = df.dropna()
# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Reshape X and y
y = y.reshape(-1,1)
X_fertility = X.reshape(-1,1)

plt.figure()
plt.scatter(X_fertility, y, color='blue')
plt.show()

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X_fertility, y))

# Plot regression line
plt.scatter(X_fertility, y, color = 'blue')
plt.plot(prediction_space, y_pred, color = 'black', linewidth=3)
plt.show()

# =============================================================================
# Train/test split for regression
# =============================================================================

#In this exercise, you will split the Gapminder dataset into training and testing
#sets, and then fit and predict a linear regression over all features. 
#
#In addition to computing the R2 score, you will also compute the Root Mean 
#Squared Error (RMSE), which is another commonly used metric to evaluate 
#regression models.

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/gapminder.csv').dropna()
y = df['life'].values
X = df.drop(['life','Country','region'], axis = 1).values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# =============================================================================
# Cross-validation
# =============================================================================

from sklearn.model_selection import cross_val_score
reg = linear_model.LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=5)
print(cv_results)
np.mean(cv_results) 

#5-fold cross-validation
#
#Cross-validation is a vital step in evaluating a model. It maximizes the amount 
#of data that is used to train the model, as during the course of training, 
#the model is not only trained, but also tested on all of the available data.

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv = 5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# -----------------------------------------------------------------------
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv = 3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv = 10)
print(np.mean(cvscores_10))

# you can use %timeit to see how long each 3-fold CV takes compared to 10-fold 
# CV by executing the following 

%timeit cross_val_score(reg, X, y, cv = 3)
%timeit cross_val_score(reg, X, y, cv = 10)

# =============================================================================
#  Regularized regression
# =============================================================================

# Boston data 
boston = pd.read_csv('data/boston-housing/train.csv')

print(boston.head()) 

# Creating feature and target arrays
X = boston.drop('medv', axis=1).values
y = boston['medv'].values

# Ridge regression in scikit-learn

from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3, 
                                                    random_state=42)

ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)

# Lasso regression in scikit-learn

from sklearn.linear_model import Lasso

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3, 
                                                    random_state=42)
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)

#Lasso regression for feature selection
#● Can be used to select important features of a dataset
#● Shrinks the coefficients of less important features to exactly 0

from sklearn.linear_model import Lasso

names = boston.drop('medv', axis=1).columns

lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_

_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()

# =============================================================================
# In this exercise, you will fit a lasso regression to the Gapminder data you 
# have been working with and plot the coefficients.
# =============================================================================

# Read the CSV file into a DataFrame: df
df = pd.read_csv('data/gapminder.csv')

df = df.dropna()

# Create arrays for features and target variable
y = df['life'].values
X = df.drop(['life','Country','region'], axis =1).values

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

df_columns = df.drop(['life','Country','region'], axis = 1).columns

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()

# =============================================================================
# Regularization II: Ridge
# Lasso is great for feature selection, but when building regression models, 
# Ridge regression should be your first choice.
# 
# Recall that lasso performs regularization by adding to the loss function a 
# penalty term of the absolute value of each coefficient multiplied by some alpha. 
# This is also known as L1 regularization because the regularization term is the 
# L1 norm of the coefficients. This is not the only way to regularize, however.
# 
# If instead you took the sum of the squared values of the coefficients 
# multiplied by some alpha - like in Ridge regression - you would be computing the L2 norm. 
# 
# In this exercise, you will practice fitting ridge regression models over a 
# range of different alphas, and plot cross-validated R2 scores for each, 
# using this function that we have defined for you, which plots the R2 score as 
# well as standard error for each alpha
# =============================================================================

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
    
    # Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)

# Notice how the cross-validation scores change with different alphas


# =============================================================================
# How good is your model?
# =============================================================================

#Classification metrics
#● Measuring model performance with accuracy:
#● Fraction of correctly classified samples
#● Not always a useful metric

#Class imbalance example: Emails
#● Spam classification
#● 99% of emails are real; 1% of emails are spam
#● Could build a classifier that predicts ALL emails as real
#● 99% accurate!
#● But horrible at actually classifying spam
#● Fails at its original purpose 

# =============================================================================
# Confusion matrix in scikit-learn
# =============================================================================

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

knn = KNeighborsClassifier(n_neighbors=8)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.4, 
                                                    random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#● High precision: Not many real emails predicted as spam
#● High recall: Predicted most spam emails correctly 

# =============================================================================
# Here, you'll work with the PIMA Indians dataset obtained from the UCI Machine 
# 
# Learning Repository. The goal is to predict whether or not a given female patient 
# will contract diabetes based on features such as BMI, age, and number of pregnancies. 
# Therefore, it is a binary classification problem. A target value of 0 indicates
#  that the patient does not have diabetes, while a value of 1 indicates that 
#  the patient does have diabetes
# =============================================================================


df = pd.read_csv('data/diabetes.csv')

df.head()
df.info()
df.columns

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

y = df['Outcome'].values
X = df.drop('Outcome', axis = 1).values

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size =0.4,random_state = 42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors = 6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =============================================================================
# Logistic regression and the ROC curve
# =============================================================================
#Logistic regression for binary classification
#● Logistic regression outputs probabilities
#● If the probability ‘p’ is greater than 0.5:
#● The data is labeled ‘1’
#● If the probability ‘p’ is less than 0.5:
#● The data is labeled ‘0’ 

# Logistic regression in scikit-learn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.4, 
                                                    random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

#Probability thresholds
#● By default, logistic regression threshold = 0.5
#● Not specific to logistic regression
#● k-NN classifiers also have thresholds
#● What happens if we vary the threshold?

#Plotting the ROC curve

from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]
# we choose the second column, ie the probability of the outcome being 1. 

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# fpr - false positive rate, tpr - true postive rate and thresholds
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate’)
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show();

# -----------------------------------------------------------------------------

# =============================================================================
# Building a logistic regression model
# 
# Time to build your first logistic regression model!
# =============================================================================

# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =============================================================================
# Plotting an ROC curve
# =============================================================================

#Classification reports and confusion matrices are great methods to 
#quantitatively evaluate model performance, while ROC curves provide a way 
#to visually evaluate models
#
#most classifiers in scikit-learn have a .predict_proba() method which returns 
#the probability of a given sample being in a particular class

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# =============================================================================
# Precision-recall Curve
# 
# When looking at your ROC curve, you may have noticed that the y-axis 
# (True positive rate) is also known as recall. Indeed, in addition to the ROC 
# curve, there are other ways to visually evaluate model performance. One such
#  way is the precision-recall curve, which is generated by plotting the 
#  precision and recall for different thresholds. As a reminder, precision 
#  and recall are defined as:
# 
# Precision=TP/(TP+FP)
# Recall=TP/(TP+FN)
# 
# A recall of 1 corresponds to a classifier with a low threshold in which all 
# females who contract diabetes were correctly classified as such, at the expense
# of many misclassifications of those who did not have diabetes.
#  
# Precision is undefined for a classifier which makes no positive predictions, 
# that is, classifies everyone as not having diabetes.
#  
# When the threshold is very close to 1, precision is also 1, because the 
# classifier is absolutely certain about its predictions.
# 
# =============================================================================

# =============================================================================
# Area under the ROC curve
# =============================================================================

df = pd.read_csv('data/diabetes.csv')

y = df['Outcome'].values
X = df.drop('Outcome', axis = 1).values

# AUC in scikit-learn

from sklearn.metrics import roc_auc_score

logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                    random_state=42)

logreg.fit(X_train, y_train)

y_pred_prob = logreg.predict_proba(X_test)[:,1]

roc_auc_score(y_test, y_pred_prob)

# AUC using cross-validation

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(logreg, X, y, cv=5, 
                            scoring='roc_auc')

print(cv_scores)  


# =============================================================================
# # AUC computation
# # 
# # Say you have a binary classifier that in fact is just randomly making guesses. 
# # It would be correct approximately 50% of the time, and the resulting ROC curve 
# # would be a diagonal line in which the True Positive Rate and False Positive Rate 
# # are always equal. The Area under this ROC curve would be 0.5
# =============================================================================

# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg,X,y, cv = 5, scoring= 'roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

# =============================================================================
# Hyperparameter tuning
# 
# ● Linear regression: Choosing parameters
# ● Ridge/lasso regression: Choosing alpha
# ● k-Nearest Neighbors: Choosing n_neighbors
# ● Parameters like alpha and k: Hyperparameters
# ● Hyperparameters cannot be learned by fitting the model
# 
# Choosing the correct hyperparameter
# ● Try a bunch of different hyperparameter values
# ● Fit all of them separately
# ● See how well each performs
# ● Choose the best performing one
# ● It is essential to use cross-validation
# =============================================================================

# GridSearchCV in scikit-learn

from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': np.arange(1, 50)}

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y) 

knn_cv.best_params_
knn_cv.best_score_

#Like the alpha parameter of lasso and ridge regularization that you saw earlier, 
#logistic regression also has a regularization parameter: C. C controls the 
#inverse of the regularization strength, and this is what you will tune in 
#this exercise. A large C can lead to an overfit model, while a small C 
#can lead to an underfit model.

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv= 5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))


# =============================================================================
# Hyperparameter tuning with RandomizedSearchCV
# =============================================================================

#GridSearchCV can be computationally expensive, especially if you are searching 
#over a large hyperparameter space and dealing with multiple hyperparameters. 
#A solution to this is to use RandomizedSearchCV, in which not all hyperparameter 
#values are tried out. Instead, a fixed number of hyperparameter settings is 
#sampled from specified probability distributions. 
#
#You'll practice using RandomizedSearchCV in this exercise and see how this works.
#
#Here, you'll also be introduced to a new model: the Decision Tree. 
#Don't worry about the specifics of how this model works. 
#
#Just like k-NN, linear regression, and logistic regression, decision 
#trees in scikit-learn have .fit() and .predict() methods that you can use 
#in exactly the same way as before. Decision trees have many parameters that 
#can be tuned, such as max_features, max_depth, and min_samples_leaf: 
#This makes it an ideal use case for RandomizedSearchCV


# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

#Note that RandomizedSearchCV will never outperform GridSearchCV. 
#Instead, it is valuable because it saves on computation time.


# =============================================================================
# Hold-out set in practice I: Classification
# =============================================================================

#You will now practice evaluating a model with tuned hyperparameters on a hold-out set. 
#
#In addition to C, logistic regression has a 'penalty' hyperparameter which 
#specifies whether to use 'l1' or 'l2' regularization. Your job in this 
#exercise is to create a hold-out set, tune the 'C' and 'penalty' hyperparameters 
#of a logistic regression classifier using GridSearchCV on the training set

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.4, 
                                                    random_state = 42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv= 5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

# =============================================================================
# Hold-out set in practice II: Regression
# =============================================================================

#Remember lasso and ridge regression from the previous chapter? Lasso used the 
#L1 penalty to regularize, while ridge used the L2 penalty. There is another type 
#of regularized regression known as the elastic net. In elastic net regularization, 
#the penalty term is a linear combination of the L1 and L2 penalties:
#
#a∗L1+b∗L2
#
#In scikit-learn, this term is represented by the 'l1_ratio' 
#parameter: An 'l1_ratio' of 1 corresponds to an L1 penalty, and anything 
#lower is a combination of L1 and L2.


# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))





























