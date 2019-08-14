#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 09:24:43 2019

@author: amin
"""

# =============================================================================
# Classication-tree
# 
# Sequence of if-else questions about individual features.
# 
# Objective: infer class labels.
# Able to capture non-linear relationships between features and labels.
# Don't require feature scaling (ex: Standardization, ..)
# =============================================================================

import pandas as pd

bc = pd.read_csv('data/breast-cancer-data.csv', index_col = 'id')

# The maximum number of branches separating the top from an extreme end is known
# as the maximum depth. 

# Classication-tree in scikit-learn

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score

X = bc.drop('diagnosis', axis = 1)
#y = bc['diagnosis'].replace('M', 1).replace('B', 0)

y = bc['diagnosis']
# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,
                                                   test_size=0.2,
                                                   stratify=y,
                                                   random_state=1)

# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# Fit dt to the training set
dt.fit(X_train,y_train)

# Predict test set labels
y_pred = dt.predict(X_test)
# Evaluate test-set accuracy
accuracy_score(y_test, y_pred)

# -----------------------------------------------------------------------------
# =============================================================================
# Train your first classification tree
# 
# You'll predict whether a tumor is malignant or benign based on two features: the 
# mean radius of the tumor (radius_mean) and its mean number of concave points 
# (concave points_mean).
# =============================================================================

# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

SEED = 1

X = bc[['radius_mean', 'concave points_mean']]

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,
                                                   test_size=0.2,
                                                   stratify=y,
                                                   random_state=1)

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = DecisionTreeClassifier(max_depth=6, random_state=SEED)

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)
print(y_pred[0:5])

# Evaluate the classification tree

# Import accuracy_score
from sklearn.metrics import accuracy_score

# Predict test set labels
y_pred = dt.predict(X_test)

# Compute test set accuracy  
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))

# =============================================================================
# Logistic regression vs classification tree
# 
# A classification tree divides the feature space into rectangular regions. 
# In contrast, a linear model such as logistic regression produces only a single 
# linear decision boundary dividing the feature space into two decision regions.
# =============================================================================

# Get function source
import inspect
print(inspect.getsource(accuracy_score))

# Import LogisticRegression from sklearn.linear_model
from sklearn.linear_model import  LogisticRegression

# Instatiate logreg
logreg = LogisticRegression(random_state=1)

# Fit logreg to the training set
logreg.fit(X_train, y_train)

# Define a list called clfs containing the two classifiers logreg and dt
clfs = [logreg, dt]

# Review the decision regions of the two classifiers
plot_labeled_decision_regions(X_test, y_test, clfs)

#Notice how the decision boundary produced by logistic regression is linear while 
#the boundaries produced by the classification tree divide the feature space 
#into rectangular regions.

# =============================================================================
# Building Blocks of a Decision-Tree
# 
# Decision-Tree: data structure consisting of a hierarchy of nodes.
# Node: question or prediction.
# 
# Three kinds of nodes:
# 1. Root: no parent node, question giving rise to two children nodes.
# 2. Internal node: one parent node, question giving rise to two children nodes.
# 3. Leaf: one parent node, no children nodes --> prediction.
# 
# Information Gain (IG)
# 
# Criteria to measure the impurity of a node I(node):
#     gini index,
#     entropy

# Nodes are grown recursively.
# At each node, split the data based on:
# feature f and split-point sp to maximize IG(node).
# If IG(node)= 0, declare the node a leaf

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,
                                                   test_size=0.2,
                                                   stratify=y,
                                                   random_state=1)

# Instantiate dt, set 'criterion' to 'gini'
dt_gini = DecisionTreeClassifier(max_depth = 8, 
                                 criterion='gini', 
                                 random_state=1)

#Information Criterion in scikit-learn

# Fit dt to the training set
dt_gini.fit(X_train,y_train)

# Predict test-set labels
y_pred_gini= dt_gini.predict(X_test)

# Evaluate test-set accuracy
accuracy_gini = accuracy_score(y_test, y_pred_gini)

# =============================================================================
# Using entropy as a criterion
# =============================================================================

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(max_depth=8, 
                                    criterion='entropy', 
                                    random_state=1)

# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)

# Use dt_entropy to predict test set labels
y_pred_entropy = dt_entropy.predict(X_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)

# Print accuracy_entropy
print('Accuracy achieved by using entropy: ', accuracy_entropy)

# Print accuracy_gini
print('Accuracy achieved by using the gini index: ', accuracy_gini)


# The gini index is slightly faster to compute and is the default criterion 
# used in the DecisionTreeClassifier model of scikit-learn.

# =============================================================================
# Decision tree for regression
# =============================================================================

auto = pd.read_csv('data/auto-mpg.csv', na_values = '?').dropna()
X = auto.drop(['car name', 'mpg'], axis = 1)
y = auto['mpg']

# Import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,
                                                   test_size=0.2,
                                                   random_state=3)

# Instantiate a DecisionTreeRegressor 'dt'
dt = DecisionTreeRegressor(max_depth=4,
                           min_samples_leaf=0.1,
                           random_state=3)

# Fit 'dt' to the training-set
dt.fit(X_train, y_train)

# Predict test-set labels
y_pred = dt.predict(X_test)

# Compute test-set MSE
mse_dt = MSE(y_test, y_pred)

# Compute test-set RMSE
rmse_dt = mse_dt**(1/2)

# Print rmse_dt
print(rmse_dt)

# -----------------------------------------------------------------------------
# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor

# Instantiate dt
dt = DecisionTreeRegressor(max_depth= 8,
             min_samples_leaf = 0.13,
            random_state=3)

# Fit dt to the training set
dt.fit(X_train, y_train)

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute y_pred
y_pred = dt.predict(X_test)

# Compute mse_dt
mse_dt = MSE(y_test, y_pred)

# Compute rmse_dt
rmse_dt = mse_dt **(1/2)

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))

# =============================================================================
# Linear regression vs regression tree
# =============================================================================

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict test set labels 
y_pred_lr = lr.predict(X_test)

# Compute mse_lr
mse_lr = MSE(y_test, y_pred_lr)

# Compute rmse_lr
rmse_lr = mse_lr**(1/2)

# Print rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))

# Print rmse_dt
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))

# =============================================================================
# Generalization Error
# =============================================================================

#Supervised Learning - Under the Hood
#Supervised Learning: y = f(x), f is unknown.
#
#Goals of Supervised Learning
#1. Find a model f hat that best approximates f: f hat ≈ f
#2. f hat can be LogisticRegression, Decision Tree, Neural Network ...
#3. Discard noise as much as possible.
#4. End goal: f hat should acheive a low predictive error on unseen datasets.
#
#Difculties in Approximating f
#1. Overtting: f hat(x) ts the training set noise.
#2. Undertting: f hat is not exible enough to approximate f
#
#Generalization Error
#Generalization Error of f hat : Does f hat generalize well on unseen data?
#It can be decomposed as follows: Generalization Error of
#
#f hat = bias^2 + variance + irreducible error
#
#Variance
#Variance: tells you how much f hat is inconsistent over different training sets.
#
#Bias
#Bias: error term that tells you, on average, how much f hat ≠ f
#
#Model Complexity
#
#ModelComplexity: sets the exibility of .
#Example: Maximum tree depth, Minimum samples per leaf, ..

# =============================================================================
# Estimating the Generalization Error
# =============================================================================
How do we estimate the generalization error of a model?

Cannot be done directly because:
1. f is unknown,
2. usually you only have one dataset,
3. noise is unpredictable

Solution:
split the data to training and test sets,
fit f hat to the training set,
evaluate the error of on the unseen test set.

generalization error of f hat ≈ test set error of f hat.


# =============================================================================
# Better Model Evaluation with Cross-Validation
# =============================================================================

#Test set should not be touched until we are condent about f hat's performance.
#Evaluating on training set: biased estimate, has already seen alltraining points.
#Solution→Cross-Validation (CV):
#1. K-Fold CV,
#2. Hold-Out CV
#
#Diagnose Variance Problems
#If suffers from high variance: CV error of f hat > training set error of f hat. 
#f hat is said to overt the training set. To remedy overtting:
#1. decrease model complexity,
#i. for ex: decrease max depth, increase min samples per leaf, ...
#ii. gather more data

#As the complexity of f̂  increases, the bias term decreases while the variance 
#term increases

# Diagnose bias and variance problems


#K-Fold CV in sklearn on the Auto Dataset

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

# Set seed for reproducibility
SEED = 123

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=SEED)

# Instantiate decision tree regressor and assign it to 'dt'
dt = DecisionTreeRegressor(max_depth=4,
                           min_samples_leaf= 0.14 ,
                           random_state=SEED)

# Evaluate the list of MSE ontained by 10-fold CV
# Set n_jobs to -1 in order to exploit all CPU cores in computation

MSE_CV = - cross_val_score(dt, X_train, y_train, 
                           cv= 10,
                           scoring= 'neg_mean_squared_error',
                           n_jobs = -1)

# Fit 'dt' to the training set
dt.fit(X_train, y_train)

# Predict the labels of training set
y_predict_train = dt.predict(X_train)

# Predict the labels of test set
y_predict_test = dt.predict(X_test)

# CV MSE
print('CV MSE: {:.2f}'.format(MSE_CV.mean()))

# Training set MSE
print('Train MSE: {:.2f}'.format(MSE(y_train, y_predict_train)))

# Test set MSE
print('Test MSE: {:.2f}'.format(MSE(y_test, y_predict_test)))

#Given that the training set error is smaller than the CV error, we can deduce
#that dt overfits the training set and it suffers from high variance 
#
#Notice how the CV and the test errors are close

# -----------------------------------------------------------------------------
#In the following set of exercises, you'll diagnose the bias and variance problems 
#of a regression tree. The regression tree you'll define in this exercise will 
#be used to predict the mpg consumption of cars from the auto dataset using all 
#available features.

# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Set SEED for reproducibility
SEED = 123

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth =4, 
                           min_samples_leaf = 0.26, 
                           random_state=SEED)

# Evaluate the 10-fold CV error
# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10, 
                       scoring ='neg_mean_squared_error',
                       n_jobs=-1)

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())**(1/2)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))


# Evaluate the training error
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict the labels of the training set
y_pred_train = dt.predict(X_train)

# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train))**(1/2)

# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))

# =============================================================================
# baseline_RMSE
# Out[1]: 5.1
# 
# In [2]: RMSE_CV
# Out[2]: 5.14
# 
# In [3]: RMSE_train
# Out[3]: 5.15
# 
# the model suffers from high bias because RMSE_CV ≈ RMSE_train and both 
# scores are greater than baseline_RMSE
# 
# The model is indeed underfitting the training set as the model is too constrained to 
# capture the nonlinear dependencies between features and labels.
# =============================================================================

#Advantages of CARTs
#Simple to understand.
#Simple to interpret.
#Easy to use.
#Flexibility: ability to describe non-linear dependencies.
#Preprocessing: no need to standardize or normalize features
#
#Limitations of CARTs
#Classication: can only produce orthogonal decision boundaries.
#Sensitive to small variations in the training set.
#High variance: unconstrained CARTs may overt the training set.
#Solution: ensemble learning.

# =============================================================================
# Ensemble Learning
# =============================================================================

#* Train different models on the same dataset.
#* Let each model make its predictions.
#* Meta-model: aggregates predictions ofindividual models.
#* Final prediction: more robust and less prone to errors.
#* Best results: models are skillful in different ways.


# Import functions to compute accuracy and split data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Import models, including VotingClassifier meta-model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier

# Set seed for reproducibility
SEED = 1

#Voting Classier in sklearn (Breast-Cancer dataset)

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size= 0.3,
                                                    random_state= SEED)
# Instantiate individual classifiers
lr = LogisticRegression(random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)

# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr),
               ('K Nearest Neighbours', knn),
               ('Classification Tree', dt)]

# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
    #fit clf to the training set
    clf.fit(X_train, y_train)
    # Predict the labels of the test set
    y_pred = clf.predict(X_test)
    # Evaluate the accuracy of clf on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))

# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers)
# Fit 'vc' to the traing set and predict test set labels
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)

# Evaluate the test-set accuracy of 'vc'
print('Voting Classifier: {.3f}'.format(accuracy_score(y_test, y_pred)))

# -----------------------------------------------------------------------------
ind = pd.read_csv('data/Indian Liver Patient Dataset (ILPD).csv')

ind_origin = pd.get_dummies(ind)

X = ind_origin.drop('is_patient', axis = 1)
y = ind_origin['is_patient']

# Imputer missing values 
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0) # Axis = 0, across columns, 1 mean rows

imp.fit(X)

X = imp.transform(X)

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size= 0.3,
                                                    random_state= SEED)

# Set seed for reproducibility
SEED=1

# Instantiate lr
lr = LogisticRegression(random_state=SEED)

# Instantiate knn
knn = KNN(n_neighbors= 27)

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf= 0.13, random_state=SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), 
               ('K Nearest Neighbours', knn), 
               ('Classification Tree', dt)]

# Evaluate individual classifiers
# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:    
 
    # Fit clf to the training set
    clf.fit(X_train, y_train)    
   
    # Predict y_pred
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) 
   
    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))

# Better performance with a Voting Classifier
# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier

# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)     

# Fit vc to the training set
vc.fit(X_train, y_train)   

# Evaluate the test set predictions
y_pred = vc.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Voting Classifier: {:.3f}'.format(accuracy))

# =============================================================================
# Bagging - Bootstrap aggregation
# =============================================================================








