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

# The maximum number of branches separating the top from an extreme end is known
# as the maximum depth. 

# Classication-tree in scikit-learn

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score
import pandas as pd

bc = pd.read_csv('data/breast-cancer-data.csv', index_col = 'id')

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
#How do we estimate the generalization error of a model?
#
#Cannot be done directly because:
#1. f is unknown,
#2. usually you only have one dataset,
#3. noise is unpredictable
#
#Solution:
#split the data to training and test sets,
#fit f hat to the training set,
#evaluate the error of on the unseen test set.
#
#generalization error of f hat ≈ test set error of f hat.


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

#Bagging
#
#Bagging: Bootstrap Aggregation.
#
#Uses a technique known as the bootstrap.
#Reduces variance ofindividual models in the ensemble
#
#Bagging: Classication & Regression
#
#Classication:
#    Aggregates predictions by majority voting.
#    BaggingClassifier in scikit-learn.
#
#Regression:
#    Aggregates predictions through averaging.
#    BaggingRegressor in scikit-learn.

#Bagging Classier in sklearn (Breast-Cancer dataset)

import pandas as pd

bc = pd.read_csv('data/breast-cancer-data.csv', index_col = 'id')
X = bc.drop('diagnosis', axis =1)
y = bc['diagnosis']

# Import models and utility functions
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    stratify = y,
                                                    random_state = SEED)

# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4, 
                            min_samples_leaf=0.16, 
                            random_state=SEED)

# Instantiate a BaggingClassifier 'bc'
bc = BaggingClassifier(base_estimator=dt, 
                       n_estimators=300, 
                       oob_score = True,
                       n_jobs=-1)

# Fit 'bc' to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))

# Extract the OOB accuracy from 'bc'
oob_accuracy = bc.oob_score_

# Print test set accuracy
print('OOB accuracy: {:.3f}'.format(oob_accuracy))

#Note that in scikit-learn OOB score corresponds to accuracy for classifiers and 
#R squared for regression.

# ----------------------INDIAN LIVER DATA--------------------------------------

ind = pd.read_csv('data/Indian Liver Patient Dataset (ILPD).csv').dropna()
ind = pd.get_dummies(ind, drop_first = True)

ind.is_patient = ind.is_patient.astype('category')  

X = ind.drop('is_patient', axis = 1)
y = ind['is_patient']

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    stratify = y,
                                                    random_state = SEED)

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf = 8, 
                            random_state=1)
# The base estimator can be any model including logistic to neural network

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, 
                       n_estimators=50, 
                       random_state=1,
                       oob_score = True,
                       n_jobs = -1)

# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# =============================================================================
# OOB Score vs Test Set Score
# =============================================================================

# Fit bc to the training set 
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
acc_test = accuracy_score(y_test, y_pred)

# Evaluate OOB accuracy
acc_oob = bc.oob_score_

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))

# =============================================================================
# Random Forests (RF)
# =============================================================================

#Bagging
#Base estimator: Decision Tree, LogisticRegression, Neural Net, ...
#Each estimator is trained on a distinct bootstrap sample ofthe training set
#Estimators use all features for training and prediction
#
#Further Diversity with Random Forests
#
#Base estimator: Decision Tree
#Each estimator is trained on a different bootstrap sample having the same size as the training set
#RF introduces further randomization in the training ofindividualtrees
#d features are sampled at each node without replacement (d < total number of features)
#
#Random Forests: Classication & Regression
#
#Classication:
#    Aggregates predictions by majority voting
#    RandomForestClassifier in scikit-learn
#
#Regression:
#    Aggregates predictions through averaging
#    RandomForestRegressor in scikit-learn

# Random Forests Regressor in sklearn (auto dataset)

auto = pd.read_csv('data/auto-mpg.csv', na_values = '?').dropna()
X = auto.drop(['car name', 'mpg'], axis = 1)
y = auto['mpg']

# Basic imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# Set seed for reproducibility
SEED = 1

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=SEED)

# Instantiate a random forests regressor 'rf' 400 estimators
rf = RandomForestRegressor(n_estimators=400,
                           min_samples_leaf=0.12,
                           random_state=SEED)

# Fit 'rf' to the training set
rf.fit(X_train, y_train)

# Predict the test set labels 'y_pred'
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)

# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

#Feature Importance
#
#Tree-based methods: enable measuring the importance of each feature in prediction.
#
#In sklearn :
#    how much the tree nodes use a particular feature (weighted average) to reduce impurity
#    accessed using the attribute feature_importance_

import pandas as pd
import matplotlib.pyplot as plt

# Create a pd.Series of features importances
importances_rf = pd.Series(rf.feature_importances_, index = X.columns)

# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()

# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen')
plt.show()

# -----------------------bike_share_train--------------------------------------

#In the following exercises you'll predict bike rental demand in the Capital 
#Bikeshare program in Washington, D.C using historical weather data from the 
#Bike Sharing Demand dataset available through Kaggle.

# Basic imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

bike = pd.read_csv('data/bike_share_train.csv', parse_dates = True, 
                   index_col = 'datetime')

X = bike.drop(['count'], axis = 1)
y = bike['count']

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=SEED)

# Instantiate rf
rf = RandomForestRegressor(n_estimators=25,
                           random_state=2)
           
# Fit rf to the training set    
rf.fit(X_train, y_train) 

# Predict the test set labels
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

#Visualizing features importances

# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()


# =============================================================================
# Boosting
# 
# Boosting: Ensemble method combining several weak learners to form a strong learner.
# Weak learner: Model doing slightly better than random guessing.
# Example of weak learner: Decision stump (CART whose maximum depth is 1).
# 
# Boosting
# 
# Train an ensemble of predictors sequentially.
# Each predictor tries to correct its predecessor.
# Most popular boosting methods:
#     AdaBoost, Gradient Boosting
# =============================================================================

# =============================================================================
# Adaboost
# =============================================================================

#Each predictor pays more attention to the instances wrongly predicted by its predecessor.
#Achieved by changing the weights oftraining instances.
#
#Each predictor is assigned a coefcient α.
#α depends on the predictor's training error.

# A small value of learning rate should be compensated by a greater number of estimators
#For classification, each predictor predicts the label of the new instance and the 
#ensemble's prediction is obtained by weighted majority voting. For 
#regression, the same procedure is applied and ensemble's prediction is obtained 
#by a weighted average. Individual predictors don't need to be CARTs however
#CARTs are used most of the time in boosting because of their high variance. 

#AdaBoost: Prediction
#Classication:
#    Weighted majority voting.
#    In sklearn: AdaBoostClassifier .
#
#Regression:
#    Weighted average.
#    In sklearn: AdaBoostRegressor

#AdaBoost Classication in sklearn (Breast Cancer dataset)

# Import models and utility functions
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

bc = pd.read_csv('data/breast-cancer-data.csv', index_col = 'id')

X = bc.drop('diagnosis', axis = 1)
y = bc['diagnosis']

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=SEED)

# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=1, 
                            random_state=SEED)

# Instantiate an AdaBoost classifier 'adab_clf'
adb_clf = AdaBoostClassifier(base_estimator=dt, 
                             n_estimators=100)

# Fit 'adb_clf' to the training set
adb_clf.fit(X_train, y_train)

# Predict the test set probabilities of positive class
y_pred_proba = adb_clf.predict_proba(X_test)[:,1]

# Evaluate test-set roc_auc_score
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)

# Print adb_clf_roc_auc_score
print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))

#--------------------- Indian Liver Patient------------------------------------
#In addition, given that this dataset is imbalanced, 
#you'll be using the ROC AUC score as a metric instead of accuracy.

ind = pd.read_csv('data/Indian Liver Patient Dataset (ILPD).csv').dropna()
ind = pd.get_dummies(ind, drop_first = True)

ind.is_patient = ind.is_patient.astype('category')  

X = ind.drop('is_patient', axis = 1)
y = ind['is_patient']

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=SEED)
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Instantiate dt
dt = DecisionTreeClassifier(max_depth = 2, random_state=1)

# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt, 
                         n_estimators=180, 
                         random_state=1)

# Fit ada to the training set
ada.fit(X_train, y_train)

# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:,1]

# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))

# =============================================================================
# Gradient Boosting (GB)
# =============================================================================

#Gradient Boosted Trees
#
#Sequential correction of predecessor's errors.
#Does not tweak the weights oftraining instances.
#Fit each predictor is trained using its predecessor's residual errors as labels.
#Gradient Boosted Trees: a CART is used as a base learner
#
#Tree 1 is trained using the features matrix X and dataset labels y. The 
#predictions labeled y1 that are used to determine the training set residual 
#errors r1. Tree2 is then trained on the feature matrix X and residual errors r1
#of tree 1 as the labels. THe predicted residuals r1 hat are then used to 
#determine the residuals of residuals which are labeled r2. THis process is 
#repeated untill all of N trees forming the ensemble are trained. 
#
#A important parameter used in training gradient boosted trees is shrinkage. 
#In this context shrinkage refers to the fact that prediction of each tree in ensemble
#is shrinked after is multiplied by a learning rate eta which is a number between
#0 and 1. 
#Similarly to Adaboost, there's a trade off between eta and the number of estimators. 
#Decreasing the learning rate needs to be compensated by increasing the number of 
#trees in order for the ensemble to reach a certain performance. 
#
#Once all trees in the ensemble are trained, predictions can be made. 
#When a new instance is available, each tree predicts a label and final prediction 
#is given by the formula below. 
#
#y-pred = y1 + ηr1 + ... + ηrN
#
#A similar method can be used for classification problems. 

# =============================================================================
# Gradient Boosting in sklearn (auto dataset)
# =============================================================================

# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

auto = pd.read_csv('data/auto-mpg.csv', na_values = '?').dropna()
X = auto.drop(['car name', 'mpg'], axis = 1)
y = auto['mpg']

# Set seed for reproducibility
SEED = 1
# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=SEED)

# Instantiate a GradientBoostingRegressor 'gbt'
gbt = GradientBoostingRegressor(n_estimators=300, 
                                max_depth=1, 
                                random_state=SEED)

# Fit 'gbt' to the training set
gbt.fit(X_train, y_train)

# Predict the test set labels
y_pred = gbt.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)

# Print the test set RMSE
print('Test set RMSE: {:.2f}'.format(rmse_test))

# -------------------- Bike Sharing Demand dataset ----------------------------
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

bike = pd.read_csv('data/bike_share_train.csv', parse_dates = True, 
                   index_col = 'datetime')

X = bike.drop(['count'], axis = 1)
y = bike['count']

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=SEED)

# Instantiate gb
gb = GradientBoostingRegressor(max_depth= 4, 
            n_estimators = 200,
            random_state=2)

# Fit gb to the training set
gb.fit(X_train, y_train)

# Predict test set labels
y_pred = gb.predict(X_test)

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute MSE
mse_test = MSE(y_test, y_pred)

# Compute RMSE
rmse_test = mse_test ** (1/2)

# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))

#Gradient Boosting: Cons

#GB involves an exhaustive search procedure.
#Each CART is trained to nd the best split points and features.
#May lead to CARTs using the same split points and maybe the same features

# =============================================================================
# Stochastic Gradient Boosting (SGB)
# 
# Gradient boosting involves an exhaustive search procedure. Each tree in the 
# ensemble is trained to find the best split-points and the best features. This
# procedure may lead to CARTs that use the same split-points and possible the same
# features. 
# 
# To mitigate these effects, you can use an algorithm known as the stochastic 
# gradient boosting. Here, each CART is trained on a random subset of the training 
# data. 
# 
# The subset is sampled without replacement. Furthermore, at the level of each node,
# features are sampled without replacement when choosing the best split points. 
# This creates further diversity in the ensemble and the net effect is adding 
# more variance to the ensemble trees. 
# 
# 
# In training, first instead of providing all the training instances to a tree, only
# a fraction of these instances are provided through sampling without replacement. 
# The sampled data is then used for training a tree. However, not all features are 
# considered when a split is made. Instead only a certain randomly sampled fraction
# of these features are used for this purpose. Once a tree is made, predictions are 
# made and the residual errors can be computed. 
# 
# These residual errors are multiplied by the learning rate eta and are fed to the 
# next tree in the ensemble. The procedure is repeated sequentially untill all  trees 
# in the ensemble are trained. 
# =============================================================================

#Stochastic Gradient Boosting
#
#Each tree is trained on a random subset of rows ofthe training data.
#The sampled instances (40%-80% ofthe training set) are sampled without replacement.
#Features are sampled (without replacement) when choosing split points.
#Result: further ensemble diversity.
#Effect: adding further variance to the ensemble oftrees.


#Stochastic Gradient Boosting in sklearn (auto dataset)

# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# Set seed for reproducibility
SEED = 1

auto = pd.read_csv('data/auto-mpg.csv', na_values = '?').dropna()
X = auto.drop(['car name', 'mpg'], axis = 1)
y = auto['mpg']

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=SEED)

# Instantiate a stochastic GradientBoostingRegressor 'sgbt'
sgbt = GradientBoostingRegressor(max_depth=1,
                                 subsample=0.8,
                                 max_features=0.2,
                                 n_estimators=300,
                                 random_state=SEED)

# Fit 'sgbt' to the training set
sgbt.fit(X_train, y_train)

# Predict the test set labels
y_pred = sgbt.predict(X_test)

# Evaluate test set RMSE 'rmse_test'
rmse_test = MSE(y_test, y_pred)**(1/2)

# Print 'rmse_test'
print('Test set RMSE: {:.2f}'.format(rmse_test))

#----------------------Bike Sharing Demand ------------------------------------
bike = pd.read_csv('data/bike_share_train.csv', parse_dates = True, 
                   index_col = 'datetime')

X = bike.drop(['count'], axis = 1)
y = bike['count']

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=SEED)

# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4, 
            subsample= 0.9,
            max_features= 0.75,
            n_estimators= 200,                                
            random_state=2)

# Fit sgbr to the training set
sgbr.fit(X_train, y_train)

# Predict test set labels
y_pred = sgbr.predict(X_test)

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute test set MSE
mse_test = MSE(y_test, y_pred)

# Compute test set RMSE
rmse_test = mse_test ** (1/2)

# Print rmse_test
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))

# =============================================================================
# Tuning a CART's Hyperprameters
# =============================================================================

#Hyperparameters
#
#Machine learning model:
#    parameters: learned from data
#        CART example: split-point of a node, split-feature of a node, ...
#
#hyperparameters: not learned from data, set prior to training
#CART example: max_depth , min_samples_leaf , splitting criterion ..
#
#
#Whatis hyperparameter tuning?
#
#Problem: search for a set of optimal hyperparameters for a learning algorithm.
#Solution: nd a set of optimal hyperparameters that results in an optimal model.
#Optimal model: yields an optimal score.
#
#Score: in sklearn defaults to accuracy (classication) and R squared (regression).
#Cross validation is used to estimate the generalization performance.
#
#Approaches to hyperparameter tuning
#
#Grid Search
#Random Search
#Bayesian Optimization
#Genetic Algorithms

#Grid search cross validation
#
#Manually set a grid of discrete hyperparameter values.
#Set a metric for scoring model performance.
#Search exhaustively through the grid.
#For each set of hyperparameters, evaluate each model's CV score.
#The optimal hyperparameters are those ofthe model achieving the best CV score.


# =============================================================================
# Inspecting the hyperparameters of a CART in sklearn
# =============================================================================

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Set seed to 1 for reproducibility
SEED = 1

bc = pd.read_csv('data/breast-cancer-data.csv', index_col = 'id')

X = bc.drop('diagnosis', axis = 1)
y = bc['diagnosis']

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,
                                                   test_size=0.2,
                                                   stratify=y,
                                                   random_state=1)

# Instantiate a DecisionTreeClassifier 'dt'
dt = DecisionTreeClassifier(random_state=SEED)

# Print out 'dt's hyperparameters
print(dt.get_params())

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of hyperparameters 'params_dt'
params_dt = {'max_depth': [3, 4,5, 6],
             'min_samples_leaf': [0.04, 0.06, 0.08],
             'max_features': [0.2, 0.4,0.6, 0.8]
             }

# Instantiate a 10-fold CV grid search object 'grid_dt'
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring= 'accuracy',
                       cv=10,
                       n_jobs=-1)

# Fit 'grid_dt' to the training data
grid_dt.fit(X_train, y_train)

# Extract best hyperparameters from 'grid_dt'
best_hyperparams = grid_dt.best_params_

print('Best hyerparameters:\n', best_hyperparams)

# Extract best CV score from 'grid_dt'
best_CV_score = grid_dt.best_score_
print('Best CV accuracy'.format(best_CV_score))

# Extract best model from 'grid_dt'
best_model = grid_dt.best_estimator_

#The best model can be extracted using .best_estimator_ attribute. Note that this
#model is fitted using the whole training set because the refit parameter of 
#GridSearchCV is set to true by default. 

# Evaluate test set accuracy
test_acc = best_model.score(X_test,y_test)

# Print test set accuracy
print("Test set accuracy of best model: {:.3f}".format(test_acc))

# -------------------------Indian Liver Patient dataset -----------------------

# Given that this dataset is imbalanced, you'll be using the ROC AUC score as a 
# metric instead of accuracy.

ind = pd.read_csv('data/Indian Liver Patient Dataset (ILPD).csv').dropna()
ind = pd.get_dummies(ind, drop_first = True)

ind.is_patient = ind.is_patient.astype('category')  

X = ind.drop('is_patient', axis = 1)
y = ind['is_patient']

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,
                                                   test_size=0.2,
                                                   stratify=y,
                                                   random_state=1)
# Instantiate a DecisionTreeClassifier 'dt'
dt = DecisionTreeClassifier(random_state=SEED)

# Define params_dt
params_dt = {'max_depth':[2,3,4], 
             'min_samples_leaf':[0.12,0.14,0.16,0.18]
             }

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt,
                       param_grid= params_dt,
                       scoring= 'roc_auc',
                       cv= 5,
                       n_jobs=-1)

grid_dt.fit(X_train, y_train)

# Import roc_auc_score from sklearn.metrics 
from sklearn.metrics import roc_auc_score

# Extract the best estimator
best_model = grid_dt.best_estimator_

# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:,1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))

# =============================================================================
# Random Forests Hyperparameters
# =============================================================================

auto = pd.read_csv('data/auto-mpg.csv', na_values = '?').dropna()
X = auto.drop(['car name', 'mpg'], axis = 1)
y = auto['mpg']

# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,
                                                   test_size=0.2,
                                                   random_state=1)
# Set seed for reproducibility
SEED = 1

# Instantiate a random forests regressor 'rf'
rf = RandomForestRegressor(random_state= SEED)

# Inspect rf' s hyperparameters
rf.get_params()

# Basic imports
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV

# Define a grid of hyperparameter 'params_rf'
params_rf = {'n_estimators': [300, 400, 500],
             'max_depth': [4, 6, 8],
             'min_samples_leaf': [0.1, 0.2],
             'max_features': ['log2','sqrt']
             }

# Instantiate 'grid_rf'
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       cv=3,
                       scoring= 'neg_mean_squared_error',
                       verbose=1,
                       n_jobs=-1)

# Fit 'grid_rf' to the training set
grid_rf.fit(X_train, y_train)

# Extract best hyperparameters from 'grid_rf'
best_hyperparams = grid_rf.best_params_

print('Best hyerparameters:\n', best_hyperparams)

# Extract best model from 'grid_rf'
best_model = grid_rf.best_estimator_

# Predict the test set labels
y_pred = best_model.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)

# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# ----------------------------bike_share_train---------------------------------

# Import GridSearchCV
from sklearn.model_selection import  GridSearchCV
# Import mean_squared_error from sklearn.metrics as MSE 
from sklearn.metrics import mean_squared_error as MSE


bike = pd.read_csv('data/bike_share_train.csv', parse_dates = True, 
                   index_col = 'datetime')

X = bike.drop(['count'], axis = 1)
y = bike['count']

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=SEED)
# Define the dictionary 'params_rf'
params_rf = {'n_estimators':[100,350,500], 
             'max_features':['log2', 'auto', 'sqrt'], 
             'min_samples_leaf':[2,10,30]}

# To evaluate each model in the grid, you'll be using the negative mean squared error metric.

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=3,
                       verbose=1,
                       n_jobs=-1)

grid_rf.fit(X_train, y_train)

# Extract the best estimator
best_model = grid_rf.best_estimator_

# Predict test set labels
y_pred = best_model.predict(X_test)

# Compute rmse_test
rmse_test = MSE(y_test, y_pred)**(1/2)

# Print rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test)) 