#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:53:18 2019

@author: amin
"""

# =============================================================================
# Loading the data
# 
# Now it's time to check out the dataset! You'll use pandas 
# (which has been pre-imported as pd) to load your data into a DataFrame and then
#  do some Exploratory Data Analysis (EDA) of it.
# 
# The training data is available as TrainingData.csv. Your first task is to 
# load it into a DataFrame in the IPython Shell using pd.read_csv() along with 
# the keyword argument index_col=0.
# 
# Use methods such as .info(), .head(), and .tail() to explore the budget data and 
# the properties of the features and labels.
# =============================================================================

import pandas as pd 
df = pd.read_csv('data/da1dd36a-a497-42c7-b3f3-4a225944bdba/TrainingData.csv', index_col =0)

df.info()
df.head()

#Some of the column names correspond to features - descriptions of the budget 
#items - such as the Job_Title_Description column. The values in this column 
#tell us if a budget item is for a teacher, custodian, or other employee.
#
#Some columns correspond to the budget item labels you will be trying to predict 
#with your model. For example, the Object_Type column describes whether the 
#budget item is related classroom supplies, salary, travel expenses

# Summarizing the data

#You'll notice that there are two numeric columns, called FTE and Total.
#
#FTE: Stands for "full-time equivalent". If the budget item is associated to an
#employee, this number tells us the percentage of full-time that the employee
#works. A value of 1 means the associated employee works for the school full-time. 
#A value close to 0 means the item is associated to a part-time or contracted employee.
#
#Total: Stands for the total cost of the expenditure. This number tells us how much 
#the budget item cost.

# Print the summary statistics
print(df.describe())

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Create the histogram
plt.hist(df['FTE'])

# Add title and labels
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')
plt.xlim([-1,50])

# Display the histogram
plt.show()

#The high variance in expenditures makes sense (some purchases are cheap some are 
#expensive). Also, it looks like the FTE column is bimodal. That is, there are 
#some part-time and some full-time employees.

# =============================================================================
# Exploring datatypes in pandas
# =============================================================================

df.dtypes.value_counts()

# =============================================================================
# Encode the labels as categorical variables
# 
# Remember, your ultimate goal is to predict the probability that a certain label 
# is attached to a budget line item. You just saw that many columns in your 
# data are the inefficient object type. Does this include the labels you're 
# trying to predict? Let's find out!
# 
# There are 9 columns of labels in the dataset. Each of these columns is a 
# category that has many possible values it can take.
# =============================================================================

LABELS = ['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type', 'Position_Type', 'Object_Type', 'Pre_K',
          'Operating_Status']

df[LABELS].dtypes

#Because category datatypes are much more efficient your task is to convert the 
#labels to category types using the .astype() method.
#
#Note: .astype() only works on a pandas Series. Since you are working with a 
#pandas DataFrame, you'll need to use the .apply() method and provide a lambda 
#function called categorize_label that applies .astype() to each column, x

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
df[LABELS] = df[LABELS].apply(categorize_label, axis = 0)

# Print the converted dtypes
print(df[LABELS].dtypes)

# =============================================================================
# Counting unique labels
# =============================================================================

#pandas, which has been pre-imported as pd, provides a pd.Series.nunique method 
#for counting the number of unique values in a Series.

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# unique number of labels for a single column
df['Function'].nunique()

# Calculate number of unique values for each label: num_unique_labels
num_unique_labels = df[LABELS].apply(pd.Series.nunique)

# Plot number of unique values for each label
num_unique_labels.plot(kind = 'bar')

# Label the axes
plt.xlabel('Labels')
plt.ylabel('Number of unique values')

# Display the plot
plt.show()

As Peter explained in the video, log loss provides a steep penalty for 
predictions that are both wrong and confident, i.e., a high probability is 
assigned to the incorrect class

# =============================================================================
# Computing log loss with NumPy
# =============================================================================

#Log loss binary classification

#Log loss for binary classification
#● Actual value: y = {1=yes, 0=no}
#● Prediction (probability that the value is 1): p
#
#logloss = − 1/N SIGMA (N TO I =1 ) (yi log(pi) + (1 − yi) log(1 − pi))
#
#Example 
#
#logloss(N=1) = y log(p) + (1 − y) log(1 − p)
#
#True label = 0
#
#Model confidently predicts 1 (with p = 0.90)
#Log loss = (1 - y)log(1-p)
#         = log(1 - 0.9)
#         = log(0.1)
#         = 2.30
## -----------------------Example 2 ----------------
#         
#True label = 1
#Model predicts 0 (with p = 0.50)
#Log loss = 0.69
#Better to be less confident than confident and wrong


import numpy as np

def compute_log_loss(predicted, actual, eps=1e-14):
     """ Computes the logarithmic loss between predicted and
     actual when these are 1D arrays.
     
     :param predicted: The predicted probabilities as floats between 0-1
     :param actual: The actual binary labels. Either 0 or 1.
     :param eps (optional): log(0) is inf, so we need to offset our
     predicted values slightly by eps from 0 or 1.
     """
     
     predicted = np.clip(predicted, eps, 1 - eps)
     # we use the clip function which sets a maximum and minimum value for elements 
     #in an array. Since log of 0 is negative infinity we want to offset our predictions so slightly
     # from being exactly 1 or exactly 0 so that the score remains a real number
     loss = -1 * np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
     
     return loss


compute_log_loss(predicted=0.9, actual=0)
compute_log_loss(predicted=0.5, actual=1)

actual_labels = np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])
correct_confident = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05])
correct_not_confident = np.array([0.65, 0.65, 0.65, 0.65, 0.65, 0.35, 0.35, 0.35, 0.35, 0.35])
wrong_not_confident = np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.65, 0.65, 0.65, 0.65, 0.65])
wrong_confident = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95])

# Compute and print log loss for 1st case
correct_confident_loss = compute_log_loss(correct_confident, actual_labels)
print("Log loss, correct and confident: {}".format(correct_confident_loss)) 

# Compute log loss for 2nd case
correct_not_confident_loss = compute_log_loss(correct_not_confident, actual_labels)
print("Log loss, correct and not confident: {}".format(correct_not_confident_loss)) 

# Compute and print log loss for 3rd case
wrong_not_confident_loss = compute_log_loss(wrong_not_confident, actual_labels)
print("Log loss, wrong and not confident: {}".format(wrong_not_confident_loss)) 

# Compute and print log loss for 4th case
wrong_confident_loss = compute_log_loss(wrong_confident, actual_labels)
print("Log loss, wrong and confident: {}".format(wrong_confident_loss)) 

# Compute and print log loss for actual labels
actual_labels_loss = compute_log_loss(actual_labels, actual_labels)
print("Log loss, actual labels: {}".format(actual_labels_loss)) 

# =============================================================================
# It's time to build a model
# =============================================================================

# Starting with a very simple model in this case logistic regression. 
# We will start with a model that uses just the numeric columns. We will use 
#multi class logistic regression which  treats each label column as independent
#
#The model will train a logistic regression classifier for each of these columns 
#separately and then use those models to predict whether the label appears or not
#for any given row.

#We want to split our data to training and test set. However, because of the nature 
#of the data, the simple approach won't work. Some labels only appear in a small
#fraction of the dataset. If we split our dataset randomly, we may end up with 
#labels in our test set that never appeared in our training set. 
#
#ONe solution is StratifiedShuffleSplit. This Scikit learn function only works
#if you have a single target variable. In our case we have many target variables.:
#As a work around, there's a utility function in module multilabel.py called 
#multilabel_train_test_split that will ensure that all the classes are represented 
#in both train and test sets. 

#You'll start with a simple model that uses just the numeric columns of your 
#DataFrame when calling multilabel_train_test_split. The data has been read 
#into a DataFrame df and a list consisting of just the numeric columns is 
#available as NUMERIC_COLUMNS

    
NUMERIC_COLUMNS = ['FTE', 'Total']

# Create the new DataFrame: numeric_data_only
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(df[LABELS])

# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,
                                                               label_dummies,
                                                               size=0.2, 
                                                               seed=123)

# Print the info
print("X_train info:")
print(X_train.info())
print("\nX_test info:")  
print(X_test.info())
print("\ny_train info:")  
print(y_train.info())
print("\ny_test info:")  
print(y_test.info())

# With the data split, you can now train a model!

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))

# =============================================================================
# Use your model to predict values on holdout data
# =============================================================================

# Load the holdout data: holdout
holdout = pd.read_csv('data/da1dd36a-a497-42c7-b3f3-4a225944bdba/TestData.csv', index_col=0)

# Generate predictions: predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))


# =============================================================================
# Writing out your results to a csv for submission
# =============================================================================

# You'll need to make sure your submission obeys the correct format.

# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns,
                             index=holdout.index,
                             data=predictions)


# Save prediction_df to csv
prediction_df.to_csv('data/predictions.csv')

# Submit the predictions for scoring: score
score = score_submission(pred_path='data/predictions.csv')

# Print score
print('Your model, trained with numeric data only, yields logloss score: {}'.format(score))

# =============================================================================
# A very brief introduction to NLP
# =============================================================================

#A very brief introduction to NLP
#● Data for NLP:
#● Text, documents, speech, …
#● Tokenization -- Spliting a string into segments
#● Store segments as list
#● Example: ‘Natural Language Processing’
#● —> [‘Natural’, ‘Language’, ‘Processing’]

#
#Bag of words representation
#● Count the number of times a particular token appears
#● “Bag of words”
#● Count the number of times a word was
#pulled out of the bag
#● This approach discards information about word order
#● “Red, not blue” is the same as “blue, not red

# =============================================================================
# # Creating a bag-of-words in scikit-learn
# =============================================================================

#In this exercise, you'll study the effects of tokenizing in different ways by 
#comparing the bag-of-words representations resulting from different token patterns.
#
#You will focus on one feature only, the Position_Extra column, which describes 
#any additional information not captured by the Position_Type label.
#
#For example, in the Shell you can check out the budget item in row 8960 of the 
#data using df.loc[8960]. Looking at the output reveals that this Object_Description
# is overtime pay. For who? The Position Type is merely "other", but the Position
# Extra elaborates: "BUS DRIVER". Explore the column further to see more instances. 
# It has a lot of NaN values.

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Fill missing values in df.Position_Extra
df.Position_Extra.fillna('', inplace = True)

# Instantiate the CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern = TOKENS_ALPHANUMERIC)

# Fit to the data
vec_alphanumeric.fit(df.Position_Extra)

# Print the number of tokens and first 15 tokens
msg = "There are {} tokens in Position_Extra if we split on non-alpha numeric"
print(msg.format(len(vec_alphanumeric.get_feature_names())))
print(vec_alphanumeric.get_feature_names()[:15])

# You've got bag-of-words in the bag!

# -------------------------------Splitting on white space example--------------

TOKENS_BASIC = '\\S+(?=\\s+)'

vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)
vec_basic.fit(df.Position_Extra) 

msg = 'There are {} tokens in Program_Description if tokens are any non-whitespace'
print(msg.format(len(vec_basic.get_feature_names()))) 


# =============================================================================
# Combining text columns for tokenization
# 
# In order to get a bag-of-words representation for all of the text data in our 
# DataFrame, you must first convert the text data in each row of the DataFrame 
# into a single string.
# 
# In the previous exercise, this wasn't necessary because you only looked at one 
# column of data, so each row was already just a single string. CountVectorizer 
# expects each row to just be a single string, so in order to use all of the text 
# columns, you'll need a method to turn a list of strings into a single string.
# =============================================================================


# Define combine_text_columns()
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """
    
    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis = 1)
    
    # Replace nans with blanks
    text_data.fillna('', inplace = True)
    
    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)

# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the basic token pattern
TOKENS_BASIC = '\\S+(?=\\s+)'

# Create the alphanumeric token pattern
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate basic CountVectorizer: vec_basic
vec_basic = CountVectorizer(token_pattern= TOKENS_BASIC)

# Instantiate alphanumeric CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern = TOKENS_ALPHANUMERIC)

# Create the text vector
text_vector = combine_text_columns(df)

# Fit and transform vec_basic
vec_basic.fit_transform(text_vector)

# Print number of tokens of vec_basic
print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))

# Fit and transform vec_alphanumeric
vec_alphanumeric.fit_transform(text_vector)

# Print number of tokens of vec_alphanumeric
print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))














