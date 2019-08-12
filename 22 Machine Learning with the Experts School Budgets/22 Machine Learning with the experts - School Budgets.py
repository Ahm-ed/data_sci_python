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

#As Peter explained in the video, log loss provides a steep penalty for 
#predictions that are both wrong and confident, i.e., a high probability is 
#assigned to the incorrect class

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

# score = score_submission(pred_path='data/predictions.csv')

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

# =============================================================================
# Instantiate pipeline
# =============================================================================

# ------------Example. Do not run as data not available ----------------------

#Sample data structure 
#
#index numeric     text  with_missing label
#0 -10.856306               4.433240     b
#1   9.973454      foo           NaN     b
#2   2.829785  foo bar      2.469828     a
#3 -15.062947               2.852981     b
#4  -5.786003  foo bar      1.826475     a

# Import Pipeline
from sklearn.pipeline import Pipeline

# Import other necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Split and select numeric data only, no nans 
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric']],
                                                    pd.get_dummies(sample_df['label']), 
                                                    random_state=22)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit the pipeline to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - numeric, no nans: ", accuracy)

# ------------------------Example 2, dealing with missing data ----------------
# Import the Imputer object
from sklearn.preprocessing import Imputer 

# Create training and test sets using only numeric data
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing']],
                                                    pd.get_dummies(sample_df['label']), 
                                                    random_state=456)

# Insantiate Pipeline object: pl
pl = Pipeline([
        ('imp', Imputer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit the pipeline to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all numeric, incl nans: ", accuracy)

# =============================================================================
# Text features
# and
# feature unions
# =============================================================================
# =============================================================================
# Preprocessing text features
# 
# Here, you'll perform a similar preprocessing pipeline step, only this time you'll 
# use the text column from the sample data.
# 
# To preprocess the text, you'll turn to CountVectorizer() to generate a 
# bag-of-words representation of the data, as in Chapter 2. Using the default
#  arguments, add a (step, transform) tuple to the steps list in your pipeline.
# 
# =============================================================================

# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Split out only the text data
X_train, X_test, y_train, y_test = train_test_split(sample_df['text'],
                                                    pd.get_dummies(sample_df['label']), 
                                                    random_state=456)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('vec', CountVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - just text data: ", accuracy)

# =============================================================================
# Multiple types of processing: FunctionTransformer
# 
# The next two exercises will introduce new topics you'll need to make 
# your pipeline truly excel.
# 
# Any step in the pipeline must be an object that implements the fit and 
# transform methods. The FunctionTransformer creates an object with these 
# methods out of any Python function that you pass to it. We'll use it to 
# help select subsets of data in a way that plays nicely with pipelines.
# 
# You are working with numeric data that needs imputation, and text data that 
# needs to be converted into a bag-of-words. You'll create functions that separate
#  the text from the numeric variables and see how the .fit() and .transform() methods work.
# =============================================================================

#Preprocessing multiple dtypes
#● Want to use all available features in one pipeline
#● Problem
#● Pipeline steps for numeric and text preprocessing
#can’t follow each other
#● e.g., output of CountVectorizer can’t be
#input to Imputer
#● Solution
#● FunctionTransformer() & FeatureUnion()

#FunctionTransformer
#● Turns a Python function into an object that a scikit-learn
#pipeline can understand
#● Need to write two functions for pipeline preprocessing
#● Take entire DataFrame, return numeric columns
#● Take entire DataFrame, return text columns
#● Can then preprocess numeric and text data in
#separate pipelines


# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

# Obtain the text data: get_text_data
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)

# Obtain the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False)

# Fit and transform the text data: just_text_data
just_text_data = get_text_data.fit_transform(sample_df)

# Fit and transform the numeric data: just_numeric_data
just_numeric_data = get_numeric_data.fit_transform(sample_df)

# Print head to check results
print('Text Data')
print(just_text_data.head())
print('\nNumeric Data')
print(just_numeric_data.head())

#You can see in the shell that fit and transform are now available to the selectors.
# Let's put the selectors to work!

# =============================================================================
# Multiple types of processing: FeatureUnion
# 
# Now that you can separate text and numeric data in your pipeline, you're ready 
# to perform separate steps on each by nesting pipelines and using FeatureUnion().
# 
# These tools will allow you to streamline all preprocessing steps for your model, 
# even when multiple datatypes are involved. Here, for example, you don't want to 
# impute our text data, and you don't want to create a bag-of-words with our numeric data. 
# Instead, you want to deal with these separately and then join the results together 
# using FeatureUnion().
# 
# In the end, you'll still have only two high-level steps in your 
# pipeline: preprocessing and model instantiation. The difference 
# is that the first preprocessing step actually consists of a pipeline 
# for numeric data and a pipeline for text data. The results of those 
# pipelines are joined using FeatureUnion()
# 
# 
# =============================================================================

# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Split using ALL data in sample_df
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']],
                                                    pd.get_dummies(sample_df['label']), 
                                                    random_state=22)

# Create a FeatureUnion with nested pipeline: process_and_join_features
process_and_join_features = FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )

# Instantiate nested pipeline: pl
pl = Pipeline([
        ('union', process_and_join_features),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])


# Fit pl to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all data: ", accuracy)

# --------------------BACK TO THE COMPETITION DATA-----------------------------

# =============================================================================
# Using FunctionTransformer on the main dataset
# 
# In this exercise you're going to use FunctionTransformer on the primary budget 
# data, before instantiating a multiple-datatype pipeline in the next exercise.
# =============================================================================

# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion

# Get the dummy encoding of the labels
dummy_labels = pd.get_dummies(df[LABELS])

# Get the columns that are features in the original df
NON_LABELS = [c for c in df.columns if c not in LABELS]

# Split into training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
                                                               dummy_labels,
                                                               0.2, 
                                                               seed=123)

# Preprocess the text data: get_text_data
get_text_data = FunctionTransformer(combine_text_columns, validate = False)

# Preprocess the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)


# Add a model to the pipeline

# Complete the pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)

# =============================================================================
# Try a different class of model
# =============================================================================

# Import random forest classifer
from sklearn.ensemble import RandomForestClassifier

# Edit model step in pipeline
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier())
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)

# =============================================================================
# Adjust the model or parameters to improve accuracy
# =============================================================================

# Try changing the parameter n_estimators of RandomForestClassifier(), whose default value is 10, to 15

# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer 

# Add model step to pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier(n_estimators = 15))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)


# =============================================================================
# Learning from the experts: Text Preprocessing
# =============================================================================

#Learning from the expert: text preprocessing
#● NLP tricks for text data
#● Tokenize on punctuation to avoid hyphens, underscores, etc.
#● Include unigrams and bi-grams in the model to
#capture important information involving multiple tokens - e.g., ‘middle school’

#Before you build up to the winning pipeline, it will be useful to look a little 
#deeper into how the text features will be processed.
#
#In this exercise, you will use CountVectorizer on the training data X_train
# to see the effect of tokenization on punctuation

# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the text vector
text_vector = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate the CountVectorizer: text_features
text_features = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit text_features to the text vector
text_features.fit(text_vector)

# Print the first 10 tokens
print(text_features.get_feature_names()[:10])

# =============================================================================
# N-gram range in scikit-learn
# 
# In this exercise you'll insert a CountVectorizer instance into your pipeline 
# for the main dataset, and compute multiple n-gram features to be used in the model.
# =============================================================================

# Import pipeline
from sklearn.pipeline import Pipeline

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Import other preprocessing modules
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import chi2, SelectKBest

# Select 300 best features
chi_k = 300

# Import functional utilities
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import FeatureUnion

# Perform preprocessing
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1,2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])


#You'll notice a couple of new steps provided in the pipeline in this and many 
#of the remaining exercises. Specifically, the dim_red step following the vectorizer 
#step , and the scale step preceeding the clf (classification) step.
#
#These have been added in order to account for the fact that you're using a 
#reduced-size sample of the full dataset in this course. To make sure the models 
#perform as the expert competition winner intended, we have to apply a dimensionality 
#reduction technique, which is what the dim_red step does, and we have to scale 
#the features to lie between -1 and 1, which is what the scale step does.
#
#The dim_red step uses a scikit-learn function called SelectKBest(), applying 
#something called the chi-squared test to select the K "best" features. 
#The scale step uses a scikit-learn function called MaxAbsScaler() in order to 
#squash the relevant features into the interval -1 to 1.

# =============================================================================
# Learning from the expert: a stats trick
# =============================================================================

#Learning from the expert: interaction terms
#● Statistical tool that the winner used: interaction terms
#● Example
#● English teacher for 2nd grade
#● 2nd grade - budget for English teacher
#● Interaction terms mathematically describe when
#tokens appear together

#Adding interaction features with scikit-learn
   
x1 = np.array([0,1])
x2 = np.array([1,1])

x = pd.DataFrame([x1, x2], columns = ['x1', 'x2'], index = ['a', 'b'])

from sklearn.preprocessing import PolynomialFeatures  
interaction = PolynomialFeatures(degree=2,
                                 interaction_only=True,
                                 include_bias=False)    
interaction.fit_transform(x)     

# Bias term allows model to have non-zero y value when x value is zero  

# Sparse interaction features

#The number of interaction terms grows exponentially
#● Our vectorizer saves memory by using a sparse matrix
#● PolynomialFeatures does not support sparse matrices
#● We have provided SparseInteractions to work for this problem:

SparseInteractions(degree=2).fit_transform(x).toarray() 

#Implement interaction modeling in scikit-learn
#
#It's time to add interaction features to your model. The PolynomialFeatures 
#object in scikit-learn does just that, but here you're going to use a custom 
#interaction object, SparseInteractions. Interaction terms are a statistical 
#tool that lets your model express what happens if two features appear together 
#in the same row.
#
#SparseInteractions does the same thing as PolynomialFeatures, but it uses sparse
# matrices to do so. You can get the code for SparseInteractions at this GitHub Gist.
#
#PolynomialFeatures and SparseInteractions both take the argument degree, which 
#tells them what polynomial degree of interactions to compute.
#
#You're going to consider interaction terms of degree=2 in your pipeline. 
#You will insert these steps after the preprocessing steps you've built out so far, 
#but before the classifier steps.

# Instantiate pipeline: pl

from sklearn.feature_selection import SelectKBest

pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),  
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree = 2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# =============================================================================
# Learning from the expert: a computational trick and the winning model
# =============================================================================

# =============================================================================
# Learning from the expert: hashing trick
# =============================================================================

#● Adding new features may cause enormous increase in array size
#● Hashing is a way of increasing memory efficiency
#● Hash function limits possible outputs, fixing array size
#
#When to use the hashing trick
#● Want to make array of features as small as possible
#● Dimensionality reduction
#● Particularly useful on large datasets e.g., lots of text data!

#Some problems are memory-bound and not easily parallelizable, and hashing 
#enforces a fixed length computation instead of using a mutable datatype (like a dictionary).
#
#Enforcing a fixed length can speed up calculations drastically, especially on large datasets!

# =============================================================================
# Implementing the hashing trick in scikit-learn
# =============================================================================

# ---------------example-------------------------------------------------------
from sklearn.feature_extraction.text import HashingVectorizer

vec = HashingVectorizer(norm = None,
                        alternate_sign = False,
                        token_pattern=TOKENS_ALPHANUMERIC,
                        ngram_range=(1, 2))

# ---------------example end --------------------------------------------------

# Import HashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Get text data: text_data
text_data = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)' 

# Instantiate the HashingVectorizer: hashing_vec
hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit and transform the Hashing Vectorizer
hashed_text = hashing_vec.fit_transform(text_data)

# Create DataFrame and print the head
hashed_df = pd.DataFrame(hashed_text.data)
print(hashed_df.head())

#As you can see, some text is hashed to the same value, but this 
#doesn't neccessarily hurt performance.

# =============================================================================
# Build the winning model
# 
# You have arrived! This is where all of your hard work pays off. 
# It's time to build the model that won DrivenData's competition.
# =============================================================================

# Import the hashing vectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Instantiate the winning model pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                     alternate_sign=False, 
                                                     norm=None, 
                                                     binary=False,
                                                     ngram_range= (1,2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

#The parameters non_negative=True, norm=None, and binary=False make the 
#HashingVectorizer perform similarly to the default settings on the CountVectorizer 
#so you can just replace one with the other.
                
                
# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)

# =============================================================================
# Can you do be!er?
# ● You’ve seen the flexibility of the pipeline steps
# ● Quickly test ways of improving your submission
# ● NLP: Stemming, stop-word removal
# ● Model: RandomForest, k-NN, Naïve Bayes
# ● Numeric Preprocessing: Imputation strategies eg handling NaN differently than the default imputer 
# ● Optimization: Grid search over pipeline objects. Perform gridsearch over every object in a pipeline
# ● Experiment with new scikit-learn techniques 
# =============================================================================
