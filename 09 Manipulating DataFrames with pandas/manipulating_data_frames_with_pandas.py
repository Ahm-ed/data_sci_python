#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:34:29 2019

@author: amin
"""

import pandas as pd
import numpy as np

# =============================================================================
# Indexing DataFrames
# =============================================================================

datafile = 'data/sales.csv'

sales = pd.read_csv(datafile, header = None, delimiter = ' ',
                    names = ['month', 'eggs', 'salt', 'spam'],
                    index_col = 0)

# 1. Indexing using square brackets. df['column labe']['row label']

sales['salt']['Jan']

# 2. Using column attribute and row label 
# Columns may also be reffered to as attributes of a DataFrame if their labels
# are valid python identifiers

sales.eggs['Mar']
sales.eggs[['Mar','May']]
sales.eggs['Mar':'May']

# 3. Using the .loc and .iloc accessors. 
# eg. df.loc['row specifier', 'column specifier']

sales.loc['May','spam']
sales.iloc[4, 2]

## Selecting only some columns 

sales[['salt','eggs']]

# exercise

election = pd.read_csv('data/election.csv',
                       index_col = 'county')

election.loc['Bedford', 'winner']

#Positional and labeled indexing

# Assign the row position of election.loc['Bedford']: x
x = 4

# Assign the column position of election['winner']: y
y = 4

# Print the boolean equivalence
print(election.iloc[x, y] == election.loc['Bedford', 'winner'])

# Indexing and column rearrangement
# Create a separate dataframe with the columns ['winner', 'total', 'voters']: results
results = election[['winner', 'total', 'voters']]

# Print the output of results.head()
print(results.head())


#Slicing DataFrames

sales['eggs'] # Selecting one column with one pair of [] a series object. 

#A series is a one - dimentional array with a labelled index (like a hyrid 
#between a numpy array and a dictionary)

sales['eggs'][1:4]
sales.loc[:, 'eggs':'salt']
sales.loc['Jan':'Apr', :]
sales.loc['Mar':'May', 'salt':'spam']

sales.iloc[2:5, 1:]

# using lists rather than slices

sales.loc['Jan':'May', ['eggs','spam']]

sales.iloc[[0,4,5], 0:2]

#Slicing rows

# Slice the row labels 'Perry' to 'Potter': p_counties
p_counties = election.loc['Perry':'Potter']

# Print the p_counties DataFrame
print(p_counties)

#Slice the row labels 'Potter' to 'Perry' in reverse order. 
#To do this for hypothetical row labels 'a' and 'b', you could use a 
#stepsize of -1 like so: df.loc['b':'a':-1]

# Slice the row labels 'Potter' to 'Perry' in reverse order: p_counties_rev
p_counties_rev = election.loc['Potter':'Perry':-1]

# Print the p_counties_rev DataFrame
print(p_counties_rev)

#Slicing columns

# Slice the columns from the starting column to 'Obama': left_columns
left_columns = election.loc[:,:'Obama']

# Print the output of left_columns.head()
print(left_columns.head())

# Slice the columns from 'Obama' to 'winner': middle_columns
middle_columns = election.loc[:,'Obama':'winner']

# Print the output of middle_columns.head()
print(middle_columns.head())

# Slice the columns from 'Romney' to the end: 'right_columns'
right_columns = election.loc[:,'Romney':]

# Print the output of right_columns.head()
print(right_columns.head())

#Subselecting DataFrames with lists

# Create the list of row labels: rows
rows = ['Philadelphia', 'Centre', 'Fulton']

# Create the list of column labels: cols
cols = ['winner', 'Obama', 'Romney']

# Create the new DataFrame: three_counties
three_counties = election.loc[rows, cols]

# Print the three_counties DataFrame
print(three_counties)

# Filtering DataFrames

# Creating a boolean series 
enough_salt_sold = sales.salt > 60

sales[enough_salt_sold]

# OR

sales[sales.salt > 60]

# Combining filters

sales[(sales.salt > 60) & (sales.eggs < 200)] # Both conditions

sales[(sales.salt >= 50) | (sales.eggs < 200)] # Either condition

#Dataframes with zeros and NaNs

sales1 = sales.copy()
sales1['bacon'] = [0, 0, 50, 60, 70, 80]

print(sales1)

# Selecting columns with all nonzero (non-zeros) entries 

sales1.loc[:, sales1.all()]

# Selecting columns with any nonzero (non - zeros) entry

sales1.loc[:, sales1.any()] # in this case, there are no all zero column, so all of sales1 is return

# Selecting columns with any NaNs

sales1.loc[:, sales1.isnull().any()]

# Selecting columns without NaNs

sales1.loc[:, sales1.notnull().all()]

# Drop rows with missing data ( with any NaNs)

sales1.dropna(how = 'any')

# using how = 'any' drops the row 'May' because it has a NaN entry
# by contrast how = 'all' will keep this row

sales1.dropna(how = 'all')

# Filtering a column based on another

sales.eggs[sales.salt > 55]

# This type of filtering allows us to calculate one column based on another

# Modifying a column based on another
sales.eggs[sales.salt > 55] += 5

# Create the boolean array: high_turnout
high_turnout = election.turnout > 70

# Filter the election DataFrame with the high_turnout array: high_turnout_df
high_turnout_df = election[high_turnout]

# Print the high_turnout_results DataFrame
print(high_turnout_df)

# =============================================================================
# Filtering columns using other columns
# 
# The election results DataFrame has a column labeled 'margin' which 
# expresses the number of extra votes the winner received over the losing 
# candidate. This number is given as a percentage of the total votes cast. 
# It is reasonable to assume that in counties where this margin was less 
# than 1%, the results would be too-close-to-call.
# 
# Your job is to use boolean selection to filter the rows where the margin 
# was less than 1. You'll then convert these rows of the 'winner' column 
# to np.nan to indicate that these results are too close to declare a winner
# =============================================================================

# Create the boolean array: too_close
too_close = election.margin < 1

# Assign np.nan to the 'winner' column where the results were too close to call
election.winner[too_close] = np.nan

# Print the output of election.info()
print(election.info())

# =============================================================================
# Filtering using NaNs
# 
# In certain scenarios, it may be necessary to remove rows and columns with 
# missing data from a DataFrame. The .dropna() method is used to perform this 
# action
# =============================================================================

titanic = pd.read_csv('data/titanic.csv')

#you will note that there are many NaNs. 
#You will focus specifically on the 'age' and 'cabin' columns in this exercise.
# Your job is to use .dropna() to remove rows where any of these two columns 
# contains missing data and rows where all of these two columns contain 
# missing data.

# Select the 'age' and 'cabin' columns: df
df = titanic[['age', 'cabin']]

# Print the shape of df
print(df.shape)

# Drop rows in df with how='any' and print the shape
print(df.dropna(how = 'any').shape)

# Drop rows in df with how='all' and print the shape
print(df.dropna(how = 'all').shape)

#use the thresh= keyword argument to drop columns from the full dataset 
#that have less than 1000 non-missing values

# Drop columns in titanic with less than 1000 non-missing values
print(titanic.dropna(thresh =1000, axis='columns').info())

# =============================================================================
# Transforming DataFrames
# =============================================================================

#The best way to transform data in pandas dataframes is the methods inherent to
#pandas dataframes. 

#Next best is to use Numpy unfuncs or Universal functions to transform entire 
#columns of data "elementwise". 

#Suppose we want to convert sales numbers into units of whole dozens (rounded
#down) rather than individual item count. The most efficient way to do this 
#is to use a Pandas built-in method like floordiv. This arithmetic is applied to
#every entry in the dataframe.

sales.floordiv(12) # Convert to dozens unit

# Another way is to use the numpy floor divide function
np.floor_divide(sales, 12)

# We can make a custom function to do this

def dozens(n):
    return n//12

sales.apply(dozens)

# We can also use a lambda function

sales.apply(lambda n: n//12)

# Storing transformations

#All of the preceding computations returned a new DataFrame without altering
#the original dataframe. 
#
#To preserve a computed result we can create a new column storing calculations. 

sales['dozens_of_eggs'] = sales.eggs.floordiv(12)

# Working with string values

#Dataframes, series, index objects come with a handy .str attribute as a kind
#of accessor for vectorized string transformations. 
#
#Here, we reassign the index using .index.str.upper() to all the index uppercase

sales.index = sales.index.str.upper()

# For the index, there is no apply method. The relevant method is called map

sales.index = sales.index.map(str.lower)

# Defining columns using other columns 

sales['salty_eggs'] = sales.salt + sales.dozens_of_eggs

# =============================================================================
# Using apply() to transform a column
# The .apply() method can be used on a pandas DataFrame to apply an 
# arbitrary Python function to every element.
# =============================================================================

pitts = pd.read_csv('data/pitt_weather2013.csv')

# Write a function to convert degrees Fahrenheit to degrees Celsius: to_celsius
def to_celsius(F):
    return 5/9*(F - 32)

# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius
df_celsius = pitts[['Mean TemperatureF', 'Mean Dew PointF']].apply(to_celsius)

# Reassign the columns df_celsius
df_celsius.columns = ['Mean TemperatureC', 'Mean Dew PointC']

# Print the output of df_celsius.head()
print(df_celsius.head())

# =============================================================================
# Using .map() with a dictionary
# 
# The .map() method is used to transform values according to a Python dictionary look-up.
# 
# use a dictionary to map the values 'Obama' and 'Romney' in the 'winner' 
# column to the values 'blue' and 'red', and assign the output to the 
# new column 'color'
# =============================================================================

# Create the dictionary: red_vs_blue. Assign values based on a column. It can also 
#be used to create factors based on a different column

red_vs_blue = {'Obama':'blue', 'Romney':'red'}

# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election.winner.map(red_vs_blue)

# Print the output of election.head()
print(election.head())

# =============================================================================
# Using vectorized functions
# 
# When performance is paramount, you should avoid using .apply() and .map() 
# because those constructs perform Python for-loops over the data stored in a 
# pandas Series or DataFrame. By using vectorized functions instead, you can 
# loop over the data at the same speed as compiled code (C, Fortran, etc.)! 
# NumPy, SciPy and pandas come with a variety of vectorized functions 
# (called Universal Functions or UFuncs in NumPy).
# 
# You can even write your own vectorized functions.
# 
# In this exercise you're going to import the zscore function from scipy.stats 
# and use it to compute the deviation in voter turnout in Pennsylvania from 
# the mean in fractions of the standard deviation. 
# 
# In statistics, the z-score is the number of standard deviations by which an 
# observation is above the mean - so if it is negative, it means the 
# observation is below the mean.
# 
# Instead of using .apply() as you did in the earlier exercises, 
# the zscore UFunc will take a pandas Series as input and return a NumPy array. 
# You will then assign the values of the NumPy array to a new column in the 
# DataFrame
# =============================================================================

# Import zscore from scipy.stats
from scipy.stats import zscore

# Call zscore with election['turnout'] as input: turnout_zscore
turnout_zscore = zscore(election['turnout'])

# Print the type of turnout_zscore
print(type(turnout_zscore))

# Assign turnout_zscore to a new column: election['turnout_zscore']
election['turnout_zscore'] = turnout_zscore

# Print the output of election.head()
print(election.head())








