#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 01:55:09 2019

@author: amin
"""

# =============================================================================
# Reading DataFrames from multiple files
# =============================================================================
# Import pandas
import pandas as pd

# Read 'Bronze.csv' into a DataFrame: bronze
bronze = pd.read_csv('Bronze.csv')

# Read 'Silver.csv' into a DataFrame: silver
silver = pd.read_csv('Silver.csv')

# Read 'Gold.csv' into a DataFrame: gold
gold = pd.read_csv('Gold.csv')

# Print the first five rows of gold
print(gold.head())

# Reading DataFrames from multiple files in a loop

# Import pandas
import pandas as pd

# Create the list of file names: filenames
filenames = ['Gold.csv', 'Silver.csv', 'Bronze.csv']

# Create the list of three DataFrames: dataframes
dataframes = []
for filename in filenames:
    dataframes.append(pd.read_csv(filename))

# Print top 5 rows of 1st DataFrame in dataframes
print(dataframes[0].head())

# =============================================================================
# Using comprehensions
# =============================================================================

filenames = ['data/sales-jan-2015.csv', 'data/sales-feb-2015.csv']

dataframes = [pd.read_csv(f) for f in filenames] 

# =============================================================================
# Using glob
# =============================================================================

from glob import glob
filenames = glob('sales*.csv')
dataframes = [pd.read_csv(f) for f in filenames]

# =============================================================================
# # Combining DataFrames from multiple data files
# 
# #In this exercise, you'll combine the three DataFrames from earlier exercises - 
# #gold, silver, & bronze - into a single DataFrame called medals. The approach 
# #you'll use here is clumsy. Later on in the course, you'll see various powerful 
# #methods that are frequently used in practice for concatenating or merging DataFrames.
# =============================================================================

# Import pandas
import pandas as pd

# Make a copy of gold: medals
medals = gold.copy()

# Create list of new column labels: new_labels
new_labels = ['NOC', 'Country', 'Gold']

# Rename the columns of medals using new_labels
medals.columns = new_labels

# Add columns 'Silver' & 'Bronze' to medals
medals['Silver'] = silver['Total']
medals['Bronze'] = bronze['Total']

# Print the head of medals
print(medals.head())

# =============================================================================
# Reindexing
# DataFrames
# =============================================================================
# =============================================================================
# Sorting DataFrame with the Index & columns
# =============================================================================

#● indices: many index labels within Index data structures
#● indexes: many pandas Index data structures

w_mean = pd.read_csv('data/quarterly_mean_temp.csv', index_col='Month')
w_max = pd.read_csv('data/quarterly_max_temp.csv', index_col='Month')

print(w_mean.index)
print(w_max.index)

#Using .reindex

ordered = ['Jan', 'Apr', 'Jul', 'Oct']
w_mean2 = w_mean.reindex(ordered)
print(w_mean2)

# Using .sort_index()

w_mean2.sort_index()

# Reindex from a DataFrame Index

w_mean.reindex(w_max.index)

# Reindexing with missing labels

w_mean3 = w_mean.reindex(['Jan', 'Apr', 'Dec']) 
print(w_mean3)

w_max.reindex(w_mean3.index).dropna()
















