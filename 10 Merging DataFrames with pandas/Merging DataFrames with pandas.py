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

# ● indices: many index labels within Index data structures
# ● indexes: many pandas Index data structures

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


# =============================================================================
# Sorting DataFrame with the Index & columns
# It is often useful to rearrange the sequence of the rows of a DataFrame by sorting. 
# 
# You don't have to implement these yourself; the principal methods for doing 
# this are .sort_index() and .sort_values()
# =============================================================================

# Read 'monthly_max_temp.csv' into a DataFrame: weather1
weather1 = pd.read_csv('monthly_max_temp.csv', index_col = 'Month')

# Print the head of weather1
print(weather1.head())

# Sort the index of weather1 in alphabetical order: weather2
weather2 = weather1.sort_index()

# Print the head of weather2
print(weather2.head())

# Sort the index of weather1 in reverse alphabetical order: weather3
weather3 = weather1.sort_index(ascending = False)

# Print the head of weather3
print(weather3.head())

# Sort weather1 numerically using the values of 'Max TemperatureF': weather4
weather4 = weather1.sort_values(by = 'Max TemperatureF')

# Print the head of weather4
print(weather4.head())

# =============================================================================
# Reindexing DataFrame from a list
# Sorting methods are not the only way to change DataFrame Indexes. 
# There is also the .reindex() method
# =============================================================================
 
year = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep', 'Oct', 'Nov', 'Dec']

# Reindex weather1 using the list year: weather2
weather2 = weather1.reindex(year)

# Print weather2
print(weather2)

# Reindex weather1 using the list year with forward-fill: weather3
weather3 = weather1.reindex(year).ffill()

# Print weather3
print(weather3)

babynames = pd.read_csv('data/namesbystate/AK.TXT', names = ['state', 'gender', 'year', 'name', 'count'])

_1981 = babynames[babynames['year'] >= 1981]
_1881 = babynames[babynames['year'] < 1981]

names_1981 = _1981.groupby(['name','gender'])['count'].sum()
names_1881 = _1881.groupby(['name','gender'])['count'].sum()

print(names_1981)

# Reindex names_1981 with index of names_1881: common_names
common_names = names_1981.reindex(names_1881.index)

# Print shape of common_names
print(common_names.shape)

# Drop rows with null counts: common_names
common_names = common_names.dropna()

# Print shape of new common_names
print(common_names.shape)

# =============================================================================
# Arithmetic with Series & DataFrames
# =============================================================================

# Loading the weather data

weather = pd.read_csv('data/pitt_weather2013.csv', index_col='Date', parse_dates=True)

weather.loc['2013-7-1':'2013-7-7', 'PrecipitationIn']

# Scalar multiplication

weather.loc['2013-07-01':'2013-07-07', 'PrecipitationIn'] * 2.54 

# Absolute temperature range

week1_range = weather.loc['2013-07-01':'2013-07-07', ['Min TemperatureF', 'Max TemperatureF']]
print(week1_range) 

# Average temperature

week1_mean = weather.loc['2013-07-01':'2013-07-07', 'Mean TemperatureF'] 
print(week1_mean)

# Relative temperature range

week1_range / week1_mean

week1_range.divide(week1_mean, axis='rows')

# Percetange changes

week1_mean.pct_change() * 100

bronze = pd.read_csv('data/bronze.csv', index_col=0)
silver = pd.read_csv('data/silver.csv', index_col=0)
gold = pd.read_csv('data/gold.csv', index_col=0)

bronze + silver

# Using the .add() method

bronze.add(silver) 

# use fill_na

bronze.add(silver, fill_value=0)

bronze + gold + silver

bronze.add(silver, fill_value=0).add(gold, fill_value=0) 

# =============================================================================
# Broadcasting in arithmetic formulas
# 
# In this exercise, you'll work with weather data pulled from wunderground.com. 
# The DataFrame weather has been pre-loaded along with pandas as pd. 
# It has 365 rows (observed each day of the year 2013 in Pittsburgh, PA) and 22 
# columns reflecting different weather measurements each day.
# Remember, ordinary arithmetic operators (like +, -, *, and /) broadcast scalar 
# values to conforming DataFrames when combining scalars & DataFrames in arithmetic expressions. 
# Broadcasting also works with pandas Series and NumPy arrays
# =============================================================================

weather = pd.read_csv('data/pitt_weather2013.csv', index_col='Date', parse_dates=True)

# Extract selected columns from weather as new DataFrame: temps_f
temps_f = weather[['Min TemperatureF', 'Mean TemperatureF', 'Max TemperatureF']]

# Convert temps_f to celsius: temps_c
temps_c = (temps_f - 32) * (5/9)

# Rename 'F' in column names with 'C': temps_c.columns
temps_c.columns = temps_c.columns.str.replace('F','C')

# Print first 5 rows of temps_c
print(temps_c.head())

# =============================================================================
# Computing percentage growth of GDP
# 
# Your job in this exercise is to compute the yearly percent-change of US GDP 
# (Gross Domestic Product) since 2008.
# =============================================================================

# Read 'GDP.csv' into a DataFrame: gdp
gdp = pd.read_csv('data/gdp.csv', parse_dates = True, index_col = "DATE")

# Slice all the gdp data from 2008 onward: post2008
post2008 = gdp['2008':]

# Print the last 8 rows of post2008
print(post2008.tail(8))

# Resample post2008 by year, keeping last(): yearly
yearly = post2008.resample('A').last()

# Print yearly
print(yearly)

# Compute percentage growth of yearly: yearly['growth']
yearly['growth'] = yearly.pct_change() * 100

# Print yearly again
print(yearly)

# =============================================================================
# Converting currency of stocks
# 
# In this exercise, stock prices in US Dollars for 
# the S&P 500 in 2015 have been obtained from Yahoo Finance. 
# The files sp500.csv for sp500 and exchange.csv for the exchange rates 
# are both provided to you.
# 
# Using the daily exchange rate to Pounds Sterling, your task is to convert both 
# the Open and Close column prices.
# =============================================================================

# Read 'sp500.csv' into a DataFrame: sp500
sp500 = pd.read_csv('sp500.csv', parse_dates= True, index_col = 'Date')

# Read 'exchange.csv' into a DataFrame: exchange
exchange = pd.read_csv('exchange.csv', parse_dates= True, index_col = 'Date')

# Subset 'Open' & 'Close' columns from sp500: dollars
dollars = sp500[['Open', 'Close']]

# Print the head of dollars
print(dollars.head())

# Convert dollars to pounds: pounds
pounds = dollars.multiply(exchange['GBP/USD'], axis = 'rows')

# Print the head of pounds
print(pounds.head())

# =============================================================================
# Appending pandas Series
# =============================================================================

# Import pandas
import pandas as pd

# Import pandas
import pandas as pd

# Load 'sales-jan-2015.csv' into a DataFrame: jan
jan = pd.read_csv('data/sales-jan-2015.csv', index_col='Date', parse_dates=True)

# Load 'sales-feb-2015.csv' into a DataFrame: feb
feb = pd.read_csv('data/sales-feb-2015.csv', index_col='Date', parse_dates=True)

# Load 'sales-mar-2015.csv' into a DataFrame: mar
mar = pd.read_csv('data/sales-mar-2015.csv', index_col='Date', parse_dates=True)

# Extract the 'Units' column from jan: jan_units
jan_units = jan['Units']

# Extract the 'Units' column from feb: feb_units
feb_units = feb['Units']

# Extract the 'Units' column from mar: mar_units
mar_units = mar['Units']

# Append feb_units and then mar_units to jan_units: quarter1
quarter1 = jan_units.append(feb_units).append(mar_units)

# Print the first slice from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])

# Print the second slice from quarter1
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])

# Compute & print total sales in quarter1
print(quarter1.sum())

# =============================================================================
# Concatenating pandas Series along row axis
# 
# Having learned how to append Series, you'll now learn how to achieve the same 
# result by concatenating Series instead. You'll continue to work with the sales 
# data you've seen previously. This time, the DataFrames jan, feb, and mar have 
# been pre-loaded.
# 
# Your job is to use pd.concat() with a list of Series to achieve the same result 
# that you would get by chaining calls to .append().
# 
# You may be wondering about the difference between pd.concat() and pandas
# ' .append() method. One way to think of the difference is that .append() 
# is a specific case of a concatenation, while pd.concat() gives you more 
# flexibility, as you'll see in later exercises.
# =============================================================================

# Initialize empty list: units
units = []

# Build the list of Series
for month in [jan, feb, mar]:
    units.append(month['Units'])

# Concatenate the list: quarter1
quarter1 = pd.concat(units, axis='rows')

# Print slices from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])








