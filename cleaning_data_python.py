#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: amin
"""

# =============================================================================
# Loading and viewing your data
# =============================================================================

import pandas as pd

df = pd.read_csv('https://data.cityofnewyork.us/resource/rvhx-8trz.csv', nrows = 1000)

# Print the head of df
print(df.head())

# Print the tail of df
print(df.tail())

# Print the shape of df
#Note: .shape and .columns are attributes, not methods, so you don't need to follow these with parentheses ()
print(df.shape)

# Print the columns of df
print(df.columns)

print(df.info())

df.describe()

# =============================================================================
# Importing excel file
# =============================================================================
file = 'TREND01-5G-educ-fertility-bubbles (1).xls'
xl = pd.ExcelFile(file)
print(xl.sheet_names)
df1 = xl.parse('data COMPILATION', skiprows=7, index_col= None)

print(df1.info())

df1.describe() ## Only variable that have a numeric type will be returned

# =============================================================================
# Frequency counts
# =============================================================================

# We can use the info method to get the data type of each column. object - string

print(df1.info())

df1.Continent.value_counts(dropna = False) 

## Another way

df1['Continent'].value_counts(dropna = False) 

# Let's check country. We will check on the the top 5 using head

df1['Country '].value_counts(dropna = False).head()

df1['fertility'].value_counts(dropna = False).head()


# =============================================================================
# .describe() can only be used on numeric columns. 
# So how can you diagnose data issues when you have categorical data? 
# One way is by using the .value_counts() method, which returns the frequency 
# counts for each unique value in a column!
# 
# This method also has an optional parameter called dropna which is True by default
# =============================================================================

# Print the value counts for 'Borough'
print(df['borough'].value_counts(dropna=False))

# Print the value_counts for 'State'
print(df['state'].value_counts(dropna = False))

# Print the value counts for 'Site Fill'
print(df['site_fill'].value_counts(dropna = False))


# =============================================================================
# Visual exploratory data analysis
# =============================================================================

import matplotlib.pyplot as plt

# Histogram
df1.population.plot('hist')
plt.show()

# Identifying errors

df1[df1.population > 1000000000]

# Boxplots 

df1.boxplot(column = 'population', by = 'Continent')
plt.show()

#In the IPython Shell, begin by computing summary statistics for the 
#'Existing Zoning Sqft' column using the .describe() method. 
#You'll notice that there are extremely large differences between 
#the min and max values, and the plot will need to be adjusted accordingly. 
#In such cases, it's good to look at the plot on a log scale. 
#The keyword arguments logx=True or logy=True can be passed in to .plot() 
#depending on which axis you want to rescale.

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Describe the column
df['Existing Zoning Sqft'].describe()

# Plot the histogram
df['Existing Zoning Sqft'].plot(kind='hist', rot=70, logx=True, logy=True)

# Display the histogram
plt.show()

# =============================================================================
# Visualizing multiple variables with boxplots
# =============================================================================

# Create the boxplot
df.boxplot(column= 'initial_cost', by='borough', rot=90)

# Display the plot
plt.show()

# =============================================================================
# Visualizing multiple variables with scatter plots
# =============================================================================

# Create and display the first scatter plot
df.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)
plt.show()

# Create and display the second scatter plot
df_subset.plot(kind = 'scatter', x = 'initial_cost', y ='total_est_fee', rot = 70)
plt.show()

# =============================================================================
# Recognizing tidy data
# =============================================================================

#Reshaping your data using melt
#Melting data is the process of turning columns of your data into rows of data
#In this exercise, you will practice melting a DataFrame using pd.melt(). 
#There are two parameters you should be aware of: id_vars and value_vars. 
#The id_vars represent the columns of the data you do not want to 
#melt (i.e., keep it in its current shape), while the value_vars represent the 
#columns you do wish to melt into rows. By default, if no value_vars are provided, 
#all columns not set in the id_vars will be melted.

# Print the head of airquality
print(airquality.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(frame = airquality, id_vars=['Month', 'Day'])

# Print the head of airquality_melt
print(airquality_melt.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'], var_name='measurement', value_name='reading')

# Print the head of airquality_melt
print(airquality_melt.head())

## You can specify the variables to be melted
pd.melt(frame=df, id_vars='name',
        value_vars=['treatment a', 'treatment b']) 

# =============================================================================
# Pivot data
# Pivoting data is the opposite of melting it. 
# Remember the tidy form that the airquality DataFrame was in before you melted it? 
# You'll now begin pivoting it back into that form using the .pivot_table() method!
# While melting takes a set of columns and turns it into a single column, 
# pivoting will create a new column for each unique value in a specified column.
# .pivot_table() has an index parameter which you can use to specify the columns 
# that you don't want pivoted: It is similar to the id_vars parameter of pd.melt(). 
# Two other parameters that you have to specify are columns 
# (the name of the column you want to pivot), and values (the values to be used 
# when the column is pivoted). 
# =============================================================================

weather_tidy = weather.pivot(index='date',columns='element', values='value') 

### When there are duplicate values, you the following below

import numpy as np 

weather2_tidy = weather.pivot(values='value', index='date', columns='element') 

# =============================================================================

# Print the head of airquality_melt
print(airquality_melt.head())

# Pivot airquality_melt: airquality_pivot
airquality_pivot = airquality_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading')

# Print the head of airquality_pivot
print(airquality_pivot.head())

# =============================================================================
# Resetting the index of a DataFrame
# After pivoting airquality_melt in the previous exercise, 
# you didn't quite get back the original DataFrame.
# 
# What you got back instead was a pandas DataFrame with a hierarchical 
# index (also known as a MultiIndex).
# 
# Hierarchical indexes are covered in depth in Manipulating DataFrames 
# with pandas. In essence, they allow you to group columns or rows by another 
# variable - in this case, by 'Month' as well as 'Day'.
# 
# There's a very simple method you can use to get back the original DataFrame 
# from the pivoted DataFrame: .reset_index()
# =============================================================================

# Print the index of airquality_pivot
print(airquality_pivot.index)

# Reset the index of airquality_pivot: airquality_pivot_reset
airquality_pivot_reset = airquality_pivot.reset_index()

# Print the new index of airquality_pivot_reset
print(airquality_pivot_reset.index)

# Print the head of airquality_pivot_reset
print(airquality_pivot_reset.head())

# =============================================================================
# Pivoting duplicate values
# =============================================================================
# Pivot table the airquality_dup: airquality_pivot
airquality_pivot = airquality_dup.pivot_table(index=['Month', 'Day'], 
                                              columns='measurement', values='reading', aggfunc=np.mean)

# Print the head of airquality_pivot before reset_index
print(airquality_pivot.head())

# Reset the index of airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the head of airquality_pivot
print(airquality_pivot.head())

# Print the head of airquality
print(airquality.head())

# =============================================================================
# Splitting a column with .str
# =============================================================================
# Melt tb: tb_melt
tb_melt = pd.melt(frame = tb, id_vars=['country', 'year'])

# Create the 'gender' column. After melting a name variable is created
tb_melt['gender'] = tb_melt.variable.str[0]

# Create the 'age_group' column
tb_melt['age_group'] = tb_melt.variable.str[1:]

# Print the head of tb_melt
print(tb_melt.head())

# =============================================================================
# Splitting a column with .split() and .get()
# =============================================================================

#You now need to use Python's built-in string method called .split(). 
#By default, this method will split a string into parts separated by a space.

# Melt ebola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')

# Create the 'str_split' column
ebola_melt['str_split'] = ebola_melt.type_country.str.split('_')

# Create the 'type' column. using the .get() method to retrieve index 0 of the 'str_split' column of ebola_melt
ebola_melt['type'] = ebola_melt.str_split.str.get(0)

# Create the 'country' column
ebola_melt['country'] = ebola_melt.str_split.str.get(1)

# Print the head of ebola_melt
print(ebola_melt.head())

# =============================================================================
# Concatenating data
# =============================================================================

# =============================================================================
# Combining rows of data
# =============================================================================

row_concat = pd.concat([uber1, uber2, uber3])

concatenated = concatenated.loc[0, :] 

pd.concat([weather_p1, weather_p2], ignore_index=True) 


# Concatenate uber1, uber2, and uber3: row_concat
row_concat = pd.concat([uber1, uber2, uber3])

# Print the shape of row_concat
print(row_concat.shape)

# Print the head of row_concat
print(row_concat.head())

# =============================================================================
# Combining columns of data
# =============================================================================

#Think of column-wise concatenation of data as stitching data together from 
#the sides instead of the top and bottom. To perform this action, you use 
#the same pd.concat() function, but this time with the keyword argument axis=1. 
#The default, axis=0, is for a row-wise concatenation

# Concatenate ebola_melt and status_country column-wise: ebola_tidy
ebola_tidy = pd.concat([ebola_melt, status_country], axis =1)

# Print the shape of ebola_tidy
print(ebola_tidy.shape)

# Print the head of ebola_tidy
print(ebola_tidy.head())

# =============================================================================
# Finding and concatenating data
# =============================================================================
# =============================================================================
# Concatenating many files
# ● Leverage Python’s features with data cleaning in
# pandas
# ● In order to concatenate DataFrames:
# ● They must be in a list
# ● Can individually load if there are a few datasets
# ● But what if there are thousands?
# ● Solution: glob function to find files based
# on a pattern
# 
# Globbing
# ● Pa!ern matching for file names
# ● Wildcards: * ?
# ● Any csv file: *.csv
# ● Any single character: file_?.csv
# ● Returns a list of file names
# ● Can use this list to load into separate DataFrames
# =============================================================================

import glob

csv_files = glob.glob('*.csv')

print(csv_files) 

# Using loops

list_data = []

for filename in csv_files:
    data = pd.read_csv(filename)
    list_data.append(data)

pd.concat(list_data)

# =============================================================================

# Import necessary modules
import glob
import pandas as pd

# Write the pattern: pattern
pattern = '*.csv'

# Save all file matches: csv_files
csv_files = glob.glob(pattern)

# Print the file names
print(csv_files)

# Load the second file into a DataFrame: csv2
csv2 = pd.read_csv(csv_files[1])

# Print the head of csv2
print(csv2.head())

#Iterating and concatenating all matches

# Create an empty list: frames
frames = []

#  Iterate over csv_files
for csv in csv_files:

    #  Read csv into a DataFrame: df
    df = pd.read_csv(csv)
    
    # Append df to frames
    frames.append(df)

# Concatenate frames into a single DataFrame: uber
uber = pd.concat(frames)

# Print the shape of uber
print(uber.shape)

# Print the head of uber
print(uber.head())

# =============================================================================
# Merge data
# =============================================================================

# Merge the DataFrames: o2o
o2o = pd.merge(left=site, right=visited, on=None, left_on='name', right_on='site')

# Print o2o
print(o2o)

# =============================================================================
# Data types
# =============================================================================

# To get data types of a dataframe

print(df.dtypes)

# Converting data types

df['treatment b'] = df['treatment b'].astype(str) 
 
df['sex'] = df['sex'].astype('category') 

df['treatment a'] = pd.to_numeric(df['treatment a'], errors='coerce') 

# Convert the sex column to type 'category'
tips.sex = tips.sex.astype('category')

# Convert the smoker column to type 'category'
tips.smoker = tips.smoker.astype('category')

# Print the info of tips
print(tips.info())

# Convert 'total_bill' to a numeric dtype
tips['total_bill'] = pd.to_numeric(tips['total_bill'], errors='coerce')

# Convert 'tip' to a numeric dtype
tips['tip'] = pd.to_numeric(tips['tip'], errors = 'coerce')

# Print the info of tips
print(tips.info())

# =============================================================================
# Using regular expressions to clean strings
# =============================================================================

#● 17 - 12345678901 - \d*
#● $17 - $12345678901 - \$\d*
#● $17.00 - $12345678901.42 - \$\d*\.\d*
#● $17.89 - $12345678901.24 - \$\d*\.\d{2}
#● $17.895 - $12345678901.999 - ^\$\d*\.\d{2}$
#
#$ means the match from the end of a string. so if we want to actually use $ we have to 
#use \$. 
#
#we use a caret (^) The caret will tell the pattern to start the match at the 
#beginning of the value and the dollar sign ($) will the pattern to match 
#at the end of the value. 

import re
pattern = re.compile('\$\d*\.\d{2}')
result = pattern.match('$17.89')
bool(result) 

# =============================================================================
# Your job in this exercise is to define a regular expression to match US phone
# numbers that fit the pattern of xxx-xxx-xxxx

# Compile the pattern: prog
prog = re.compile('\d{3}-\d{3}-\d{4}')

# See if the pattern matches
result = prog.match('123-456-7890')
print(bool(result))

# See if the pattern matches
result2 = prog.match('1123-456-7890')
print(bool(result2))


# =============================================================================
# Extracting numerical values from strings
# =============================================================================

#\d is the pattern required to find digits. This should be followed with a + so 
#that the previous element is matched one or more times. This ensures that 10 is 
#viewed as one number and not as 1 and 0

# Find the numeric values: matches
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')

# Print the matches
print(matches)

# =============================================================================
# Pattern matching
# =============================================================================

# Write the first pattern
pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890'))
print(pattern1)

# Write the second pattern
pattern2 = bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45'))
print(pattern2)

# Write the third pattern
#Use [A-Z] to match any capital letter followed by \w* to match an arbitrary 
#number of alphanumeric characters.

pattern3 = bool(re.match(pattern='[A-Z]\w*', string='Australia'))
print(pattern3)

# =============================================================================
# Apply a python function across rows and columns. 
# =============================================================================
#accross column
df.apply(np.mean, axis=0)

# accross each row
df.apply(np.mean, axis = 1)

# =============================================================================

#We want to check that the initial cost and total estimated fee to make sure it is
#a valid monetary value, then we want to remove the dollar sign and convert it 
#to a float and find the difference between the two values and save it to a new column. 
#If the entry isn't a valid monetary value, return a NaN missing value.

import pandas as pd 
import re
from numpy import NaN

df = pd.read_csv('https://data.cityofnewyork.us/resource/rvhx-8trz.csv', nrows = 1000)

pattern = re.compile('^\$\d*\.\d{2}$')

#When we apply a function across rows of a dataframe, what gets passed into the function 
#is the row of the data, even though we only need a few values from the row, the 
#entire row will be passed into the function. 
#
#So our function will take two parameters, the row of data from our dataframe and 
#the pattern we will use to validate monetary values. 
#we can get the initial cost and total est fee values by slicing the row and we can
#use an if else statement to make sure the values are valid. 

def diff_money(row, pattern):
    icost = row['initial_cost']
    tef = row['total_est__fee']
    
    if bool(pattern.match(icost)) and bool(pattern.match(tef)):
        icost = icost.replace("$", "")
        tef = tef.replace("$", "")
        
        icost = float(icost)
        tef = float(tef)
        
        return icost - tef
    
    else:
        
        return(NaN)
        

df['diff'] = df.apply(diff_money, axis=1, pattern=pattern)

df1 = df[['initial_cost','total_est__fee', 'diff']]

# =============================================================================
# The tips dataset has been pre-loaded into a DataFrame called tips.
# It has a 'sex' column that contains the values 'Male' or 'Female'. 
# Your job is to write a function that will recode 'Female' to 0, 'Male' to 1, 
# and return np.nan for all entries of 'sex' that are neither 'Female' nor 'Male'
# =============================================================================

# Here, you will apply your function over the 'sex' column

# Define recode_gender()
def recode_gender(gender):

    # Return 0 if gender is 'Female'
    if gender == 'Female':
        return 0
    
    # Return 1 if gender is 'Male'
    elif gender == 'Male':
        return 1
        
    # Return np.nan    
    else:
        return np.nan

# Apply the function to the sex column
tips['recode'] = tips.sex.apply(recode_gender)

# =============================================================================
# Lambda functions
# =============================================================================

#Your job is to clean its 'total_dollar' column by removing the dollar sign. 
#You'll do this using two different methods: With the .replace() method, 
#and with regular expressions.


# Write the lambda function using replace
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))

# Write the lambda function using regular expressions. 
# Notice that because re.findall() returns a list, you have to slice it in order to access the actual value.
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])


# =============================================================================
# Duplicate and missing data
# =============================================================================

# drop duplicates
df = df.drop_duplicates()

# Drop missing values

df = df.dropna()

# Fill missing values with .fillna() method

tips_nan['sex'] = tips_nan['sex'].fillna('missing')

tips_nan[['total_bill', 'size']] = tips_nan[['total_bill', 'size']].fillna(0)

#Fill missing values with a test statistic

mean_value = tips_nan['tip'].mean() 

tips_nan['tip'] = tips_nan['tip'].fillna(mean_value)

# =============================================================================

# Create the new DataFrame: tracks
tracks = billboard[['year', 'artist', 'track', 'time']]

# Print info of tracks
print(tracks.info())

# Drop the duplicates: tracks_no_duplicates
tracks_no_duplicates = tracks.drop_duplicates()

# Print info of tracks
print(tracks_no_duplicates.info())

# =============================================================================
# Filling missing data
# =============================================================================

# Calculate the mean of the Ozone column: oz_mean
oz_mean = airquality['Ozone'].mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Ozone'] = airquality['Ozone'].fillna(oz_mean)

# Print the info of airquality
print(airquality.info())

# =============================================================================
# Testing with asserts
# =============================================================================

assert 1==1 

# Test a column

assert df.initial_cost.notnull().all()


## Testing a dataframe

# The first .all() method will return a True or False for each column, 
# while the second .all() method will return a single True or False

# Assert that there are no missing values
assert pd.notnull(df).all().all()

# Assert that all values are >= 0
assert (df >= 0).all().all()

# =============================================================================
# GAPMINDER DATA
# =============================================================================


gapminder = pd.read_csv('data/life_expectancy_years.csv')

# Exploratory analysis
print(gapminder.head())

print(gapminder.info())

print(gapminder.describe())

print(gapminder.shape)

# Visualizing your data

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Create the scatter plot
gapminder.plot(kind='scatter', x='1800', y='1899')

# Specify axis labels
plt.xlabel('Life Expectancy by Country in 1800')
plt.ylabel('Life Expectancy by Country in 1899')

# Specify axis limits
plt.xlim(20, 55)
plt.ylim(20, 55)

# Display the plot
plt.show()

# =============================================================================
# Thinking about the question at hand
# =============================================================================

def check_null_or_valid(row_data):
    """Function that takes a row of data,
    drops all missing values,
    and checks if all remaining values are greater than or equal to 0
    """
    no_na = row_data.dropna()
    numeric = pd.to_numeric(no_na)
    ge0 = numeric >= 0
    return ge0

# Check whether the first column is 'Life expectancy'
assert gapminder.columns[0] == 'country'

# Check whether the values in the row are valid
assert gapminder.iloc[:, 1:].apply(check_null_or_valid, axis=1).all().all()

# Check that there is only one instance of each country
assert gapminder['country'].value_counts()[0] == 1

#Specifically, index 0 of .value_counts() will contain the most frequently occuring value. 
#If this is equal to 1 for the 'Life expectancy' column, then you can be certain 
#that no country appears more than once in the data.

# =============================================================================
# Checking data types
# =============================================================================
df.dtypes

df['column'] = df['column'].to_numeric()

df['column'] = df['column'].astype(str)

# =============================================================================
# Additional calculations and saving your data
# =============================================================================

df['new_column'] = df['column_1'] + df['column_2']
 
df['new_column'] = df.apply(my_function, axis=1)

# Saving dataset to csv
df.to_csv['my_data.csv'] 

#Currently, the gapminder DataFrame has a separate column for each year. 
#What you want instead is a single column that contains the year, and a single 
#column that represents the average life expectancy for each year and country. 
#By having year in its own column, you can use it as a predictor variable in a 
#later analysis.


import pandas as pd
import numpy as np

# Melt gapminder: gapminder_melt
gapminder_melt = pd.melt(gapminder, id_vars='country', 
                         var_name = 'year', value_name = 'life_expectancy')

# Rename the columns

#gapminder_melt.columns = ['country', 'year', 'life_expectancy']

# Print the head of gapminder_melt
print(gapminder_melt.head())


# Convert the year column to numeric
gapminder_melt.year = pd.to_numeric(gapminder_melt['year'])

# Test if country is of type object
assert gapminder_melt.country.dtypes == np.object

# Test if year is of type int64
assert gapminder_melt.year.dtypes == np.int64

# Test if life_expectancy is of type float64
assert gapminder_melt.life_expectancy.dtypes == np.float64

# =============================================================================
# Looking at country spellings
# =============================================================================

# Create the series of countries: countries
countries = gapminder_melt['country']

# Drop all the duplicates from countries
countries = countries.drop_duplicates()

# Write the regular expression: pattern
#Use A-Za-z to match the set of lower and upper case letters, \. to 
#match periods, and \s to match whitespace between words.

pattern = '^[A-Za-z\.\s]*$'

# Create the Boolean vector: mask
mask = countries.str.contains(pattern)

# Invert the mask: mask_inverse. Invert the mask by placing a ~ before it
mask_inverse = ~ mask

# Subset countries using mask_inverse: invalid_countries
invalid_countries = countries.loc[mask_inverse]

# Print invalid_countries
print(invalid_countries)


# =============================================================================
# More data cleaning and processing: Dealing with missing data and duplicates
# =============================================================================

# Assert that country does not contain any missing values
assert pd.notnull(gapminder_melt.country).all()

# Assert that year does not contain any missing values
assert pd.notnull(gapminder_melt.year).all()

# Drop the missing values
gapminder_melt = gapminder_melt.dropna()

# Print the shape of gapminder
print(gapminder_melt.shape)

# =============================================================================
# Wrapping up
# =============================================================================

# Add first subplot
plt.subplot(2, 1, 1) 

# Create a histogram of life_expectancy
gapminder_melt.life_expectancy.plot(kind = 'hist')


# Group gapminder: gapminder_agg
gapminder_agg = gapminder_melt.groupby('year')['life_expectancy'].mean()

# Print the head of gapminder_agg
print(gapminder_agg.head())

# Print the tail of gapminder_agg
print(gapminder_agg.tail())

# Add second subplot
plt.subplot(2, 1, 2)

# Create a line plot of life expectancy per year
gapminder_agg.plot(kind = 'line')

# Add title and specify axis labels
plt.title('Life expectancy over the years')
plt.ylabel('Life expectancy')
plt.xlabel('Year')

# Display the plots
plt.tight_layout()
plt.show()

# Save both DataFrames to csv files
gapminder_melt.to_csv('data/gapminder_melt.csv')
gapminder_agg.to_csv('data/gapminder_agg.csv')











































































