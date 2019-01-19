#importing pandas
import pandas as pd
import numpy as np

#Reading in the dataset
apple = pd.read_csv('data/aapl.csv', index_col = 0)

# check how many rows and columns
apple.shape

#check the column names
apple.columns

# the apple.columns attribute is a pandas index
type(apple.columns)

# the apple.index attribute is a special kind of index called the Datetimeindex
type(apple.index)

# Dataframes can be sliced like numpy arrays

apple.iloc[:5,:]

apple.iloc[-5:,:]

apple.head(7)

apple.info()

#pandas dataframes slices also support broadcasting

# This selects every 3 row starting from 0 of the last column and assigns them to nan
apple.iloc[::3, -1] = np.nan

apple.head(6)

# The columns of a dataframe are themselves a specialized pandas structure called a series

low = apple['Low']
type(low)

# =============================================================================
# Notice that the Series extracted has its own head() method and inherits its 
# name attribute from the dataframe column.
# 
# To extract the numerical entries from the series, use the values attribute. 
# The data in the Series actually form a numpy array which is what the values attribute yields. 
# 
# A pandas series is a one dimensional labelled Numpy array and a dataframe is a 2 dimensional 
# labelled array whose columns are series. 
# =============================================================================

# =============================================================================
# NumPy and pandas working together
# Pandas depends upon and interoperates with NumPy, the Python library for fast 
# numeric array computations. For example, you can use the DataFrame 
# attribute .values to represent a DataFrame df as a NumPy array. 
# You can also pass pandas data structures to NumPy methods. 
# In this exercise, we have imported pandas as pd and loaded world population 
# data every 10 years since 1960 into the DataFrame df. 
# This dataset was derived from the one used in the previous exercise.
# =============================================================================

year = [1960, 1970, 1980, 1990, 2000, 2010 ]
population = [3.034971e+09, 3.684823e+09, 4.436590e+09, 5.282716e+09, 6.115974e+09, 6.924283e+09]

world = {'population':population}

world_df = pd.DataFrame.from_dict(world)

world_df.index = year

# Create array of DataFrame values: np_vals
np_vals = world_df.values

# Create new array of base 10 logarithm values: np_vals_log10
np_vals_log10 = np.log10(np_vals)

# Create array of new DataFrame by passing df to np.log10(): df_log10
df_log10 = np.log10(world_df)


# Print original and new data containers
[print(x, 'has type', type(eval(x))) for x in ['np_vals', 'np_vals_log10', 'world_df', 'df_log10']]


# =============================================================================
# Building DataFrames from scratch
# =============================================================================
cities = ['Austin', 'Dallas', 'Austin', 'Dallas']
signups = [7, 12, 3, 5]
visitors = [139, 237, 326, 456]
weekdays = ['Sun', 'Sun', 'Mon', 'Mon']

# creating labels
list_labels = ['city', 'signups', 'visitors', 'weekday']

list_col = [cities, signups, visitors, weekdays]

zipped = list(zip(list_labels, list_col))

data = dict(zipped)

users = pd.DataFrame(data)

# =============================================================================
# # BROADCASTING
# =============================================================================

users['fees'] = 0 #Broadcast to the entire column

# we can change the columns and index labels using the columns and index attribute

result.columns = ['height', 'sex'] # has to be of suitable length

# Zip the 2 lists together into one list of (key,value) tuples: zipped
list_keys = ['Country', 'Total']
list_values = [['United States', 'Soviet Union', 'United Kingdom'], [1118, 473, 273]]
zipped = list(zip(list_keys, list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)

# =============================================================================
# Labeling your data
# You can use the DataFrame attribute df.columns to view and assign new 
# string labels to columns in a pandas DataFrame.
# =============================================================================

# Build a list of labels: list_labels
list_labels = ['year','artist','song','chart weeks']

# Assign the list of labels to the columns attribute: df.columns
df.columns = list_labels

# =============================================================================
# Building DataFrames with broadcasting
# You can implicitly use 'broadcasting', a feature of NumPy, 
# when creating pandas DataFrames. In this exercise, you're going to create a 
# DataFrame of cities in Pennsylvania that contains the city name in one column
#  and the state name in the second. We have imported the names of 15 cities as 
#  the list cities
# =============================================================================

cities = ['Manheim', 'Preston park', 'Biglerville', 'Indiana', 'Curwensville', 
          'Crown', 'Harveys lake', 'Mineral springs', 'Cassville', 'Hannastown', 
          'Saltsburg', 'Tunkhannock', 'Pittsburgh', 'Lemasters', 'Great bend']


# Make a string with the value 'PA': state
state = 'PA'

# Construct a dictionary: data
data = {'state': state, 'city': cities}

# Construct a DataFrame from dictionary data: df
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

# =============================================================================
# Importing & exporting data
# =============================================================================

filepath = 'data/SN_d_tot_V2.0.csv'

sunspots = pd.read_csv(filepath, sep = ';', header = None)

sunspots.info()

# let use iloc to view a slice in the middle of the data

sunspots.iloc[10:20,:]

# =============================================================================
# Column 1-3: Gregorian calendar date - year, month, day
# Column 4: Date in fraction of year.
# Column 5: Daily total sunspot number.
# A value of -1 indicates that no number is available for that day (missing value).
# Column 6: Daily standard deviation of the input sunspot numbers from individual stations.
# Column 7: Number of observations used to compute the daily value.
# Column 8: Definitive/provisional indicator. 
# '1' indicates that the value is definitive. '0' indicates that the value is still provisional.
# 
# =============================================================================

col_names = ['year', 'month', 'day', 'dec_date', 'sunspots', 'std', 'observations', 'definite']

sunspots = pd.read_csv(filepath, sep = ';', header = None, 
                       names = col_names,
                       na_values = '  -1') 
# The na_values would work but will affect other columns as well. We can use the different
# method below to select different columns and their different NA value coding

sunspots = pd.read_csv(filepath, sep = ';', header = None, 
                       names = col_names,
                       na_values = {'sunspots':['  -1']},
                       parse_dates = [[0,1,2]]) # since these columns contain year, month and day

sunspots.iloc[10:20,:]

sunspots.index = sunspots['year_month_day']
sunspots.index.name = 'date'

sunspots.info()

# Trimming redundant columns

cols = ['sunspots', 'definite']
sunspots = sunspots[cols]

# Writing files. Saving csv. exporting files

out_csv = 'data/sunspots.csv'
sunspots.to_csv(out_csv)

out_tsv = 'data/sunspots.tsv'
sunspots.to_csv(out_tsv, sep = '\t')

out_xlsx = 'data/sunspots.xlsx'
sunspots.to_excel(out_xlsx)


# Read in the file: df1
df1 = pd.read_csv(data_file)

# Create a list of the new column labels: new_labels
new_labels = ['year', 'population']

# Read in the file, specifying the header and names parameters: df2
df2 = pd.read_csv(data_file, header=0, names=new_labels)

# Print both the DataFrames
print(df1)
print(df2)

# Read the raw file as-is: df1
df1 = pd.read_csv('/usr/local/share/datasets/messy_stock_data.tsv')

# Print the output of df1.head()
print(df1.head())

# Read in the file with the correct parameters: df2
df2 = pd.read_csv('/usr/local/share/datasets/messy_stock_data.tsv',
                  delimiter=' ', 
                  header=3, 
                  comment='#')

# Print the output of df2.head()
print(df2.head())

# Save the cleaned up DataFrame to a CSV file without the index
df2.to_csv('tmp_clean_stock_data.csv', index=False)

# Save the cleaned up DataFrame to an excel file without the index
df2.to_excel('file_clean.xlsx', index=False)


# =============================================================================
# Plotting with pandas
# =============================================================================


































































































