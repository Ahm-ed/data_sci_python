#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: amin
"""

# =============================================================================
# IPython, which is running on DataCamp's servers, has a bunch of cool commands, 
# including its magic commands. For example, starting a line with ! gives you 
# complete system shell access. This means that the IPython magic command ! ls will 
# display the contents of your current directory. Your task is to use the IPython 
# magic command ! ls to check out the contents of your current directory and answer 
# the following question: which of the following files is in your working directory?
# =============================================================================

### REading a file 
filename = 'data/tweets.txt'
file = open(filename, mode='r') # 'r' is to read
text = file.read()
file.close() ## this prevents accidental changes

### Writing to a file

filename = 'huck_finn.txt'
file = open(filename, mode='w') # 'w' is to write
file.close() 

## Context manager with - best practice

with open('data/tweets.txt', 'r') as file:
    print(file.read()) 
    
# =============================================================================
# In this exercise, you'll be working with the file moby_dick.txt. 
# It is a text file that contains the opening sentences of Moby Dick, 
# one of the great American novels! Here you'll get experience opening a text 
# file, printing its contents to the shell and, finally, closing it.
# =============================================================================

# Open a file: file
file = open('moby_dick.txt', 'r')

# Print it
print(file.read())

# Check whether file is closed
print(file.closed)

# Close file
file.close()

# Check whether file is closed
print(file.closed)

# =============================================================================
# Importing text files line by line
# For large files, we may not want to print all of their content to the shell: 
# you may wish to print only the first few lines. Enter the readline() method, 
# which allows you to do this. When a file called file is open, you can print 
# out the first line by executing file.readline(). If you execute the same command 
# again, the second line will print, and so on.
# 
# In the introductory video, Hugo also introduced the concept of a context manager. 
# He showed that you can bind a variable file by using a context manager construct:
# =============================================================================

# Read & print the first 3 lines
with open('data/tweets.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())

#You can print the Zen of Python in your shell by typing import this into it! 
#You're going to do this now and the 5th aphorism (line) will say something of particular interest.

# =============================================================================
# Flat files
# ● Text files containing records
# ● That is, table data
# ● Record: row of fields or attributes
# ● Column: feature or attribute
# =============================================================================

# =============================================================================
# Using NumPy to import flat files
# In this exercise, you're now going to load the MNIST digit recognition dataset 
# using the numpy function loadtxt() and see just how easy it can be:
# 
# The first argument will be the filename.
# The second will be the delimiter which, in this case, is a comma.
# You can find more information about the MNIST dataset here on the webpage of 
# Yann LeCun, who is currently Director of AI Research at Facebook and Founding 
# Director of the NYU Center for Data Science, among many other things.
# =============================================================================

# Import package
import numpy as np
import matplotlib.pyplot as plt

# Assign filename to variable: file
file = 'data/mnist_train.csv'

# Load file as array: digits
digits = np.loadtxt(file, delimiter=',')

# Print datatype of digits
print(type(digits))

# Select and reshape a row
im = digits[21, 1:]
im_sq = np.reshape(im, (28, 28))

# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.show()

# =============================================================================
# Customizing your NumPy import
# What if there are rows, such as a header, that you don't want to import? 
# What if your file has a delimiter other than a comma? What if you only 
# wish to import particular columns?
# 
# There are a number of arguments that np.loadtxt() takes that you'll find 
# useful: delimiter changes the delimiter that loadtxt() is expecting, 
# for example, you can use ',' and '\t' for comma-delimited and tab-delimited 
# respectively; skiprows allows you to specify how many rows (not indices) you 
# wish to skip; usecols takes a list of the indices of the columns you wish to keep.
# =============================================================================


# Assign the filename: file
file = 'MNIST.txt'

# Load the data: data
data = np.loadtxt(file, delimiter=' ', skiprows=1, usecols=[0, 2]) # USE '\t' for tab delimited

# Print data
print(data)

# Assign filename: file
file = 'seaslug.txt'

# Import file: data
data = np.loadtxt(file, delimiter='\t', dtype=str)

# Print the first element of data
print(data[0])

# Import data as floats and skip the first row: data_float
data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)

# Print the 10th element of data_float
print(data_float[9])

# Plot a scatterplot of the data
plt.scatter(data_float[:, 0], data_float[:, 1])
plt.xlabel('time (min.)')
plt.ylabel('percentage of larvae')
plt.show()

# =============================================================================
# Working with mixed datatypes (1)
# Much of the time you will need to import datasets which have different 
# datatypes in different columns; one column may contain strings and another 
# floats, for example. The function np.loadtxt() will freak at this. 
# There is another function, np.genfromtxt(), which can handle such structures. 
# If we pass dtype=None to it, it will figure out what types each column should be.
# Because the data are of different types, data is an object called a structured array.
# =============================================================================

#Accessing rows and columns of structured arrays is super-intuitive: to get 
#the ith row, merely execute data[i] and to get the column with name 'Fare', execute data['Fare']

data = np.genfromtxt('data/winequality-red.csv', 
                     delimiter=';', 
                     names=True, 
                     dtype=None,
                     encoding = None)

print(np.shape(data))

# =============================================================================
# Working with mixed datatypes (2)
# You have just used np.genfromtxt() to import data containing mixed datatypes. 
# There is also another function np.recfromcsv() that behaves similarly to np.genfromtxt(), 
# except that its default dtype is None and default delimiter is ','
# 
# =============================================================================

# Assign the filename: file
file = 'titanic.csv'

# Import file using np.recfromcsv: d
d = np.recfromcsv(file)

# Print out first three entries of d
print(d[:3])

# =============================================================================
# Using pandas to import flat files as DataFrames (1)
# =============================================================================

# Import pandas as pd
import pandas as pd

# Assign the filename: file
file = 'titanic.csv'

# Read the file into a DataFrame: df
df = pd.read_csv(file)

# View the head of the DataFrame
print(df.head())

## Convert pandas dataframe to array
data_array = df.values

# Read the first 5 rows of the file into a DataFrame: data
data = pd.read_csv(file, nrows= 5, header = None)

# =============================================================================
# The pandas package is also great at dealing with many of the issues you will 
# encounter when importing data as a data scientist, such as comments occurring 
# in flat files, empty lines and missing values. Note that missing values are 
# also commonly referred to as NA or NaN
# 
# =============================================================================

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Assign filename: file
file = 'titanic_corrupt.txt'

# Import file: data
data = pd.read_csv(file, sep='\t', comment='#', na_values='Nothing')

# Print the head of the DataFrame
print(data.head())

# Plot 'Age' variable in a histogram
pd.DataFrame.hist(data[['Age']])
plt.xlabel('Age (years)')
plt.ylabel('count')
plt.show()

# =============================================================================
# In Chapter 1, you learned how to use the IPython magic command ! ls to explore
#  your current working directory. You can also do this natively in Python using 
#  the library os, which consists of miscellaneous operating system interfaces.
# 
# The first line of the following code imports the library os, the second line 
# stores the name of the current directory in a string called wd and the third 
# outputs the contents of the directory in a list to the shell.
# =============================================================================

import os
wd = os.getcwd()
os.listdir(wd)

# =============================================================================
# Loading a pickled file
# There are a number of datatypes that cannot be saved easily to flat files, 
# such as lists and dictionaries. If you want your files to be human readable, 
# you may want to save them as text files in a clever manner. 
# JSONs, which you will see in a later chapter, are appropriate for Python dictionaries.
# 
# However, if you merely want to be able to import them into Python, you can 
# serialize them. All this means is converting the object into a sequence of 
# bytes, or a bytestream.
# 
# In this exercise, you'll import the pickle package, open a previously 
# pickled data structure from a file and load it
# =============================================================================

# Import pickle package
import pickle 

# Open pickle file and load data: d
with open('data.pkl', 'rb') as file: #rb means read and binary (meaning can be read by only a computer)
    d = pickle.load(file)

# Print d
print(d)

# Print datatype of d
print(type(d))

# =============================================================================
# Here, you'll learn how to use pandas to import Excel spreadsheets and how to 
# list the names of the sheets in any loaded .xlsx file.
# 
# Recall from the video that, given an Excel file imported into a variable spreadsheet, 
# you can retrieve a list of the sheet names using the attribute spreadsheet.sheet_names
# =============================================================================

# Import pandas
import pandas as pd

# Assign spreadsheet filename: file
file = 'PRIO Battle Deaths Dataset 3.1.xls'

# Load spreadsheet: xl
xl = pd.ExcelFile(file)

# Print sheet names
print(xl.sheet_names)

# =============================================================================
# In this exercise, you'll learn how to import any given sheet of your loaded
#  .xlsx file as a DataFrame. You'll be able to do so by specifying either 
#  the sheet's name or its index.
# =============================================================================

# Load a sheet into a DataFrame by name: df1
df1 = xl.parse('bdonly')

# Print the head of the DataFrame df1
print(df1.head())

# Load a sheet into a DataFrame by index: df2
df2 = xl.parse(0)

# Print the head of the DataFrame df2
print(df2.head())

# =============================================================================
# As before, you'll use the method parse(). 
# This time, however, you'll add the additional arguments skiprows, names and parse_cols. 
# These skip rows, name the columns and designate which columns to parse, respectively. 
# All these arguments can be assigned to lists containing the specific row numbers, 
# strings and column numbers, as appropriate.
# =============================================================================

# Parse the first sheet and rename the columns: df1 
df1 = xl.parse(0, skiprows=[0], names=['Country', 'AAM due to War (2002)'])

# Print the head of the DataFrame df1
print(df1.head())

# Parse the first column of the second sheet and rename the column: df2
df2 = xl.parse(1, parse_cols=[0], skiprows=[0], names=['Country'])

# Print the head of the DataFrame df2
print(df2.head())

# =============================================================================
# In this exercise, 
# you'll figure out how to import a SAS file as a DataFrame using SAS7BDAT and pandas
# =============================================================================

# Import sas7bdat package
from sas7bdat import SAS7BDAT

# Save file to a DataFrame: df_sas
with SAS7BDAT('data/ais.sas7bdat') as file:
    df_sas = file.to_data_frame()

# Print head of DataFrame
print(df_sas.head())

# Plot histogram of DataFrame features (pandas and pyplot already imported)
pd.DataFrame.hist(df_sas[['P']])
plt.ylabel('count')
plt.show()

# Using read_stata to import Stata files

# Import pandas
import pandas as pd

# Load Stata file into a pandas DataFrame: df
df = pd.read_stata('disarea.dta')

# Print the head of the DataFrame df
print(df.head())

# Plot histogram of one column of the DataFrame
pd.DataFrame.hist(df[['disa10']])
plt.xlabel('Extent of disease')
plt.ylabel('Number of countries')
plt.show()

# =============================================================================
# Using File to import HDF5 files
# The h5py package has been imported in the environment and the file LIGO_data.hdf5 is 
# loaded in the object h5py_file
# =============================================================================

import h5py
filename = 'H-H1_LOSC_4_V1-815411200-4096.hdf5'

data = h5py.File(filename, 'r') # 'r' is to read
print(type(data)) 

### The structure of HDF5 files

for key in data.keys():
    print(key) 

### If you already know the structure
for key in data['meta'].keys():
    print(key) 

print(data['meta']['Description'].value, data['meta']['Detector'].value)

# =============================================================================

# Import packages
import numpy as np
import h5py

# Assign filename: file
file = 'LIGO_data.hdf5'

# Load file: data
data = h5py.File(file, 'r')

# Print the datatype of the loaded file
print(type(data))

# Print the keys of the file
for key in data.keys():
    print(key)

# Get the HDF5 group: group
group = data['strain']

# Check out keys of group
for key in group.keys():
    print(key)

# Set variable equal to time series data: strain
#Assign to the variable strain the values of the time series data #data['strain']['Strain'] using the attribute .value
strain = data['strain']['Strain'].value

# Set number of time points to sample: num_samples
num_samples = 10000

# Set time vector
time = np.arange(0, 1, 1/num_samples)

# Plot data
plt.plot(time, strain[:num_samples])
plt.xlabel('GPS Time (s)')
plt.ylabel('strain')
plt.show()

# =============================================================================
# ### MATLAB files
# SciPy to the rescue!
# ● scipy.io.loadmat() - read .mat files
# ● scipy.io.savemat() - write .mat files
# =============================================================================

# Import package
import scipy.io

# Load MATLAB file: mat
mat = scipy.io.loadmat('albeck_gene_expression.mat')

# Print the datatype type of mat
print(type(mat))


# Print the keys of the MATLAB dictionary
for key in mat.keys():
    print(key)

## or you can use this 
    
print(mat.keys())

# Print the type of the value corresponding to the key 'CYratioCyt'
print(type(mat['CYratioCyt']))

# Print the shape of the value corresponding to the key 'CYratioCyt'
print(np.shape(mat['CYratioCyt']))

print(mat['CYratioCyt'].shape)

# Subset the array and plot it
data = mat['CYratioCyt'][25, 5:]
fig = plt.figure()
plt.plot(data)
plt.xlabel('time (min.)')
plt.ylabel('normalized fluorescence (measure of expression)')
plt.show()

#### CHAPTER 3

# =============================================================================
# Introduction to relational databases
# 
# =============================================================================

# Import necessary module
from sqlalchemy import create_engine

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Save the table names to a list: table_names
table_names = engine.table_names()

# Print the table names to the shell
print(table_names)


# =============================================================================
# Workflow of SQL querying
# ● Import packages and functions
# ● Create the database engine
# ● Connect to the engine
# ● Query the database
# ● Save query results to a DataFrame
# ● Close the connection
# =============================================================================

# =============================================================================
# Your first SQL query
# =============================================================================

from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///Northwind.sqlite')

con = engine.connect()

rs = con.execute("SELECT * FROM Orders")

df = pd.DataFrame(rs.fetchall()) # Fetch all gets all rows

df.columns = rs.keys()
 
con.close() # close the connection

# =============================================================================
# Using the context manager
# =============================================================================
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')

with engine.connect() as con:
    rs = con.execute("SELECT OrderID, OrderDate, ShipName FROM Orders")
    ## GEt only 5 rows 
    df = pd.DataFrame(rs.fetchmany(size=5))
    df.columns = rs.keys()


# =============================================================================
# PRACTICE
# =============================================================================

# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Open engine connection: con
con = engine.connect()

# Perform query: rs
rs = con.execute("SELECT * FROM Album")

# Save results of the query to DataFrame: df
df = pd.DataFrame(rs.fetchall())

# Close connection
con.close()

# Print head of DataFrame df
print(df.head())

# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT LastName, Title FROM Employee")
    df = pd.DataFrame(rs.fetchmany(size = 3))
    df.columns = rs.keys()

# Print the length of the DataFrame df
print(len(df))

# Print the head of the DataFrame df
print(df.head())

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute('SELECT * FROM Employee WHERE EmployeeId  >= 6')
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

# Print the head of the DataFrame df
print(df.head())

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Open engine in context manager
with engine.connect() as con:
    rs = con.execute('SELECT * FROM Employee ORDER BY Birthdate')
    df = pd.DataFrame(rs.fetchall())

    # Set the DataFrame's column names
    df.columns = rs.keys()

# Print head of DataFrame
print(df.head())

# =============================================================================
# Pandas and The Hello World of SQL Queries!
# =============================================================================

# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Execute query and store records in DataFrame: df
df = pd.read_sql_query('SELECT * FROM Album', engine)

# Print head of DataFrame
print(df.head())

# Open engine in context manager and store query result in df1
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Album")
    df1 = pd.DataFrame(rs.fetchall())
    df1.columns = rs.keys()

# Confirm that both methods yield the same result
print(df.equals(df1))

# =============================================================================

# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

# Execute query and store records in DataFrame: df
df = pd.read_sql_query('SELECT * FROM Employee WHERE EmployeeID >= 6 ORDER BY Birthdate', engine)

# Print head of DataFrame
print(df.head())

# =============================================================================
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute('SELECT Title, Name FROM Album INNER JOIN Artist ON Album.ArtistID = Artist.ArtistID')
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

# Print head of DataFrame df
print(df.head())


# Execute query and store records in DataFrame: df
df = pd.read_sql_query('SELECT * FROM PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackId WHERE Milliseconds < 250000', engine)

# Print head of DataFrame
print(df.head())
