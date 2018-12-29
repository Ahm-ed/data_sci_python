#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: amin
"""

# =============================================================================
# Iterators vs Iterables
# Let's do a quick recall of what you've learned about iterables and iterators. 
# Recall from the video that an iterable is an object that can return an iterator,
#  while an iterator is an object that keeps state and produces the next value 
#  when you call next() on it
# =============================================================================

# =============================================================================
# ### Iterable is an object that has an associated iter() method. Once this 
# iter method applied to an iterable, an iterator object is created. 
# 
# An iterator is an object that has an associated next() method that produces 
# the consecutive values 
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

flash1 = ['jay garrick', 'barry allen', 'wally west', 'bart allen']

# We can iterate over a list using a for loop

for names in flash1:
    print(names) 
    
# We can iterate over a string using a for loop
    
for letters in "Amin Yakubu":
    print(letters)
    
#We can iterate over a range object using a for loop

for i in range(4):
    print(i)
    
# =============================================================================
# Iterators vs. iterables
# ● Iterable
# ● Examples: lists, strings, dictionaries, file connections
# ● An object with an associated iter() method
# ● Applying iter() to an iterable creates an iterator
# ● Iterator
# ● Produces next value with next()
# =============================================================================

word = 'Data'

it = iter(word)

next(it)
next(it)

# Iterating at once with *

print(*it) 


# Iterating over dictionaries


pythonistas = {'hugo': 'bowne-anderson', 'francis':'castro'}

for key, value in pythonistas.items():
    print(key, value)

# Iterating over file connections

file = open('file.txt') 

it = iter(file) 
print(next(it))

# Create a list of strings: flash
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']

# Print each list item in flash using a for loop
for person in flash:
    print(person)


# Create an iterator for flash: superspeed
superspeed = iter(flash)

# Print each item from the iterator
print(next(superspeed))
print(next(superspeed))
print(next(superspeed))
print(next(superspeed))

# =============================================================================
# Iterating over iterables (2)
# One of the things you learned about in this chapter is that not all iterables 
# are actual lists. A couple of examples that we looked at are strings and the 
# use of the range() function. In this exercise, we will focus on the range() function.
# =============================================================================

# Create an iterator for range(3): small_value
small_value = iter(range(3))

# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))

# Loop over range(3) and print the values
for num in range(3):
    print(num)


# Create an iterator for range(10 ** 100): googol
googol = iter(range(10**100))

# Print the first 5 values from googol
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))

# =============================================================================
# You've been using the iter() function to get an iterator object, as well as 
# the next() function to retrieve the values one by one from the iterator object.
# 
# There are also functions that take iterators as arguments. 
# For example, the list() and sum() functions return a list and the sum of 
# elements, respectively.
# 
# In this exercise, you will use these functions by passing an iterator 
# from range() and then printing the results of the function calls.
# =============================================================================

# Create a range object: values
values = range(10, 21)

# Print the range object
print(values)

# Create a list of integers: values_list
values_list = list(values)

# Print values_list
print(values_list)

# Get the sum of values: values_sum
values_sum = sum(values)

# Print values_sum
print(values_sum)

# =============================================================================
# The enumerate function will allow us to add a counter to any iterable while 
# the second function, zip, will allow us to stitch together an arbitrary number 
# of iterables. enumerate() takes any iterable as an argument, such as a list, and
# returns a special enumerator object which consists of pairs containing the element
# of the original iterable, along with their index within the iterable. 
# We can then use the function list to turn this enumerate object into a list of
# tuples and print it. The enumerate object itself is an iterable, so we can loop
# over it while unpacking its element using the clause for index value  in enumerate(avengers)
# =============================================================================

avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
 
e = enumerate(avengers)

print(type(e)) 
e_list = list(e)

print(e_list)

## Enumerate and unpack
for index, value in enumerate(avengers):
    print(index, value) 
  
### We can start at 10 instead of the default 0 index
for index, value in enumerate(avengers, start=10):
    print(index, value) 

### Using zip()
    
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
names = ['barton', 'stark', 'odinson', 'maximoff'] 
z = zip(avengers, names)
print(type(z))
z_list = list(z) 
print(z_list) 


### Zip and unpack
for z1, z2 in zip(avengers, names):
    print(z1, z2) 

### Another way
z = zip(avengers, names)
print(*z) 

#Recall that enumerate() returns an enumerate object that produces a 
#sequence of tuples, and each of the tuples is an index-value pair.

# Create a list of strings: mutants
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pryde']

# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)

# Unpack and print the tuple pairs
for index1, value1 in enumerate(mutants):
    print(index1, value1)

# Change the start index
for index2, value2 in enumerate(mutants, start =1):
    print(index2, value2)

# =============================================================================
# Another interesting function that you've learned is zip(), which takes any 
# number of iterables and returns a zip object that is an iterator of tuples. 
# If you wanted to print the values of a zip object, you can convert it into a 
# list and then print it. Printing just a zip object will not return the values 
# unless you unpack it first. In this exercise, you will explore this for yourself.
# =============================================================================
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pryde']

aliases = ['prof x', 'iceman', 'nightcrawler', 'magneto', 'shadowcat']

powers = ['telepathy',
 'thermokinesis',
 'teleportation',
 'magnetokinesis',
 'intangibility']

# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants, aliases, powers))

# Print the list of tuples
print(mutant_data)

# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants, aliases, powers)

# Print the zip object
print(mutant_zip)

# Unpack the zip object and print the tuple values
for value1, value2, value3 in zip(mutants, aliases, powers):
    print(value1, value2, value3)

# =============================================================================
# There is no unzip function for doing the reverse of what zip() does. 
# We can, however, reverse what has been zipped together by using zip() with a 
# little help from *! * unpacks an iterable such as a list or a tuple into positional 
# arguments in a function call.
# =============================================================================


# Create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# Print the tuples in z1 by unpacking with *
print(*z1)

# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1)

# Check if unpacked tuples are equivalent to original tuples
print(result1 == mutants)
print(result2 == powers)


# =============================================================================
# Processing large amounts of Twitter data
# Sometimes, the data we have to process reaches a size that is too much for a 
# computer's memory to handle. This is a common problem faced by data scientists. 
# A solution to this is to process an entire data source chunk by chunk, 
# instead of a single go all at once
# 
# =============================================================================

result = [] 
for chunk in pd.read_csv('data.csv', chunksize=1000):
    result.append(sum(chunk['x']))

total = sum(result) 
print(total)

## OR

total = 0

for chunk in pd.read_csv('data.csv', chunksize=1000):
    total += sum(chunk['x'])
print(total) 

# =============================================================================

# Initialize an empty dictionary: counts_dict
counts_dict ={}

# Iterate over the file chunk by chunk
for chunk in pd.read_csv("tweets.csv", chunksize= 10):

    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

# Print the populated dictionary
print(counts_dict)

# =============================================================================
# It's good to know how to process a file in smaller, 
# more manageable chunks, but it can become very tedious having to write and 
# rewrite the same code for the same task each time. In this exercise, you will 
# be making your code more reusable by putting your work in the last exercise in 
# a function definition
# =============================================================================

# Define count_entries()
def count_entries(csv_file, c_size, colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file, chunksize= c_size):

        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict

# Call count_entries(): result_counts
result_counts = count_entries('tweets.csv', 10, 'lang')

# Print result_counts
print(result_counts)

# =============================================================================
# ### CHAPTER 2
# ### LIST COMPREHENSIONS
# 
# =============================================================================

nums = [12, 8, 21, 3, 16]

## to do some operation to this, we could use a for loop
new_nums = []

for i in nums:
    new_nums.append(i + 1)
    
print(new_nums)

### But this method is inefficient especially for large data

#This is the general syntax of a list comprehension
#The syntax is as follows: within square brackets, you write the values you wish to create, 
#otherwise known as the output expression, followed by the for clause referencing the 
#original list. 

#Components of a list comprehension
#● Iterable
#● Iterator variable (represent members of iterable)
#● Output expression

## The code above will be 

new_nums = [i + 1 for i in nums]

#You can write a list comprehension over any iterable

result = [i for i in range(11)]
print(result)

# List comprehensions in place of nested for loops 
pairs_1 = []

for num1 in range(0, 2):
    for num2 in range(6, 8):
        pairs_1.append(num1, num2)

print(pairs_1)

pairs_2 = [(num1, num2) for num1 in range(0, 2) for num2
in range(6, 8)] 

print(pairs_2)

#list comprehension that produces a list of the first character of each string in doctor

doctor = ['house', 'cuddy', 'chase', 'thirteen', 'wilson']

first_letter = [doc[0] for doc in doctor]
print(first_letter)

#list comprehension that produces a list of the squares of the numbers ranging from 0 to 9

# Create list comprehension: squares
squares = [i ** 2 for i in range(10)]

print(squares)

#Let's step aside for a while from strings. 
#One of the ways in which lists can be used are in representing 
#multi-dimension objects such as matrices. Matrices can be represented 
#as a list of lists in Python. For example a 5 x 5 matrix with values 0 to 4 
#in each row can be written as:

matrix = [[0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4]]

# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]

# Print the matrix
for row in matrix:
    print(row)

#Using conditionals in comprehensions (1)

#An interesting mechanism in list comprehensions is that you can also create 
#lists with values that meet only a certain condition. One way of doing this 
#is by using conditionals on iterator variables. In this exercise, you will do exactly that!
#
#[ output expression for iterator variable in iterable if predicate expression ]

adv_com =[num ** 2 for num in range(10) if num % 2 == 0] 

print(adv_com)

adv = [num ** 2 if num % 2 == 0 else 0 for num in range(10)] 
print(adv)

#Dict comprehensions
#● Create dictionaries
#● Use curly braces {} instead of brackets []

pos_neg = {num: -num for num in range(9)} 
print(pos_neg)

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member for member in fellowship if len(member) >= 7]

# Print the new list
print(new_fellowship)

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member if len(member)>= 7 else '' for member in fellowship]

# Print the new list
print(new_fellowship)


# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create dict comprehension: new_fellowship
new_fellowship = {member:len(member) for member in fellowship}

# Print the new list
print(new_fellowship)


# =============================================================================
# ### GENERATORS
# A list comprehension produces a list as output, a generator produces a generator object.
# A generator object does not store the list in memory. It does not construct the list 
# but it is an object we can iterate over to produce elements of the list as required. 
# 
# List comprehensions vs. generators
# ● List comprehension - returns a list
# ● Generators - returns a generator object
# ● Both can be iterated over
# 
# =============================================================================
# Use ( ) instead of [ ]

gen = (2 * num for num in range(10)) 
print(gen)
 
result = (num for num in range(6)) 
 
for num in result:
    print(num) 
    
## Or
    
print(list(result)) 

result = (num for num in range(6)) 
print(next(result)) 
print(next(result)) 
print(next(result)) 
print(next(result)) 

# Conditionals in generator expressions
even_nums = (num for num in range(10) if num % 2 == 0) 

## generator functions 

# =============================================================================
# Generator functions
# ● Produces generator objects when called
# ● Defined like a regular function - def
# ● Yields a sequence of values instead of returning a single
# value
# ● Generates a value with yield keyword
# =============================================================================

def num_sequence(n):
    """Generate values from 0 to n."""
    i = 0
    while i < n:
        yield i
        i += 1

result = num_sequence(5) 
print(result)


# Create generator object: result
result = (num for num in range(31))

# Print the first 5 values
print(next(result))
print(next(result))
print(next(result))
print(next(result))
print(next(result))

# Print the rest of the values
for value in result:
    print(value)

#Write a generator expression that will generate the lengths of each string in 
#lannister. Use person as the iterator variable. Assign the result to lengths.

# Create a list of strings: lannister
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Create a generator object: lengths
lengths = (len(person) for person in lannister)

# Iterate over and print the values in lengths
for value in lengths:
    print(value)

# Let's use a function to do it
    
# Create a list of strings
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Define generator function get_lengths
def get_lengths(input_list):
    """Generator function that yields the
    length of the strings in input_list."""

    # Yield the length of a string
    for person in input_list:
        yield len(person)

# Print the values generated by get_lengths()
for value in get_lengths(lannister):
    print(value)


# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time]

# Print the extracted times
print(tweet_clock_time)

# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time if entry[17:19] == '19']

# Print the extracted times
print(tweet_clock_time)


### CASE STUDY
feature_names = ['CountryName',
                 'CountryCode',
                 'IndicatorName',
                 'IndicatorCode',
                 'Year',
                 'Value']

row_vals = ['Arab World',
            'ARB',
            'Adolescent fertility rate (births per 1,000 women ages 15-19)',
            'SP.ADO.TFRT',
            '1960',
            '133.56090740552298']

# Zip lists: zipped_lists
zipped_lists = zip(feature_names, row_vals)

# Create a dictionary: rs_dict
rs_dict = dict(zipped_lists)

# Print the dictionary
print(rs_dict)

## Let's write a function for the above

# Define lists2dict()
def lists2dict(list1, list2):
    """Return a dictionary where list1 provides
    the keys and list2 provides the values."""

    # Zip lists: zipped_lists
    zipped_lists = zip(list1, list2)

    # Create a dictionary: rs_dict
    rs_dict = dict(zipped_lists)

    # Return the dictionary
    return rs_dict

# Call lists2dict: rs_fxn
rs_fxn = lists2dict(feature_names, row_vals)

# Print rs_fxn
print(rs_fxn)

row_lists = [['Arab World',
  'ARB',
  'Adolescent fertility rate (births per 1,000 women ages 15-19)',
  'SP.ADO.TFRT',
  '1960',
  '133.56090740552298'],
 ['Arab World',
  'ARB',
  'Age dependency ratio (% of working-age population)',
  'SP.POP.DPND',
  '1960',
  '87.7976011532547'],
 ['Arab World',
  'ARB',
  'Age dependency ratio, old (% of working-age population)',
  'SP.POP.DPND.OL',
  '1960',
  '6.634579191565161'],
 ['Arab World',
  'ARB',
  'Age dependency ratio, young (% of working-age population)',
  'SP.POP.DPND.YG',
  '1960',
  '81.02332950839141'],
 ['Arab World',
  'ARB',
  'Arms exports (SIPRI trend indicator values)',
  'MS.MIL.XPRT.KD',
  '1960',
  '3000000.0'],
 ['Arab World',
  'ARB',
  'Arms imports (SIPRI trend indicator values)',
  'MS.MIL.MPRT.KD',
  '1960',
  '538000000.0'],
 ['Arab World',
  'ARB',
  'Birth rate, crude (per 1,000 people)',
  'SP.DYN.CBRT.IN',
  '1960',
  '47.697888095096395'],
 ['Arab World',
  'ARB',
  'CO2 emissions (kt)',
  'EN.ATM.CO2E.KT',
  '1960',
  '59563.9892169935'],
 ['Arab World',
  'ARB',
  'CO2 emissions (metric tons per capita)',
  'EN.ATM.CO2E.PC',
  '1960',
  '0.6439635478877049'],
 ['Arab World',
  'ARB',
  'CO2 emissions from gaseous fuel consumption (% of total)',
  'EN.ATM.CO2E.GF.ZS',
  '1960',
  '5.041291753975099'],
 ['Arab World',
  'ARB',
  'CO2 emissions from liquid fuel consumption (% of total)',
  'EN.ATM.CO2E.LF.ZS',
  '1960',
  '84.8514729446567'],
 ['Arab World',
  'ARB',
  'CO2 emissions from liquid fuel consumption (kt)',
  'EN.ATM.CO2E.LF.KT',
  '1960',
  '49541.707291032304'],
 ['Arab World',
  'ARB',
  'CO2 emissions from solid fuel consumption (% of total)',
  'EN.ATM.CO2E.SF.ZS',
  '1960',
  '4.72698138789597'],
 ['Arab World',
  'ARB',
  'Death rate, crude (per 1,000 people)',
  'SP.DYN.CDRT.IN',
  '1960',
  '19.7544519237187'],
 ['Arab World',
  'ARB',
  'Fertility rate, total (births per woman)',
  'SP.DYN.TFRT.IN',
  '1960',
  '6.92402738655897'],
 ['Arab World',
  'ARB',
  'Fixed telephone subscriptions',
  'IT.MLT.MAIN',
  '1960',
  '406833.0'],
 ['Arab World',
  'ARB',
  'Fixed telephone subscriptions (per 100 people)',
  'IT.MLT.MAIN.P2',
  '1960',
  '0.6167005703199'],
 ['Arab World',
  'ARB',
  'Hospital beds (per 1,000 people)',
  'SH.MED.BEDS.ZS',
  '1960',
  '1.9296220724398703'],
 ['Arab World',
  'ARB',
  'International migrant stock (% of population)',
  'SM.POP.TOTL.ZS',
  '1960',
  '2.9906371279862403'],
 ['Arab World',
  'ARB',
  'International migrant stock, total',
  'SM.POP.TOTL',
  '1960',
  '3324685.0']]


# Print the first two lists in row_lists
print(row_lists[0])
print(row_lists[1])

# Turn list of lists into list of dicts: list_of_dicts
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]

# Print the first two dictionaries in list_of_dicts
print(list_of_dicts[0])
print(list_of_dicts[1])

# Turn list of dicts into a DataFrame: df
df = pd.DataFrame(list_of_dicts)

# Print the head of the DataFrame
df.head()

# =============================================================================
# Processing data in chunks (1)
# Sometimes, data sources can be so large in size that storing the entire 
# dataset in memory becomes too resource-intensive. 
# In this exercise, you will process the first 1000 rows of a file line by line, 
# to create a dictionary of the counts of how many times each country appears 
# in a column in the dataset.
# 
# The csv file 'world_dev_ind.csv' is in your current directory for your use. 
# To begin, you need to open a connection to this file using what is known as a 
# context manager. For example, the command with open('datacamp.csv') as datacamp 
# binds the csv file 'datacamp.csv' as datacamp in the context manager. 
# Here, the with statement is the context manager, and its purpose is to 
# ensure that resources are efficiently allocated when opening a connection to a file.
# 
# =============================================================================


# Open a connection to the file
with open('data/WDI_csv/WDIData.csv') as file:

    # Skip the column names
    file.readline()

    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Process only the first 1000 rows
    for j in range(0, 1000):

        # Split the current line into a list: line
        line = file.readline().split(',')

        # Get the value for the first column: first_col
        first_col = line[0]

        # If the column value is in the dict, increment its value
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1

        # Else, add to the dict and set value to 1
        else:
            counts_dict[first_col] = 1

# Print the resulting dictionary
print(counts_dict)

# =============================================================================
# Writing a generator to load data in chunks (2)
# In the previous exercise, you processed a file line by line for a given number 
# of lines. What if, however, you want to do this for the entire file?
# 
# In this case, it would be useful to use generators. Generators allow users to 
# lazily evaluate data. This concept of lazy evaluation is useful when you have 
# to deal with very large datasets because it lets you generate values in an 
# efficient manner by yielding only chunks of data at a time instead of the whole thing at once.
# 
# In this exercise, you will define a generator function read_large_file() that 
# produces a generator object which yields a single line from a file each time 
# next() is called on it. The csv file 'world_dev_ind.csv' is in your current 
# directory for your use.
# 
# Note that when you open a connection to a file, the resulting file object is 
# already a generator! So out in the wild, you won't have to explicitly create 
# generator objects in cases such as this
# =============================================================================

# Define read_large_file()
def read_large_file(file_object):
    """A generator function to read a large file lazily."""

    # Loop indefinitely until the end of the file
    while True:

        # Read a line from the file: data
        data = file_object.readline()

        # Break if this is the end of the file
        if not data:
            break

        # Yield the line of data
        yield data
        
# Open a connection to the file
with open('data/WDI_csv/WDIData.csv') as file:

    # Create a generator object for the file: gen_file
    gen_file = read_large_file(file)

    # Print the first three lines of the file
    print(next(gen_file))
    print(next(gen_file))
    print(next(gen_file))


# =============================================================================
# Now let's use your generator function to process the World Bank dataset like 
# you did previously. You will process the file line by line, to create a 
# dictionary of the counts of how many times each country appears in a column 
# in the dataset. For this exercise, however, you won't process just 1000 rows 
# of data, you'll process the entire dataset!
# =============================================================================

# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Open a connection to the file
with open('data/WDI_csv/WDIData.csv') as file:
    # Skip the column names
    file.readline()

    # Iterate over the generator from read_large_file()
    for line in read_large_file(file):

        row = line.split(',')
        first_col = row[0]

        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1
        else:
            counts_dict[first_col] = 1

# Print            
print(counts_dict)

# =============================================================================
# Writing an iterator to load data in chunks (1)
# Another way to read data too large to store in memory in chunks is to read the 
# file in as DataFrames of a certain length, say, 100. For example, with the 
# pandas package (imported as pd), you can do pd.read_csv(filename, chunksize=100). 
# This creates an iterable reader object, which means that you can use next() on it.
# 
# In this exercise, you will read a file in small DataFrame chunks with read_csv(). 
# You're going to use the World Bank Indicators data 'ind_pop.csv', available in 
# your current directory, to look at the urban population indicator for numerous 
# countries and years.
# =============================================================================

# Initialize reader object: df_reader
df_reader = pd.read_csv('data/WDI_csv/WDIData.csv', chunksize = 10)

# Print two chunks
print(next(df_reader))
print(next(df_reader))

# =============================================================================
# To process the data, you will create another DataFrame composed of only the 
# rows from a specific country. You will then zip together two of the 
# columns from the new DataFrame, 'Total Population' and 'Urban population (% of total)'. 
# Finally, you will create a list of tuples from the zip object, 
# where each tuple is composed of a value from each of the two columns mentioned.
# =============================================================================

# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('data/Indicators.csv', chunksize = 1000)

# Get the first DataFrame chunk: df_urb_pop
df_urb_pop = next(urb_pop_reader)

# Check out the head of the DataFrame
print(df_urb_pop.head())

# Check out specific country: df_pop_ceb
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

#df_pop_ceb = df_pop_ceb[np.logical_or(df_pop_ceb['IndicatorName'] == 'Urban population (% of total)',
#                           df_pop_ceb['IndicatorName'] == 'Population, total')]
     
# Zip DataFrame columns of interest: pops
pops = zip(df_pop_ceb['Population, total'], df_pop_ceb['Urban population (% of total)'])

# Turn zip object into list: pops_list
pops_list = list(pops)

# Print pops_list
print(pops_list)

# =============================================================================
# Writing an iterator to load data in chunks (3)
# You're getting used to reading and processing data in chunks by now. 
# Let's push your skills a little further by adding a column to a DataFrame.
# 
# Starting from the code of the previous exercise, you will be using a list 
# comprehension to create the values for a new column 'Total Urban Population' 
# from the list of tuples that you generated earlier. Recall from the previous 
# exercise that the first and second elements of each tuple consist of, respectively, 
# values from the columns 'Total Population' and 'Urban population (% of total)'. 
# The values in this new column 'Total Urban Population', therefore, are the product 
# of the first and second element in each tuple. Furthermore, because the 2nd element 
# is a percentage, you need to divide the entire result by 100, or alternatively, 
# multiply it by 0.01.
# 
# You will also plot the data from this new column to create a visualization of 
# the urban population data.
# =============================================================================

# Code from previous exercise
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)
df_urb_pop = next(urb_pop_reader)
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']
pops = zip(df_pop_ceb['Total Population'], 
           df_pop_ceb['Urban population (% of total)'])
pops_list = list(pops)

# Use list comprehension to create new DataFrame column 'Total Urban Population'
df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]

# Plot urban population data
df_pop_ceb.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()

# =============================================================================
# Writing an iterator to load data in chunks (4)
# In the previous exercises, you've only processed the data from the first DataFrame 
# chunk. This time, you will aggregate the results over all the DataFrame chunks in 
# the dataset. This basically means you will be processing the entire dataset now. 
# This is neat because you're going to be able to process the entire large dataset 
# by just working on smaller pieces of it!
# =============================================================================

# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

# Initialize empty DataFrame: data
data = pd.DataFrame()

# Iterate over each DataFrame chunk
for df_urb_pop in urb_pop_reader:

    # Check out specific country: df_pop_ceb
    df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

    # Zip DataFrame columns of interest: pops
    pops = zip(df_pop_ceb['Total Population'],
                df_pop_ceb['Urban population (% of total)'])

    # Turn zip object into list: pops_list
    pops_list = list(pops)

    # Use list comprehension to create new DataFrame column 'Total Urban Population'
    df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
    
    # Append DataFrame chunk to data: data
    data = data.append(df_pop_ceb)

# Plot urban population data
data.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()

# =============================================================================
# In this last exercise, you will put all the code for processing the data into a 
# single function so that you can reuse the code without having to rewrite the 
# same things all over again.
# 
# You're going to define the function plot_pop() which takes two arguments: 
# the filename of the file to be processed, and the country code of the rows you 
# want to process in the dataset.
# 
# Because all of the previous code you've written in the previous exercises will 
# be housed in plot_pop(), calling the function already does the following:
# 
# Loading of the file chunk by chunk,
# Creating the new column of urban population values, and
# Plotting the urban population data.
# That's a lot of work, but the function now makes it convenient to repeat the 
# same process for whatever file and country code you want to process and visualize!
# 
# =============================================================================


# Define plot_pop()
def plot_pop(filename, country_code):

    # Initialize reader object: urb_pop_reader
    urb_pop_reader = pd.read_csv(filename, chunksize=1000)

    # Initialize empty DataFrame: data
    data = pd.DataFrame()
    
    # Iterate over each DataFrame chunk
    for df_urb_pop in urb_pop_reader:
        # Check out specific country: df_pop_ceb
        df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == country_code]

        # Zip DataFrame columns of interest: pops
        pops = zip(df_pop_ceb['Total Population'],
                    df_pop_ceb['Urban population (% of total)'])

        # Turn zip object into list: pops_list
        pops_list = list(pops)

        # Use list comprehension to create new DataFrame column 'Total Urban Population'
        df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
    
        # Append DataFrame chunk to data: data
        data = data.append(df_pop_ceb)

    # Plot urban population data
    data.plot(kind='scatter', x='Year', y='Total Urban Population')
    plt.show()

# Set the filename: fn
fn = 'ind_pop_data.csv'

# Call plot_pop for country code 'CEB'
plot_pop('ind_pop_data.csv', 'CEB')

# Call plot_pop for country code 'ARB'
plot_pop('ind_pop_data.csv', 'ARB')



















