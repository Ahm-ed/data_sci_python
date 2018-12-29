#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: amin
"""

import pandas as pd
import numpy as np

x = 1
y1 = str(x)
y2 = print(x)

type(x)
type(y1)
type(y2)

# =============================================================================
# It is important to remember that assigning a variable y2 to a function that 
# prints a value but does not return a value will result in 
# that variable y2 being of type NoneType
# =============================================================================

x = str(5)

print(type(x))

def square(): # This is the function header
    new_value = 4 ** 2
    print(new_value) ## Function body
    
def square(value): # We write parameters in the function header
    new_value = value ** 2
    print(new_value) ## Function body
    
## When we call a function, we pass arguments into the function
    
## If we don't want to print it but rather be able to return the square value
 #   and assign it to a new variable
 
def square(value): # We write parameters in the function header
    new_value = value ** 2
    return new_value ## Function body


# =============================================================================
# Docstrings --- they are used to describe what your function does. 
# They are placed in the immediate line after the function header 
# and are placed between triple quotation marks. """ --- """
# =============================================================================
    
def square(value): # We write parameters in the function header
    """ Returns the square of a value"""
    new_value = value ** 2
    return new_value ## Function body

# Define the function shout
def shout():
    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = 'congratulations' + '!!!'

    # Print shout_word
    print(shout_word)

# Call shout
shout()


# Define shout with the parameter, word
def shout(word):
    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = word + '!!!'

    # Print shout_word
    print(shout_word)

# Call shout with the string 'congratulations'
shout("congratulations")


# =============================================================================
# Functions that return single values
# You're getting very good at this! Try your hand at another modification to 
# the shout() function so that it now returns a single value instead of printing
#  within the function. Recall that the return keyword lets you return values 
#  from functions. Parts of the function shout(), which you wrote earlier,
#  are shown. Returning values is generally more desirable than printing them out 
#  'because, as you saw earlier, a print() call assigned to a variable has type NoneType
# =============================================================================

# Define shout with the parameter, word
def shout(word):
    """Return a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = word + "!!!"

    # Replace print with return
    return shout_word

# Pass 'congratulations' to shout: yell
yell = shout("congratulations")

print(yell)

def raise_to_power(value1, value2):
    """Raise value1 to the power of value2"""
    new_value = value1 ** value2
    return new_value

# Define shout with parameters word1 and word2
def shout(word1, word2):
    """Concatenate strings with three exclamation marks"""
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + "!!!"
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + "!!!"
    # Concatenate shout1 with shout2: new_shout
    new_shout = shout1 + shout2
    # Return new_shout
    return new_shout

# Pass 'congratulations' and 'you' to shout(): yell
yell = shout("congratulations", "you")

# Print yell
print(yell)

# =============================================================================
# A brief introduction to tuples
# Alongside learning about functions, you've also learned about tuples! 
# Here, you will practice what you've learned about tuples: 
# how to construct, unpack, and access tuple elements. 
# Recall how Hugo unpacked the tuple even_nums in the video:
# 
# a, b, c = even_nums
# 
# Tuples are constructed using parenthesis ()
# =============================================================================

nums = (3,4,6)

# Unpack nums into num1, num2, and num3
a, b, c = nums

num1 = a
num2 = b
num3 = c

print(num1)

# Construct even_nums
even_nums = (2, 4, 6)


# =============================================================================
# Functions that return multiple values
# In the previous exercise, you constructed tuples, assigned tuples to variables, 
# and unpacked tuples. Here you will return multiple values from a function using tuples. 
# Let's now update our shout() function to return multiple values. 
# Instead of returning just one string, we will return two '
# strings with the string !!! concatenated to each.
# 
# Note that the return statement return x, y has the same 
# result as return (x, y): the former actually packs x and y into a tuple under the hood!
# =============================================================================

# Define shout_all with parameters word1 and word2
def shout_all(word1, word2):
    """Return a tuple of strings"""
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'
    
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'
    
    # Construct a tuple with shout1 and shout2: shout_words
    shout_words = (shout1, shout2)

    # Return shout_words
    return shout_words

# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
yell1, yell2 = shout_all('congratulations', 'you')

# Print yell1 and yell2
print(yell1)
print(yell2)

# =============================================================================
# Bringing it all together (1)
# You've got your first taste of writing your own functions in the previous exercises. 
# You've learned how to add parameters to your own function definitions, 
# return a value or multiple values with tuples, and how to call the functions you've defined.
# 
# In this and the following exercise, you will bring together all these concepts
#  and apply them to a simple data science problem. You will load a dataset and 
#  develop functionalities to extract simple insights from the data.
# 
# For this exercise, your goal is to recall how to load a dataset into a DataFrame. 
# The dataset contains Twitter data and you will iterate over entries in a column 
# to build a dictionary in which the keys are the names of languages and the values 
# are the number of tweets in the given language. 
# The file tweets.csv is available in your current directory.
# =============================================================================

# Import Twitter data as DataFrame: df
df = pd.read_csv('tweets.csv')

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:

    # If the language is in langs_count, add 1
    if entry in langs_count.keys():
        langs_count[entry] += 1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry] = 1

# Print the populated dictionary
print(langs_count)

# =============================================================================
# Bringing it all together (2)
# Great job! You've now defined the functionality for iterating over entries 
# in a column and building a dictionary with keys the names of languages and 
# values the number of tweets in the given language.
# 
# In this exercise, you will define a function with the functionality you 
# developed in the previous exercise, return the resulting dictionary from 
# within the function, and call the function with the appropriate arguments.
# 
# For your convenience, the pandas package has been imported as pd and the 
# 'tweets.csv' file has been imported into the tweets_df variable.
# 
# =============================================================================


# Define count_entries()
def count_entries(df, col_name):
    """Return a dictionary with counts of 
    occurrences as value for each key."""

    # Initialize an empty dictionary: langs_count
    langs_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over lang column in DataFrame
    for entry in col:

        # If the language is in langs_count, add 1
        if entry in langs_count.keys():
            langs_count[entry] += 1
        # Else add the language to langs_count, set the value to 1
        else:
            langs_count[entry] = 1

    # Return the langs_count dictionary
    return langs_count

# Call count_entries(): result
result = count_entries(tweets_df, "lang")

# Print the result
print(result)


# =============================================================================
# SCOPING
# 1. Global
# 2. Local
# 3. Built in
# =============================================================================

def func1():
    num = 3
    print(num)

def func2():
    global num
    double_num = num * 2
    num = 6
    print(double_num)
    
#func1() prints out 3, func2() prints out 10, and the value of num in the global scope is 6.

# Create a string: team
team = "teen titans"

# Define change_team()
def change_team():
    """Change the value of the global variable team."""
    # Use team in global scope
    global team
    # Change the value of team in global: team
    team = "justice league"
    
# Print team
print(team)

# Call change_team()
change_team()

# Print team
print(team)

# =============================================================================
# Python's built-in scope
# Here you're going to check out Python's built-in scope, 
# which is really just a built-in module called builtins. 
# However, to query builtins, you'll need to import builtins 'because the name 
# builtins is not itself built in...No, I’m serious!' 
# (Learning Python, 5th edition, Mark Lutz). 
# After executing import builtins in the IPython Shell, 
# execute dir(builtins) to print a list of all the names in the module builtins. 
# Have a look and you'll see a bunch of names that you'll recognize!
#  Which of the following names is NOT in the module builtins?
# =============================================================================


import builtins

dir(builtins)


### NESTED FUNCTIONS

# =============================================================================
# Returning functions!
# =============================================================================

def raise_val(n):
    """Return the inner function"""
    def inner(x):
        """Raise x to the power of n"""
        raised = x ** n
        return raised
    
    return inner ## REturns a function!!!!!! 

square = raise_val(2)
cube = raise_val(3)

print(square(2), cube(4))


#You can use the keyword global in function definition to create and change global
#names. Similarly, in nested functions, you can use the keyword nonlocal to create
#and change names in an eclosing scope. 

def outer():
    """prints the value of n"""
    n = 1
    def inner():
        nonlocal n
        n = 2
        print(n)
    inner()
    print(n)

# =============================================================================
# #LEGB rule - the order of scope searching. Local scope, Enclosing functions, Global and Built-in
# 
# Assigning names will only create or change local names, unless they declared in global
# or nonlocal statements using the keyword global or the keyword nonlocal respectively
# =============================================================================

# =============================================================================
# Nested Functions I
# You've learned in the last video about nesting functions within functions. 
# One reason why you'd like to do this is to avoid writing out the same 
# computations within functions repeatedly. 
# There's nothing new about defining nested functions: 
# you simply define it as you would a regular function with def and embed it inside another function!
# 
# In this exercise, inside a function three_shouts(), 
# you will define a nested function inner() that concatenates a string object with !!!. 
# three_shouts() then returns a tuple of three elements, each a string concatenated with !!! 
# using inner(). Go for it!
# 
# =============================================================================
    
# Define three_shouts
def three_shouts(word1, word2, word3):
    """Returns a tuple of strings
    concatenated with '!!!'."""

    # Define inner
    def inner(word):
        """Returns a string concatenated with '!!!'."""
        return word + '!!!'

    # Return a tuple of strings
    return (inner(word1), inner(word2), inner(word3))

# Call three_shouts() and print
print(three_shouts('a', 'b', 'c'))


# =============================================================================
# Nested Functions II
# Great job, you've just nested a function within another function.
#  One other pretty cool reason for nesting functions is the idea of a closure. 
#  This means that the nested or inner function remembers the state of its 
#  enclosing scope when called. Thus, anything defined locally in the enclosing
#  scope is available to the inner function even when the outer function has finished execution.
# 
# =============================================================================

# Define echo
def echo(n):
    """Return the inner_echo function."""

    # Define inner_echo
    def inner_echo(word1):
        """Concatenate n copies of word1."""
        echo_word = word1 * n
        return echo_word

    # Return inner_echo
    return inner_echo

# Call echo: twice
twice = echo(2)

# Call echo: thrice
thrice = echo(3)

# Call twice() and thrice() then print
print(twice('hello'), thrice('hello'))

# =============================================================================
# In this exercise, you will use the keyword nonlocal within a nested function 
# to alter the value of a variable defined in the enclosing scope.
# 
# =============================================================================


# Define echo_shout()
def echo_shout(word):
    """Change the value of a nonlocal variable"""
    
    # Concatenate word with itself: echo_word
    echo_word = word + word
    
    # Print echo_word
    print(echo_word)
    
    # Define inner function shout()
    def shout():
        """Alter a variable in the enclosing scope"""    
        # Use echo_word in nonlocal scope
        nonlocal echo_word
        
        # Change echo_word to echo_word concatenated with '!!!'
        echo_word = echo_word + "!!!"
    
    # Call function shout()
    shout()
    
    # Print echo_word
    print(echo_word)

# Call function echo_shout() with argument 'hello'
echo_shout("hello")

### DEFAULT ARGUMENTS 

# Define shout_echo
def shout_echo(word1, echo  = 1):
    """Concatenate echo copies of word1 and three
     exclamation marks at the end of the string."""

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Concatenate '!!!' to echo_word: shout_word
    shout_word = echo_word + '!!!'

    # Return shout_word
    return shout_word

# Call shout_echo() with "Hey": no_echo
no_echo = shout_echo("Hey")

# Call shout_echo() with "Hey" and echo=5: with_echo
with_echo = shout_echo("Hey", 5)

# Print no_echo and with_echo
print(no_echo)
print(with_echo)

# Define shout_echo
def shout_echo(word1, echo = 1, intense = False):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Capitalize echo_word if intense is True
    if intense is True:
        # Capitalize and concatenate '!!!': echo_word_new
        echo_word_new = echo_word.upper() + '!!!'
    else:
        # Concatenate '!!!' to echo_word: echo_word_new
        echo_word_new = echo_word + '!!!'

    # Return echo_word_new
    return echo_word_new

# Call shout_echo() with "Hey", echo=5 and intense=True: with_big_echo
with_big_echo = shout_echo("Hey", 5, True)

# Call shout_echo() with "Hey" and intense=True: big_no_echo
big_no_echo = shout_echo("Hey", intense = True)

# Print values
print(with_big_echo)
print(big_no_echo)

# =============================================================================
# Functions with variable-length arguments (*args)
# Flexible arguments enable you to pass a variable number of arguments to a function. 
# In this exercise, you will practice defining a function that accepts a 
# variable number of string arguments.
# 
# The function you will define is gibberish() which can accept a variable number
#  of string values. Its return value is a single string composed of all the string 
#  arguments concatenated together in the order they were passed to the function call.
#  You will call the function with a single string argument and see how the 
#  output changes with another call using more than one string argument. 
#  Recall from the previous video that, within the function definition, args is a tuple
# =============================================================================

# Define gibberish
def gibberish(*args):
    """Concatenate strings in *args together."""

    # Initialize an empty string: hodgepodge
    hodgepodge = ''

    # Concatenate the strings in args
    for word in args:
        hodgepodge += word

    # Return hodgepodge
    return hodgepodge

# Call gibberish() with one string: one_word
one_word = gibberish("luke")

# Call gibberish() with five strings: many_words
many_words = gibberish("luke", "leia", "han", "obi", "darth")

# Print one_word and many_words
print(one_word)
print(many_words)

def add_all(*args):
    """Sum all values in *args together."""
    ## Initialize sum
    sum_all = 0
    ## Accumulate the sum
    for num in args:
        sum_all += num
    return sum_all

add_all(1)
add_all(4, 5, 6)

# =============================================================================
# Functions with variable-length keyword arguments (**kwargs)
# Let's push further on what you've learned about flexible arguments - 
# you've used *args, you're now going to use **kwargs! What makes **kwargs 
# different is that it allows you to pass a variable number of keyword 
# arguments to functions. Recall from the previous video that, within the 
# function definition, kwargs is a dictionary.
# 
# To understand this idea better, you're going to use **kwargs in this exercise 
# to define a function that accepts a variable number of keyword arguments. 
# The function simulates a simple status report system that prints out the 
# status of a character in a movie.
# =============================================================================

# Define report_status
def report_status(**kwargs):
    """Print out the status of a movie character."""

    print("\nBEGIN: REPORT\n")

    # Iterate over the key-value pairs of kwargs
    for key, value in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(key + ": " + value)

    print("\nEND REPORT")

# First call to report_status()
report_status(name = "luke", affiliation = "jedi", status = "missing")

# Second call to report_status()
report_status(name="anakin", affiliation="sith lord", status="deceased")

def print_all(**kwargs):
    """Print out key-value pairs in **kwargs."""
    ## Print out key-value pairs 
    for key, value in kwargs.items():
        print(key + ": " + value)
        
print_all(name = "Amin Yakubu", job = "Data Scientist", education = "MD/MPH")

# =============================================================================
# #### BRINGING IT ALL TOGETHER 
# =============================================================================

# Define count_entries()
def count_entries(df, col_name = 'lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over the column in DataFrame
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1

        # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df)

# Call count_entries(): result2
result2 = count_entries(tweets_df, col_name = 'source')

# Print result1 and result2
print(result1)
print(result2)

# =============================================================================
# =============================================================================

# Define count_entries()
def count_entries(df, *args):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    #Initialize an empty dictionary: cols_count
    cols_count = {}
    
    # Iterate over column names in args
    for col_name in args:
    
        # Extract column from DataFrame: col
        col = df[col_name]
    
        # Iterate over the column in DataFrame
        for entry in col:
    
            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
    
            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'lang', 'source')

# Print result1 and result2
print(result1)
print(result2)

# =============================================================================
# Writing a lambda function you already know
# Some function definitions are simple enough that they can be converted to a 
# lambda function. By doing this, you write less lines of code, which is pretty 
# awesome and will come in handy, especially when you're writing and maintaining 
# big programs. In this exercise, you will use what you know about lambda functions 
# to convert a function that does a simple task into a lambda function. 
# 
# =============================================================================
# Define echo_word as a lambda function: echo_word
echo_word = (lambda word1, echo: word1 * echo)

# Call echo_word: result
result = echo_word('hey', 5)

# Print result
print(result)

raise_to_power = lambda x, y: x ** y 

raise_to_power(2, 3) 

# =============================================================================
# The best use case for lambda functions, however, are for when you want these 
# simple functionalities to be anonymously embedded within larger expressions. 
# What that means is that the functionality is not stored in the environment, 
# unlike a function defined with def. To understand this idea better, 
# you will use a lambda function in the context of the map() function.
# =============================================================================

# =============================================================================
# Anonymous function
# Function map takes two arguments: map(func, seq)
# ● map() applies the function to ALL elements in the
# sequence
# =============================================================================

nums = [48, 6, 9, 21, 1] 

square_all = map(lambda num: num ** 2, nums) 

print(square_all)

print(list(square_all))

# Create a list of strings: spells
spells = ["protego", "accio", "expecto patronum", "legilimens"]

# Use map() to apply a lambda function over spells: shout_spells
shout_spells = map((lambda item: item + '!!!'), spells)

# Convert shout_spells to a list: shout_spells_list
shout_spells_list = list(shout_spells)

# Convert shout_spells into a list and print it
print(shout_spells_list)


# =============================================================================
# Filter() and lambda functions
# In the previous exercise, you used lambda functions to anonymously embed 
# an operation within map(). You will practice this again in this exercise by 
# using a lambda function with filter(), which may be new to you! 
# The function filter() offers a way to filter out elements from a list that 
# don't satisfy certain criteria.
# 
# Your goal in this exercise is to use filter() to create, from an input list of 
# strings, a new list that contains only strings that have more than 6 characters.
# =============================================================================

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']

# Use filter() to apply a lambda function over fellowship: result
result = filter(lambda member: len(member) > 6, fellowship)

# Convert result to a list: result_list
result_list = list(result)

# Convert result into a list and print it
print(result_list)


# =============================================================================
# Reduce() and lambda functions
# You're getting very good at using lambda functions! 
# Here's one more function to add to your repertoire of skills. 
# The reduce() function is useful for performing some computation on a list and, 
# unlike map() and filter(), returns a single value as a result. To use reduce(), 
# you must import it from the functools module
# =============================================================================

# Import reduce from functools
from functools import reduce

# Create a list of strings: stark
stark = ['robb', 'sansa', 'arya', 'brandon', 'rickon']

# Use reduce() to apply a lambda function over stark: result
result = reduce(lambda item1, item2: item1 + item2, stark)

# Print the result
print(result)

# =============================================================================
# Error handling with try-except
# A good practice in writing your own functions is also anticipating the ways in 
# which other people (or yourself, if you accidentally misuse your own function) 
# might use the function you defined.
# 
# As in the previous exercise, you saw that the len() function is able to handle 
# input arguments such as strings, lists, and tuples, but not int type ones and 
# raises an appropriate error and error message when it encounters invalid input arguments. 
# One way of doing this is through exception handling with the try-except block
# =============================================================================

# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Initialize empty strings: echo_word, shout_words
    echo_word = ''
    shout_words = ''
    

    # Add exception handling with try-except
    try:
        # Concatenate echo copies of word1 using *: echo_word
        echo_word = word1 * echo

        # Concatenate '!!!' to echo_word: shout_words
        shout_words = echo_word + "!!!"
    except:
        # Print error message
        print("word1 must be a string and echo must be an integer.")

    # Return shout_words
    return shout_words

# Call shout_echo
shout_echo("particle", echo="accelerator")

# =============================================================================
# Error handling by raising an error
# Another way to raise an error is by using raise. 
# In this exercise, you will add a raise statement to the shout_echo() 
# function you defined before to raise an error message when the value 
# supplied by the user to the echo argument is less than 0.
# 
# The call to shout_echo() uses valid argument values. To test and see how the 
# raise statement works, simply change the value for the echo argument to a 
# negative value.
# =============================================================================

# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Raise an error with raise
    if echo < 0:
        raise ValueError('echo must be greater than 0')

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Concatenate '!!!' to echo_word: shout_word
    shout_word = echo_word + '!!!'

    # Return shout_word
    return shout_word

# Call shout_echo
shout_echo("particle", echo=5)

#write a lambda function and use filter() to select retweets, that is, 
#tweets that begin with the string 'RT'

#To get the first 2 characters in a tweet x, use x[0:2]

# Select retweets from the Twitter DataFrame: result
result = filter(lambda x:x[0:2]== 'RT', tweets_df['text'])

# Create list from filter object result: res_list
res_list = list(result)

# Print all retweets in res_list
for tweet in res_list:
    print(tweet)
# =============================================================================
# 
# This will allow your function to provide a helpful message when the user calls
#  your count_entries() function but provides a column name that isn't in the DataFrame.
# =============================================================================
    
# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Add try block
    try:
        # Extract column from DataFrame: col
        col = df[col_name]
        
        # Iterate over the column in dataframe
        for entry in col:
    
            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1
    
        # Return the cols_count dictionary
        return cols_count

    # Add except block
    except:
        print('The DataFrame does not have a ' + col_name + ' column.')

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Print result1
print(result1)

# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Raise a ValueError if col_name is NOT in DataFrame
    if col_name not in df.columns:
        raise ValueError('The DataFrame does not have a ' + col_name + ' column.')

    # Initialize an empty dictionary: cols_count
    cols_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over the column in DataFrame
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1
            # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1
        
        # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Print result1
print(result1)


