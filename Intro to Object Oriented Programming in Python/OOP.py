#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:32:19 2019

@author: amin
"""

# =============================================================================
# Intro to Object Oriented Programming in Python
# =============================================================================
#A way to build flexible, reproducible code.
#Developing building blocks to developing more advanced modules and libraries

#Imperative Style and OOP Style

#IMPERATIVE

our_list = [1,2,3]
for item in our_list:
    print(f"Item {item}")
   
#OOP

class PrintList:
    def __init__(self,numberlist):
        self.numberlist = numberlist
        
    def print_list(self):
        for item in self.numberlist:
            print(f"Item {item}")

A = PrintList([1,2,3])
A.print_list()

# =============================================================================
# Creating functions
# 
# In this exercise, we will review functions, as they are key building blocks of 
# object-oriented programs.
# 
# For this, we will create a simple function average_numbers() which averages a 
# list of numbers. Remember that lists are a basic data type in Python that we 
# can build using the [] bracket notation
# =============================================================================


# Create function that returns the average of an integer list
def average_numbers(num_list): 
    avg = sum(num_list)/float(len(num_list)) # divide by length of list
    return avg

# Take the average of a list: my_avg
my_avg = average_numbers([1, 2, 3, 4, 5, 6])

# Print out my_avg
print(my_avg)

# =============================================================================
# Introduction to NumPy Internals
# =============================================================================

#Create a function that returns a NumPy array

# Import numpy as np
import numpy as np

# List input: my_matrix
my_matrix = [[1,2,3,4], [5,6,7,8]] 

# Function that converts lists to arrays: return_array
def return_array(matrix):
    array = np.array(matrix, dtype = float)
    return array
    
# Call return_array on my_matrix, and print the output
print(return_array(my_matrix))

# =============================================================================
# Introduction to Objects and Classes
# =============================================================================

#What is a class?
#
#A reuseable chunk of code that has methods and variables. A Class is a template for an object
#
#OOP vocabulary
#variable = attribute/field/class variable
#function = method

#Creating a class
#
#We're going to be working on building a class, which is a way to organize 
#functions and variables in Python. To start with, let's look at the simplest 
#possible way to create a class

# Create a class: DataShell
class DataShell: 
    pass

# Objects are instances of classes and can have both variables and functions

#The init function is known as the constructor for the class. Special underscores 
#initialize the class. Within the constructor, there's the filename variable 
#being passed as a parameter. This is a class variable, or its attribute, and it's
#initialized when we create the class. 
#
#There are two other methods. One that creates the datashell by taking a csv and
#returning arrays, and the second method which renames columns in the datashell. 
#
#Special keywords and features to take note of include self, return and the class 
#attribute called filename. 

class DataShell:
    #constructor
    def __init__(self, filename):
        self.filename = filename
        
    def create_datashell(self):
        self.array = np.genfromtxt(self.filename, delimiter=',', dtype=None)
        return self.array
    
    def rename_column(self, old_colname, new_colname):
        for index, value in enumerate(self.array[0]):
            if value == old_colname.encode('UTF-8'):
                self.array[0][index] = new_colname
        return self.array
            
    def show_shell(self):
        print(self.array)
        
    def five_figure_summary(self, col_pos):
        statistics = stats.describe(self.array[1:,col_pos].astype(np.float))
        return f"Five-figure stats of column {col_position}: {statistics}"

#Object: Instance of a Class

# Create empty class: DataShell
class DataShell():
  
    # Pass statement
    pass

# Instantiate DataShell: my_data_shell
my_data_shell = DataShell()

# Print my_data_shell
print(my_data_shell)

# =============================================================================
# Initializing a Class and Self
# =============================================================================

#An important special method is the __init__ method.THis is also known as a 
#constructor and it's job is to set up the object the way you want it from the
#very beginning, before anything is passed in. It's called automatically
#when an object is created and therefore takes in values outside of the object, 
#or a set of values within the object. 
#
#Init can be empty and it can also take values. These values are created and 
#passed into the class, and can be assessed from the class later on. 

#Empty constructor 

class Dinasaur:
    
    def __init__(self):
        pass
    
#Empty constructor with attributes 
        
class Dinasaur:
    
    def __init__(self):
        self.tail = 'Yes' 

#We set the tail attribute of the class to yes, and from now on every object 
#that is a dinasaur class will have a tail. 

# Modeled on Pandas read_csv. 
import pandas as pd
x = pd.read_csv('data/auto-mpg.csv', sep = ',')

#read_csv can take in a number of objects. If you can read_csv, you need to pass
#in a filename

#Below, we are creating a class DataShell, and we are initializing it with a 
#filename. In this case, the filename is passed in from the outside of the class.
#We need to pass in a filename to create that object; otherwise, we can't do
#anything with our datashell. 

class DataShell:
    
    def __init__(self, filename):
        self.filename = filename
        
#The filename variable is not available for all members(methods) of the class
#to use. 

# =============================================================================
# # Understanding self
# =============================================================================

#Within the constructor, self represents the instance of the class or the 
#specific object. Remember an object is an instance of a class. That object needs 
#a way to reference that instance. THe first variable is always a reference to the
#current instance of the class. In this case, the instance of the class is the 
#class itself, so we put self. 


# =============================================================================
# The Init Method
# 
# Now it's time to explore the special __init__ method!
# 
# __init__ is an initialization method used to construct class instances in custom ways
# =============================================================================

# Create class: DataShell
class DataShell:
  
	# Initialize class with self argument
    def __init__(self):
      
        # Pass statement
        pass

# Instantiate DataShell: my_data_shell
my_data_shell = DataShell()

# Print my_data_shell
print(my_data_shell)

#Instance Variables
#
#Class instances are useful in that we can store values in them at the time of instantiation. 
#We store these values in instance variables. This means that we can have many 
#instances of the same class whose instance variables hold different values!

# Create class: DataShell
class DataShell:
  
	# Initialize class with self and integerInput arguments
    def __init__(self, integerInput):
      
		# Set data as instance variable, and assign the value of integerInput
        self.integerInput = integerInput

# Declare variable x with value of 10
x = 10    

# Instantiate DataShell passing x as argument: my_data_shell
my_data_shell = DataShell(10)

# Print my_data_shell
print(my_data_shell.integerInput)

# *****************************************************************************

# Create class: DataShell
class DataShell:
  
	# Initialize class with self and integerInput arguments
    def __init__(self, integerInput):
      
		# Set data as instance variable, and assign the value of integerInput
        self.data = integerInput

# Declare variable x with value of 10
x = 10    

# Instantiate DataShell passing x as argument: my_data_shell
my_data_shell = DataShell(10)

# Print my_data_shell
print(my_data_shell.data)

# =============================================================================
# Multiple Instance Variables
# 
# We are not limited to declaring only one instance variable; in fact, we can declare many!
# 
# In this exercise we will declare two instance variables: identifier and data. 
# Their values will be specified by the values passed to the initialization method, as before.
# =============================================================================

# Create class: DataShell
class DataShell:
  
	# Initialize class with self, identifier and data arguments
    def __init__(self, identifier, data):
      
		# Set identifier and data as instance variables, assigning value of input arguments
        self.identifier = identifier
        self.data = data

# Declare variable x with value of 100, and y with list of integers from 1 to 5
x = 100
y = [1, 2, 3, 4, 5]

# Instantiate DataShell passing x and y as arguments: my_data_shell
my_data_shell = DataShell(x, y)

# Print my_data_shell.identifier
print(my_data_shell.identifier)

# Print my_data_shell.data
print(my_data_shell.data)


# =============================================================================
# More on Self and Passing in Variables
# =============================================================================

#when we create a class, we can choose for it to have static variables. These
#variables don't change no matter what we do to the members of the class. 

class Dinosaur():
    eyes = 2
    
    def __init__(self, teeth):
        self.teeth = teeth

# eyes variable is static. Teeth is passed in and we can set when we construct
# the class. This is known as an instance variable. 

stegosaurus = Dinosaur(40)
stegosaurus.eyes
stegosaurus.teeth


# =============================================================================
# Class Variables
# 
# We saw that we can specify different instance variables.
# 
# But, what if we want any instance of a class to hold the same value for a 
# specific variable? Enter class variables.
# 
# Class variables must not be specified at the time of instantiation and instead, 
# are declared/specified at the class definition phase
# 
# =============================================================================

# Create class: DataShell
class DataShell:
  
    # Declare a class variable family, and assign value of "DataShell"
    family = 'DataShell'
    
    # Initialize class with self, identifier arguments
    def __init__(self, identifier):
      
        # Set identifier as instance variable of input argument
        self.identifier = identifier

# Declare variable x with value of 100
x = 100

# Instantiate DataShell passing x as argument: my_data_shell
my_data_shell = DataShell(x)

# Print my_data_shell class variable family
print(my_data_shell.family)

# =============================================================================
# Overriding Class Variables
# =============================================================================

# Create class: DataShell
class DataShell:
  
    # Declare a class variable family, and assign value of "DataShell"
    family = 'DataShell'
    
    # Initialize class with self, identifier arguments
    def __init__(self, identifier):
      
        # Set identifier as instance variables, assigning value of input arguments
        self.identifier = identifier

# Declare variable x with value of 100
x = 100

# Instantiate DataShell passing x and y as arguments: my_data_shell
my_data_shell = DataShell(x)

# Print my_data_shell class variable family
print(my_data_shell.family)

# Override the my_data_shell.family value with "NotDataShell"
my_data_shell.family = 'NotDataShell'

# Print my_data_shell class variable family once again
print(my_data_shell.family)

# =============================================================================
# Methods in Classes
# =============================================================================

#A class can have multiple attributes as shown below

class DataShell:
    def __init__(self, filename):
        self.filename = filename
    
    def create_datashell(self):
        self.array = np.genfromtxt(self.filename, delimiter=',', dtype=None)
        return self.array
    
    def rename_column(self, old_colname, new_colname):
        for index, value in enumerate(self.array[0]):
            if value == old_colname.encode('UTF-8'):
                self.array[0][index] = new_colname
        return self.array

#Each of the methods above returns variables which get passed back into the class
#as class attributes. 
#
#create_datashell method takes self as an argument. We pass in self to ensure
#the method is part of the class. Notice that when we pass in filename,
#it's called self.filename cos it's a class attribute. We don't have to pass it 
#in explicitly as a parameter when we define the method. 
#
#in rename_column, we still have self to reference to the instance object. 

# =============================================================================
# Methods I
# =============================================================================

# Create class: DataShell
class DataShell:
  
	# Initialize class with self argument
    def __init__(self):
        pass
      
	# Define class method which takes self argument: print_static
    def print_static(self):
        # Print string
        print("You just executed a class method!")
        
# Instantiate DataShell taking no arguments: my_data_shell
my_data_shell = DataShell()

# Call the print_static method of your newly created object
my_data_shell.print_static()

# =============================================================================
# Methods II
# =============================================================================

# Create class: DataShell
class DataShell:
  
	# Initialize class with self and dataList as arguments
    def __init__(self, dataList):
      	# Set data as instance variable, and assign it the value of dataList
        self.data= dataList
        
	# Define class method which takes self argument: show
    def show(self):
        # Print the instance variable data
        print(self.data)

# Declare variable with list of integers from 1 to 10: integer_list   
integer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
# Instantiate DataShell taking integer_list as argument: my_data_shell
my_data_shell = DataShell(integer_list)

# Call the show method of your newly created object
my_data_shell.show()

# =============================================================================
# Methods III
# =============================================================================

# Create class: DataShell
class DataShell:
  
	# Initialize class with self and dataList as arguments
    def __init__(self, dataList):
      	# Set data as instance variable, and assign it the value of dataList
        self.data = dataList
        
	# Define method that prints data: show
    def show(self):
        print(self.data)
        
    # Define method that prints average of data: avg 
    def avg(self):
        # Declare avg and assign it the average of data
        avg = sum(self.data)/float(len(self.data))
        # Print avg
        print(avg)
        
# Instantiate DataShell taking integer_list as argument: my_data_shell
my_data_shell = DataShell(integer_list)

# Call the show and avg methods of your newly created object
my_data_shell.show()
my_data_shell.avg()



































