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

# =============================================================================
# Working with a DataSet to
# Create DataFrames
# =============================================================================

class DataShell:
    def __init__(self, filename):
        self.filename = filename
        
    def create_datashell(self):
        self.array = np.genfromtxt(self.filename, delimiter=',', dtype=None)
        return self.array

    def show_shell(self):
        print(self.array)

# creating an instance of a DataShell
car_data = DataShell('data/auto-mpg.csv')

# print the object
print(car_data)

x = car_data.create_datashell()

car_data.show_shell()

# =============================================================================
# Return Statement 
# =============================================================================

# Create class: DataShell
class DataShell:
  
	# Initialize class with self and dataList as arguments
    def __init__(self, dataList):
      	# Set data as instance variable, and assign it the value of dataList
        self.data = dataList
        
	# Define method that returns data: show
    def show(self):
        return self.data
        
    # Define method that prints average of data: avg 
    def avg(self):
        # Declare avg and assign it the average of data
        avg = sum(self.data)/float(len(self.data))
        # Return avg
        return avg
        
# Instantiate DataShell taking integer_list as argument: my_data_shell
my_data_shell = DataShell(integer_list)

# Print output of your object's show method
print(my_data_shell.show())

# Print output of your object's avg method
print(my_data_shell.avg())

# =============================================================================
# A More Powerful DataShell
# =============================================================================

# Load numpy as np and pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:
  
    # Initialize class with self and inputFile
    def __init__(self, inputFile):
        self.file = inputFile
        
    # Define generate_csv method, with self argument
    def generate_csv(self):
        self.data_as_csv = pd.read_csv(self.file)
        return self.data_as_csv

# Instantiate DataShell with us_life_expectancy as input argument
data_shell = DataShell('data/auto-mpg.csv')

# Call data_shell's generate_csv method, assign it to df
df = data_shell.generate_csv()

# Print df
print(df)

# =============================================================================
# Renaming Columns and the Five-Figure Summary
# =============================================================================

from scipy import stats

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
    
    def five_figure_summary(self):
        statistics = stats.describe(self.array[1:,col_pos].astype(np.float))
        return f"Five-figure stats of column {col_position}: {statistics}"

myDatashell = DataShell('data/auto-mpg.csv')

myDatashell.create_datashell()

myDatashell.rename_column('cyl','cylinders')

print(myDatashell.array)

myDatashell.five_figure_summary()

# =============================================================================
# Data as Attributes
# =============================================================================

us_life_expectancy = 'https://assets.datacamp.com/production/repositories/2097/datasets/5dd3a8250688a4f08306206fa1d40f63b66bc8a9/us_life_expectancy.csv'

# Import numpy as np, pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:
  
    # Define initialization method
    def __init__(self, filepath):
        # Set filepath as instance variable  
        self.filepath = filepath
        # Set data_as_csv as instance variable
        self.data_as_csv = pd.read_csv(filepath)

# Instantiate DataShell as us_data_shell
us_data_shell = DataShell(us_life_expectancy)

# Print your object's data_as_csv attribute
print(us_data_shell.data_as_csv)

#Now your classes have the ability of storing data as instance variables, which means you can execute methods on them!

# Renaming Columns

# Create class DataShell
class DataShell:
  
    # Define initialization method
    def __init__(self, filepath):
        self.filepath = filepath
        self.data_as_csv = pd.read_csv(filepath)
    
    # Define method rename_column, with arguments self, column_name, and new_column_name
    def rename_column(self, column_name, new_column_name):
        self.data_as_csv.columns = self.data_as_csv.columns.str.replace(column_name, new_column_name)

# Instantiate DataShell as us_data_shell with argument us_life_expectancy
us_data_shell = DataShell(us_life_expectancy)

# Print the datatype of your object's data_as_csv attribute
print(us_data_shell.data_as_csv.dtypes)

# Rename your objects column 'code' to 'country_code'
us_data_shell.rename_column('code', 'country_code')

# Again, print the datatype of your object's data_as_csv attribute
print(us_data_shell.data_as_csv.dtypes)

#Self-Describing DataShells

# Create class DataShell
class DataShell:

    # Define initialization method
    def __init__(self, filepath):
        self.filepath = filepath
        self.data_as_csv = pd.read_csv(filepath)

    # Define method rename_column, with arguments self, column_name, and new_column_name
    def rename_column(self, column_name, new_column_name):
        self.data_as_csv.columns = self.data_as_csv.columns.str.replace(column_name, new_column_name)
        
    # Define get_stats method, with argument self
    def get_stats(self):
        # Return a description data_as_csv
        return self.data_as_csv.describe()
    
# Instantiate DataShell as us_data_shell
us_data_shell = DataShell(us_life_expectancy)

# Print the output of your objects get_stats method
print(us_data_shell.get_stats())

# =============================================================================
# Inheritance & Polymorphism
# =============================================================================

#Inheritance - A class that takes on attributes of from another class, 'parent' 
#class and adds some more of it's own functionality 


# Create a class Animal
class Animal:
	def __init__(self, name):
		self.name = name

# Create a class Mammal, which inherits from Animal
class Mammal(Animal):
	def __init__(self, name, animal_type):
		self.animal_type = animal_type

# Create a class Reptile, which also inherits from Animal
class Reptile(Animal):
	def __init__(self, name, animal_type):
		self.animal_type = animal_type

# Instantiate a mammal with name 'Daisy' and animal_type 'dog': daisy
daisy = Mammal('Daisy', 'dog')

# Instantiate a reptile with name 'Stella' and animal_type 'alligator': stella
stella = Reptile('Stella', 'alligator')

# Print both objects
print(daisy)
print(stella)

# Another example 

# Create a class Vertebrate
class Vertebrate:
    spinal_cord = True
    def __init__(self, name):
        self.name = name

# Create a class Mammal, which inherits from Vertebrate
class Mammal(Vertebrate):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type
        self.temperature_regulation = True

# Create a class Reptile, which also inherits from Vertebrate
class Reptile(Vertebrate):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type
        self.temperature_regulation = False

# Instantiate a mammal with name 'Daisy' and animal_type 'dog': daisy
daisy = Mammal('Daisy', 'dog')

# Instantiate a reptile with name 'Stella' and animal_type 'alligator': stella
stella = Reptile('Stella', 'alligator')

# Print stella's attributes spinal_cord and temperature_regulation
print("Stella Spinal cord: " + str(stella.spinal_cord))
print("Stella temperature regulation: " + str(stella.temperature_regulation))

# Print daisy's attributes spinal_cord and temperature_regulation
print("Daisy Spinal cord: " + str(daisy.spinal_cord))
print("Daisy temperature regulation: " + str(daisy.temperature_regulation))

# =============================================================================
# Inheritance with DataShells
# =============================================================================

# DataShell class

class DataShell:
    def __init__(self, filename):
        self.filename = filename

    def create_datashell(self):
        data_array = np.genfromtxt(self.filename, delimiter=',', dtype=None)
        self.array = data_array
        return self.array
    
    def show_shell(self):
        print(self.array)

    def rename_column(self, old_colname, new_colname):
        for index, value in enumerate(self.array[0]):
            if value == old_colname.encode('UTF-8'):
                self.array[0][index] = new_colname
        return self.array

    def five_figure_summary(self,col_position):
        statistics = stats.describe(self.array[1:,col_position].astype(np.float))
        return f"Five-figure stats of column {col_position}: {statistics}"

# DataStDev will now inherit from DataShell
        
class DataStDev(DataShell):
    def __init__(self,filename):
        DataShell.filename = filename
        
    def get_stdev(self,col_position):
        column = self.array[1:,col_position].astype(np.float)
        stdev = np.ndarray.std(column,axis=0)
        return f"Standard Deviation of column {col_position}: {stdev}"

#So now we have a class that takes in all the other aspects of DataShell, but 
#has standard deviation also. 
#
#In the constructor usually, we have self.filename but now we have DataShell.filename.
#When we initialize the class with mtcars.csv, what we are really doing is pulling in
#the DataShell to do the work of initialization and running through all of our
#activations. 

my_st_dev_shell = DataStDev(us_life_expectancy)

my_st_dev_shell.create_datashell()

my_st_dev_shell.get_stdev(3)

# =============================================================================
# Abstract Class DataShell
# =============================================================================

# Load numpy as np and pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:
    def __init__(self, inputFile):
        self.file = inputFile

# Instantiate DataShell as my_data_shell
my_data_shell = DataShell(us_life_expectancy)

# Print my_data_shell
print(my_data_shell)

# =============================================================================
# Abstract Class DataShell II
# =============================================================================


# Load numpy as np and pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:
    def __init__(self, inputFile):
        self.file = inputFile

# Create class CsvDataShell, which inherits from DataShell
class CsvDataShell(DataShell):
    # Initialization method with arguments self, inputFile
    def __init__(self, inputFile):
        # Instance variable data
        self.data = pd.read_csv(inputFile)

# Instantiate CsvDataShell as us_data_shell, passing us_life_expectancy as argument
us_data_shell = CsvDataShell(us_life_expectancy)

# Print us_data_shell.data
print(us_data_shell.data)

# =============================================================================
# Composition
# =============================================================================

import pandas 

class DataShell:
    def __init__(self, filename):
        self.filename = filename
        
    def create_datashell(self):
        data_array = np.genfromtxt(self.filename, delimiter=',', dtype=None)
        self.array = data_array
        return self.array
    
class DataShellComposed:
    def __init__(self, filename):
        self.filename = filename
    
    def create_datashell(self):
        self.df = pandas.read_csv()
        return self.df

# =============================================================================
# Composition and Inheritance I
# =============================================================================

# Define abstract class DataShell
class DataShell:
    # Class variable family
    family = 'DataShell'
    # Initialization method with arguments, and instance variables
    def __init__(self, name, filepath): 
        self.name = name
        self.filepath = filepath

# Define class CsvDataShell      
class CsvDataShell(DataShell):
    # Initialization method with arguments self, name, filepath
    def __init__(self, name, filepath):
        # Instance variable data
        self.data = pd.read_csv(filepath)
        # Instance variable stats
        self.stats = self.data.describe()

# Instantiate CsvDataShell as us_data_shell
us_data_shell = CsvDataShell("US", us_life_expectancy)

# Print us_data_shell.stats
print(us_data_shell.stats)

# =============================================================================
# Composition and Inheritance II
# =============================================================================

france_life_expectancy = 'https://assets.datacamp.com/production/repositories/2097/datasets/e3620bc33a17d7ce5cf0ae87e723171284c81df3/france_life_expectancy.csv'

# Define abstract class DataShell
class DataShell:
    family = 'DataShell'
    def __init__(self, name, filepath): 
        self.name = name
        self.filepath = filepath

# Define class CsvDataShell
class CsvDataShell(DataShell):
    def __init__(self, name, filepath):
        self.data = pd.read_csv(filepath)
        self.stats = self.data.describe()

# Define class TsvDataShell
class TsvDataShell(DataShell):
    # Initialization method with arguments self, name, filepath
    def __init__(self, name, filepath):
        # Instance variable data
        self.data = pd.read_table(filepath)
        # Instance variable stats
        self.stats = self.data.describe()

# Instantiate CsvDataShell as us_data_shell, print us_data_shell.stats
us_data_shell = CsvDataShell("US", us_life_expectancy)
print(us_data_shell.stats)

# Instantiate TsvDataShell as france_data_shell, print france_data_shell.stats
france_data_shell = TsvDataShell('France', france_life_expectancy)
print(france_data_shell.stats)


