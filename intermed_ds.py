#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 19:30:54 2018

@author: amin
"""

### Data visualization

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


year = list(range(1950, 2101))

pop = [2.53,2.57,2.62,2.67,2.71,2.76,2.81,2.86,2.92,2.97,3.03,3.08,3.14,3.2,3.26,
       3.33,3.4,3.47,3.54,3.62,3.69,3.77,3.84,3.92,4.0,4.07,4.15,4.22,4.3,4.37,
       4.45,4.53,4.61,4.69, 4.78,4.86,4.95,5.05,5.14,5.23,5.32,5.41,5.49,5.58,
       5.66,5.74,5.82,5.9,5.98,6.05,6.13,6.2,6.28,6.36,6.44,6.51,6.59,6.67,6.75,6.83,6.92,7.0,7.08,7.16,7.24,
       7.32,7.4,7.48,7.56,7.64,7.72,7.79,7.87,7.94,8.01,8.08,8.15,8.22,8.29,
       8.36,8.42,8.49,8.56,8.62,8.68,8.74,8.8,8.86,8.92,8.98,9.04,9.09,9.15,
       9.2,9.26,9.31,9.36,9.41,9.46,9.5,9.55,9.6,9.64,9.68,9.73,9.77,9.81,9.85,9.88,9.92,9.96,
       9.99,10.03,10.06,10.09,10.13,10.16,10.19,10.22,10.25,10.28,10.31,10.33,10.36,10.38,10.41,10.43,
       10.46,10.48,10.5,10.52,10.55,10.57,10.59,10.61,10.63,10.65,10.66,10.68,
       10.7,10.72,10.73,10.75,10.77,10.78,10.79,10.81,10.82,10.83,10.84,10.85]

# Print the last item from year and pop
print(year[-1])
print(pop[-1])


# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year, pop)

# Display the plot with plt.show()
plt.show()

gdp_cap = [974.5803384,5937.029525999998,6223.367465,4797.231267,12779.37964,
           34435.367439999995,36126.4927,29796.04834,1391.253792,33692.60508,
           1441.284873,3822.137084,7446.298803,12569.85177,9065.800825,10680.79282,
           1217.032994,430.0706916,1713.778686,2042.09524,36319.23501,706.016537,1704.063724,13171.63885,
           4959.114854,7006.580419,986.1478792,277.5518587,3632.557798,9645.06142,
           1544.750112,14619.222719999998,8948.102923,22833.30851,35278.41874,2082.4815670000007,
           6025.3747520000015,6873.262326000001,5581.180998,5728.353514,12154.08975,
           641.3695236000002,690.8055759,33207.0844,30470.0167,13206.48452,752.7497265,
           32170.37442,1327.60891,27538.41188,5186.050003,942.6542111,579.2317429999998,
           1201.637154,3548.3308460000007,39724.97867,18008.94444,36180.78919,2452.210407,3540.651564,
           11605.71449,4471.061906,40675.99635,25523.2771,28569.7197,7320.8802620000015,
           31656.06806,4519.461171,1463.249282,1593.06548,23348.139730000006,47306.98978,
           10461.05868,1569.331442,414.5073415,12057.49928,1044.770126,759.3499101,
           12451.6558,1042.581557,1803.151496,10956.99112,11977.57496,3095.7722710000007,9253.896111,3820.17523,
           823.6856205,944.0,4811.060429,1091.359778,36797.93332,25185.00911,2749.320965,
           619.6768923999998,2013.977305,49357.19017,22316.19287,2605.94758,9809.185636,
           4172.838464,7408.905561,3190.481016,15389.924680000002,20509.64777,
           19328.70901,7670.122558,10808.47561,863.0884639000002,1598.435089,21654.83194,1712.472136,9786.534714,
           862.5407561000002,47143.17964,18678.31435,25768.25759,926.1410683,9269.657808,
           28821.0637,3970.095407,2602.394995,4513.480643,33859.74835,37506.41907,4184.548089,28718.27684,
           1107.482182,7458.396326999998,882.9699437999999,18008.50924,
           7092.923025,8458.276384,1056.380121,33203.26128,42951.65309,10611.46299,11415.80569,2441.576404,
           3025.349798,2280.769906,1271.211593,469.70929810000007]

life_exp = [43.828,76.423,72.301,42.731,75.32,81.235,79.829,75.635,64.062,
            79.441,56.728,65.554,74.852,50.728,72.39,73.005,52.295,49.58,
            59.723,50.43,80.653,44.74100000000001,50.651,
            78.553,72.961,72.889,65.152,46.462,55.322,78.782,48.328,75.748,
            78.273,76.486,78.332,54.791,72.235,74.994,71.33800000000002,
            71.878,51.57899999999999,58.04,
            52.947,79.313,80.657,56.735,59.448,79.406,60.022,79.483,70.259,
            56.007,46.38800000000001,60.916,70.19800000000001,82.208,
            73.33800000000002,81.757,64.69800000000001,
            70.65,70.964,59.545,78.885,80.745,80.546,72.567,82.603,72.535,
            54.11,67.297,78.623,77.58800000000002,71.993,42.592,45.678,73.952,
            59.44300000000001,48.303,74.241,
            54.467,64.164,72.801,76.195,66.803,74.543,71.164,42.082,62.069,
            52.90600000000001,63.785,79.762,80.204,72.899,56.867,46.859,
            80.196,75.64,65.483,75.53699999999998,
            71.752,71.421,71.688,75.563,78.098,78.74600000000002,76.442,72.476,
            46.242,65.528,72.777,63.062,74.002,42.56800000000001,79.972,74.663,
            77.926,48.159,49.339,80.941,
            72.396,58.556,39.613,80.884,81.70100000000002,74.143,78.4,52.517,
            70.616,58.42,69.819,73.923,71.777,51.542,79.425,78.242,76.384,73.747,
            74.249,73.422,62.698,42.38399999999999,43.487]

pop_2007 = [31.889923,3.600523,33.333216,12.420476,40.301927,20.434176,8.199783,
            0.708573,150.448339,10.392226,8.078314,9.119152,4.552198,1.639131,190.010647,7.322858,14.326203,
            8.390505,14.131858,17.696293,33.390141,4.369038,10.238807,16.284741,
            1318.683096,44.22755,0.71096,64.606759,3.80061,4.133884,18.013409,
            4.493312,11.416987,10.228744,5.46812,0.496374,9.319622,13.75568,
            80.264543,6.939688,0.551201,4.906585,76.511887,5.23846,61.083916,
            1.454867,1.688359,82.400996,22.873338,10.70629,
            12.572928,9.947814,1.472041,8.502814,7.483763,6.980412,9.956108,
            0.301931,1110.396331,223.547,69.45357,27.499638,4.109086,6.426679,
            58.147733,2.780132,127.467972,
            6.053193,35.610177,23.301725,49.04479,2.505559,3.921278,2.012649,
            3.193942,6.036914,19.167654,13.327079,24.821286,12.031795,3.270065,1.250882,108.700891,
            2.874127,0.684736,33.757175,19.951656,47.76198,2.05508,28.90179,
            16.570613,4.115771,5.675356,12.894865,135.031164,4.627926,3.204897,
            169.270617,3.242173,6.667147,28.674757,91.077287,38.518241,10.642836,3.942491,
            0.798094,22.276056,8.860588,0.199579,27.601038,12.267493,10.150265,6.144562,4.553009,
            5.447502,2.009245,9.118773,43.997828,40.448191,20.378239,42.292929,
            1.133066,9.031088,7.554661,19.314747,23.174294,38.13964,65.068149,
            5.701579,1.056608,10.276158,71.158647,29.170398,60.776238,301.139947,3.447496,
            26.084662,85.262356,4.018332,22.211743,11.746035,12.311143]

### Scatter plot

plt.scatter(gdp_cap, life_exp)

# Put the x-axis on a logarithmic scale
plt.xscale('log')

# Show plot
plt.show()

# Build Scatter plot
plt.scatter(pop, life_exp)

# Show plot
plt.show()

### Histograms

# Create histogram of life_exp data
plt.hist(life_exp)

# Display histogram
plt.show()

# Build histogram with 5 bins
plt.hist(life_exp, bins = 5)

# Show and clean up plot
plt.show()
plt.clf()

# Build histogram with 20 bins
plt.hist(life_exp, bins =20)

# Show and clean up again
plt.show()
plt.clf()

life_exp1950 = [28.8,55.23,43.08,30.02,62.48,69.12,66.8,50.94,37.48,68.0,
                38.22,40.41,53.82,47.62,50.92,59.6,31.98,39.03,39.42,38.52,
                68.75,35.46,38.09,54.74,44.0,50.64,40.72,39.14,42.11,57.21,
                40.48,61.21,59.42,66.87,70.78,34.81,45.93,48.36,41.89,45.26,
                34.48,35.93,34.08,66.55,67.41,37.0,30.0,67.5,43.15,65.86,
                42.02,33.61,32.5,37.58,41.91,60.96,64.03,72.49,7.37,37.47,44.87,
                45.32,66.91,65.39,65.94,58.53,63.03,43.16,42.27,50.06,47.45,
                55.56,55.93,42.14,38.48,42.72,36.68,36.26,48.46,33.68,40.54,
                50.99,50.79,42.24,59.16,42.87,31.29,36.32,41.72,36.16,
                72.13,69.39,42.31,37.44,36.32,72.67,37.58,43.44,55.19,62.65,
                43.9,47.75,61.31,59.82,64.28,52.72,61.05,40.0,46.47,39.88,
                37.28,58.0,30.33,60.4,64.36,65.57,32.98,45.01,64.94,57.59,
                38.64,41.41,71.86,69.62,45.88,58.5,41.22,50.85,38.6,59.1,44.6,
                43.58,39.98,69.18,68.44,66.07,55.09,40.41,43.16,32.55,42.04,48.45]

#### Comparing

# Histogram of life_exp, 15 bins
plt.hist(life_exp, bins =15)

# Show and clear plot
plt.show()
plt.clf()

# Histogram of life_exp1950, 15 bins
plt.hist(life_exp1950, bins =15)

# Show and clear plot again
plt.show()
plt.clf()

### Labels

# Add axis labels
# Basic scatter plot, log scale
plt.scatter(gdp_cap, life_exp)
plt.xscale('log') 

# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'

plt.xlabel(xlab)
plt.ylabel(ylab)

# Definition of tick_val and tick_lab
tick_val = [1000, 10000, 100000]
tick_lab = ['1k', '10k', '100k']

# Adapt the ticks on the x-axis
plt.xticks(tick_val, tick_lab)

# Add title
plt.title(title)

# After customizing, display the plot
plt.show()

### Sizes 

# Store pop as a numpy array: np_pop
np_pop = np.array(pop)

# Double np_pop
np_pop_2007 = pop_2007 * 2


# Update: set s argument to np_pop
plt.scatter(gdp_cap, life_exp, s = np_pop_2007  )

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])

# Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Add grid() call
plt.grid(True)

# Display the plot
plt.show()


### Dictionaries

### Motivation
# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# Get index of 'germany': ind_ger
ind_ger = countries.index("germany")

# Use ind_ger to print out capital of Germany
print(capitals[ind_ger])

### CReating dictionary
# From string in countries and capitals, create dictionary europe
europe = { 'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Print europe
print(europe)

# Print out the keys in europe
print(europe.keys())

# Print out value that belongs to key 'norway'
print(europe['norway'])

### Dictionary manipuation

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Add italy to europe
europe["italy"] = "rome"

# Print out italy in europe
print("italy" in europe)

# Add poland to europe

europe["poland"] = "warsaw"
# Print europe
print(europe)

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw',
          'australia':'vienna' }

# Update capital of germany
europe["germany"] = 'berlin'

# Remove australia
del(europe['australia'])

# Print europe
print(europe)

# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }


# Print out the capital of France
print(europe['france']['population'])

# Create sub-dictionary data
data = {'capital':'rome', 'population': 59.83}

# Add data to europe under key 'italy'
europe['italy'] = data

# Print europe
print(europe)

### Creating tables from dictionaries

dict = {
 "country":["Brazil", "Russia", "India", "China", "South Africa"],
 "capital":["Brasilia", "Moscow", "New Delhi", "Beijing", "Pretoria"],
 "area":[8.516, 17.10, 3.286, 9.597, 1.221],
 "population":[200.4, 143.5, 1252, 1357, 52.98] }

### Keys are column labels and values (data, column by column)

brics = pd.DataFrame(dict)

### Another example

# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Import pandas as pd
import pandas as pd

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }

# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

# Print cars
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels

# Print cars again
print(cars)


# brics = pd.read_csv("path/to/brics.csv", index_col = 0)

### Square brackets 

#The single bracket version gives a Pandas Series, 
#the double bracket version gives a Pandas DataFrame

# Print out country column as Pandas Series
print(cars['country'])

# Print out country column as Pandas DataFrame
print(cars[['country']])

# Print out DataFrame with country and drives_right columns
print(cars[['country', 'drives_right']])

#Square brackets can do more than just selecting columns. 
#You can also use them to get rows, or observations, from a DataFrame.

# Print out first 3 observations
print(cars[0:3])

# Print out fourth, fifth and sixth observation
print(cars[3:7])

# =============================================================================
# ### loc and iloc
# #With loc and iloc you can do practically any data selection operation on 
# #DataFrames you can think of. loc is label-based, which means that you have to 
# #specify rows and columns based on their row and column labels. iloc is integer 
# #index based, so you have to specify rows and columns by their integer index
# 
# =============================================================================
###Comes as a python series object because of the single square brackets
cars.loc['RU']
cars.iloc[4]

cars.loc[['RU']]
cars.iloc[[4]]

### Comes as a dataframe because of the double square brackets
cars.loc[['RU', 'AUS']]
cars.iloc[[4, 1]]


# Print out observation for Japan
print(cars.loc[['JAP']])

# Print out observations for Australia and Egypt
print(cars.loc[['AUS', 'EG']])
print(cars.iloc[[1,-1]])

#loc and iloc also allow you to select both rows and columns from a DataFrame. 

cars.loc['IN', 'cars_per_cap']

cars.iloc[3, 0]

cars.loc[['IN', 'RU'], 'cars_per_cap']
cars.iloc[[3, 4], 0]

cars.loc[['IN', 'RU'], ['cars_per_cap', 'country']]
cars.iloc[[3, 4], [0, 1]]

# Print out drives_right value of Morocco
print(cars.loc['MOR', 'drives_right'])

# Print sub-DataFrame
print(cars.loc[['RU','MOR'], ['country', 'drives_right']])


# =============================================================================
# #It's also possible to select only columns with loc and iloc. In both cases,
# # you simply put a slice going from beginning to end in front of the comma
# 
# =============================================================================
cars.loc[:, 'country']
cars.iloc[:, 1]

cars.loc[:, ['country','drives_right']]
cars.iloc[:, [1, 2]]

# Print out drives_right column as Series
print(cars.loc[:,'drives_right'])

# Print out drives_right column as DataFrame
print(cars.loc[:,['drives_right']])

# Print out cars_per_cap and drives_right as DataFrame
print(cars.loc[:, ['cars_per_cap', 'drives_right']])


## Equality

# Comparison of booleans
True == False

# Comparison of integers
-5 * 15 != 75

# Comparison of strings
"pyscript" == "PyScript"

# Compare a boolean with an integer
True == 1

## Compare arrays
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than or equal to 18
print(my_house >= 18)

# my_house less than your_house
print(my_house < your_house)

# Define variables
my_kitchen = 18.0
your_kitchen = 14.0

# my_kitchen bigger than 10 and smaller than 18?
print(my_kitchen > 10 and my_kitchen < 18)

# my_kitchen smaller than 14 or bigger than 17?
print(my_kitchen < 14 or my_kitchen > 17)

# Double my_kitchen smaller than triple your_kitchen?
print(my_kitchen * 2 < 3 * your_kitchen)

x = 8
y = 9
not(not(x < 3) and not(y > 14 or y > 10))

### Boolean operators with Numpy

# =============================================================================
# Before, the operational operators like < and >= worked with Numpy 
# arrays out of the box. Unfortunately, this is not true for the boolean 
# operators and, or, and not.
# 
# To use these operators with Numpy, you will need np.logical_and(), 
# np.logical_or() and np.logical_not()
# =============================================================================

# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10
print(np.logical_or(my_house > 18.5, my_house < 10))

# Both my_house and your_house smaller than 11
print(np.logical_and(my_house < 11, your_house < 11))

### Conditions

# Define variables
room = "bed"
area = 14.0

# if-elif-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
elif room == "bed":
    print("looking around in the bedroom.")
else :
    print("looking around elsewhere.")

# if-elif-else construct for area
if area > 15 :
    print("big place!")
elif area > 10:
    print("medium size, nice!")
else :
    print("pretty small.")
    
# =============================================================================
# Filtering Pandas DataFrame
# =============================================================================


### Step 1: Get Column
brics["area"] 

## Alternatives
brics.loc[:,"area"]
brics.iloc[:,2]

### Step 2: Compare

is_huge = brics["area"] > 8

### Step 3: Subset DF

brics[is_huge] 

### In short

brics[brics["area"] > 8] 

### Boolean operators
np.logical_and(brics["area"] > 8, brics["area"] < 10) 

brics[np.logical_and(brics["area"] > 8, brics["area"] < 10)] 

# Extract drives_right column as Series: dr
dr = cars["drives_right"]

# Use dr to subset cars: sel
sel = cars[dr]

# Print sel
print(sel)

# Create car_maniac: observations that have a cars_per_cap over 500
cpc = cars["cars_per_cap"]
many_cars = cpc >500
car_maniac = cars[many_cars]

# Print car_maniac
print(car_maniac)

# Create medium: observations with cars_per_cap between 100 and 500
medium = cars[np.logical_and(cars["cars_per_cap"] > 100, cars["cars_per_cap"]<500)]

# Print medium
print(medium)

# =============================================================================
# while: warming up
# The while loop is like a repeated if statement. The code is executed over and
#  over again, as long as the condition is True.
# =============================================================================


error = 50.0
while error > 1 :
    error = error / 4
    print(error)

# Initialize offset
offset = 8

# Code the while loop
while offset != 0:
    print("correcting...")
    offset = offset - 1
    print(offset)

# =============================================================================
# Adding conditions!
# The while loop that corrects the offset is a good start, but what if offset is 
# negative? 
# You can try to run the sample code on the right where offset is initialized to -6, 
# but your sessions will be disconnected. The while loop will never stop running, 
# because offset will be further decreased on every run. 
# offset != 0 will never become False and the while loop continues forever
# =============================================================================
# Initialize offset
offset = - 6
# Code the while loop
while offset != 0 :
    print("correcting...")
    if offset > 0:
        offset = offset - 1
    else:
        offset = offset + 1
    print(offset)
    
### For loops
    
fam = [1.73, 1.68, 1.71, 1.89]
for height in fam : 
    print(height)
    
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for i in areas:
    print(i)
    
# =============================================================================
# Indexes and values (1)
# Using a for loop to iterate over a list only gives you access to every 
# list element in each run, one after the other. If you also want to access 
# the index information, so where the list element you're iterating over is 
# located, you can use enumerate()
# =============================================================================
    
fam = [1.73, 1.68, 1.71, 1.89]
for index, height in enumerate(fam) :
    print("person " + str(index) + ": " + str(height))

# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate() and update print()
for index, a in enumerate(areas) :
    print("room " + str(index + 1) + ": " + str(a))
    
# =============================================================================
# Loop over list of lists
# Remember the house variable from the Intro to Python course? Have a look at its
# definition on the right. It's basically a list of lists, where each sublist 
# contains the name and area of a room in your house.
# =============================================================================

# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# Build a for loop from scratch
for x in house:
    print("the " + x[0] + " is " + str(x[1]) + " sqm")
    
# =============================================================================
#     
# Loop over dictionary
# In Python 3, you need the items() method to loop over a dictionary:
# =============================================================================

world = { "afghanistan":30.55, 
          "albania":2.77,
          "algeria":39.21 }

for key, value in world.items() :
    print(key + " -- " + str(value))

### The order changed because dictionaries are unordered 
 
### Numpy arrays
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
bmi = np_weight / np_height ** 2

for val in bmi :
 print(val)

### 2D numpy arrays
 
meas = np.array([np_height, np_weight])
    
for val in meas :
 print(val) 

## you see that it just list the arrays

for val in np.nditer(meas): 
    print(val)

# =============================================================================
# Recap
# ● Dictionary
# ● for key, val in my_dict.items() :
# ● Numpy array
# ● for val in np.nditer(my_array) :
# =============================================================================

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe
for key, value in europe.items():
    print("the capital of " + key + " is " + value) 
    
    
for lab, row in brics.iterrows():
    print(lab)
    print(row)
  
### This simply prints only the column names 
for i in brics:
    print(i)
# =============================================================================
# Selective print
# =============================================================================
    
for lab, row in brics.iterrows():
    print(str(lab) + ": " + row['capital'])

# =============================================================================
# Add column
# =============================================================================

## This creates a series on every iteration
for lab, row in brics.iterrows() :
    brics.loc[lab, "name_length"] = len( row["country"])
    
print(brics)
    
# =============================================================================
# apply
# =============================================================================

brics["name_length"] = brics["country"].apply(len)

print(brics)

# =============================================================================
# Loop over DataFrame (1)
# Iterating over a Pandas DataFrame is typically done with the iterrows() method. 
# Used in a for loop, every observation is iterated over and on every iteration 
# the row label and actual row contents are available:
# 
# =============================================================================

# Iterate over rows of cars
for lab, row in cars.iterrows():
    print(lab)
    print(row)


# =============================================================================
# Loop over DataFrame (2)
# The row data that's generated by iterrows() on every run is a Pandas Series.
#  This format is not very convenient to print out. Luckily, you can easily 
#  select variables from the Pandas Series using square brackets:
# =============================================================================

for lab, row in cars.iterrows() :
    print(lab + ": " + str(row['cars_per_cap']))


# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows():
    cars.loc[lab, "COUNTRY"] = row['country'].upper()

# Print cars
print(cars)

# =============================================================================
# Add column (2)
# Using iterrows() to iterate over every observation of a Pandas DataFrame is 
# easy to understand, but not very efficient. On every iteration, 
# you're creating a new Pandas Series.
# 
# If you want to add a column to a DataFrame by calling a function on another 
# column, the iterrows() method in combination with a for loop is not the preferred 
# way to go. Instead, you'll want to use apply().
# =============================================================================

for lab, row in brics.iterrows() :
    brics.loc[lab, "name_length"] = len(row["country"])

brics["name_length"] = brics["country"].apply(len)

# =============================================================================
# We can do a similar thing to call the upper() method on every name in the country column. 
# However, upper() is a method, so we'll need a slightly different approach:
# 
# =============================================================================

cars["COUNTRY"] = cars['country'].apply(str.upper)


# =============================================================================
# Random float
# Randomness has many uses in science, art, statistics, cryptography, gaming, 
# gambling, and other fields. You're going to use randomness to simulate a game.
# 
# All the functionality you need is contained in the random package, a sub-package 
# of numpy. In this exercise, you'll be using two functions from this package:
# 
# seed(): sets the random seed, so that your results are the reproducible between 
# simulations. As an argument, it takes an integer of your choosing. 
# If you call the function, no output will be generated.
# rand(): if you don't specify any arguments, it generates a random float between zero and one.
# 
# =============================================================================

# Set the seed
np.random.seed(123)

# Generate and print random float
print(np.random.rand())

# Use randint() to simulate a dice
print(np.random.randint(1, 7))

# Use randint() again
print(np.random.randint(1, 7))

# =============================================================================
# Set the seed
np.random.seed(123)

# Starting step
step = 50

# Roll the dice
dice = np.random.randint(1,7)

# Finish the control construct
if dice <= 2 :
    step = step - 1
elif step <= 5 :
    step = step + 1
else :
    step = step + np.random.randint(1,7)

# Print out dice and step
print(dice)
print(step)

#### Random walk

np.random.seed(456)
outcomes = [] # initialize an empty list

for i in range(10):
    coin = np.random.randint(0,2)
    if coin == 0:
        outcomes.append('heads')
    else:
        outcomes.append('tails')

print(outcomes)

# =============================================================================
# #### This list is random, but not a random walk because the items in the list
# #are not based on the previous ones. It's just a bunch of random steps
# 
# We could turn this into a random walk by tracking the total number of tails while
# you are simulating a game
# =============================================================================

np.random.seed(456)
tails =[0]
for x in range(10):
    coin = np.random.randint(0,2)
    print(coin)
    tails.append(tails[x] + coin)

print(tails)

##### The final element in the list tells you how often tails was thrown. 
#This is how you convert a bunch of random steps into random walk

# =============================================================================

# Initialize random_walk
random_walk = [0]

for x in range(100) :
    # Set step: last element in random_walk
    step = random_walk[-1]

    # Roll the dice
    dice = np.random.randint(1,7)

    # Determine next step
    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    # append next_step to random_walk
    random_walk.append(step)

# Print random_walk
print(random_walk)

# =============================================================================
# How low can you go?
# Things are shaping up nicely! You already have code that calculates your 
# location in the Empire State Building after 100 dice throws. 
# However, there's something we haven't thought about - you can't go below 0!
# 
# A typical way to solve problems like this is by using max(). 
# If you pass max() two arguments, the biggest one gets returned. 
# For example, to make sure that a variable x never goes below 10 when you 
# decrease it, you can use:
# =============================================================================

x =5

x = max(10, x - 1)

print(x)


# =============================================================================
# Visualize the walk
# Let's visualize this random walk! Remember how you could use matplotlib 
# to build a line plot?
# 
# import matplotlib.pyplot as plt
# plt.plot(x, y)
# plt.show()
# The first list you pass is mapped onto the x axis and the second 
# list is mapped onto the y axis.
# 
# If you pass only one argument, Python will know what to do and will use the 
# index of the list to map onto the x axis, and the values in the list onto the y axis
# 
# =============================================================================

# Plot random_walk
plt.plot(random_walk)

# Show the plot
plt.show()


# =============================================================================
# Simulate multiple walks
# A single random walk is one thing, but that doesn't tell you 
# if you have a good chance at winning the bet.
# 
# To get an idea about how big your chances are of reaching 60 steps, 
# you can repeatedly simulate the random walk and collect the results. 
# That's exactly what you'll do in this exercise.
# 
# The sample code already sets you off in the right direction. 
# Another for loop is wrapped around the code you already wrote. 
# It's up to you to add some bits and pieces to make sure all of the 
# results are recorded correctly.
# =============================================================================
### The coin example
np.random.seed(123)
final_tails = []
for x in range(100000) :
    tails = [0]
    for x in range(10) :
        coin = np.random.randint(0,2)
        tails.append(tails[x] + coin)
    final_tails.append(tails[-1])

plt.hist(final_tails, bins = 10)
plt.show()


# Initialize all_walks
all_walks = []

# Simulate random walk 10 times
for i in range(10) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)

    # Append random_walk to all_walks
    all_walks.append(random_walk)

# Print all_walks
print(all_walks)

#Visualize all walks
#all_walks is a list of lists: every sub-list represents a single random walk. 
#If you convert this list of lists to a Numpy array, you can start making interesting plots!
np_aw = np.array(all_walks)

# Plot np_aw and show
plt.plot(np_aw)
plt.show()

# Clear the figure
plt.clf()

#Transpose np_aw by calling np.transpose() on np_aw. 
#Call the result np_aw_t. 
#Now every row in np_all_walks represents the position after 1 throw for the 10 random walks.

np_aw_t = np.transpose(np_aw)

plt.plot(np_aw_t)
plt.show()

# =============================================================================
# There's still something we forgot! 
# You're a bit clumsy and you have a 0.1% chance of falling down. 
# That calls for another random number generation. 
# Basically, you can generate a random float between 0 and 1. 
# If this value is less than or equal to 0.001, you should reset step to 0.
# 
# =============================================================================

# Initialize all_walks
all_walks = []

# Simulate random walk 10 times
for i in range(250) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
         # Implement clumsiness
        if np.random.rand() <= 0.001:
            step = 0
        random_walk.append(step)

    # Append random_walk to all_walks
    all_walks.append(random_walk)
    
# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
plt.show()

# =============================================================================
# Plot the distribution
# All these fancy visualizations have put us on a sidetrack. 
# We still have to solve the million-dollar problem: 
# What are the odds that you'll reach 60 steps high on the Empire State Building?
# 
# Basically, you want to know about the end points of all the random walks you've simulated. 
# These end points have a certain distribution that you can visualize with a histogram
# =============================================================================

# Simulate random walk 500 times
all_walks = []
for i in range(500) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        if np.random.rand() <= 0.001 :
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))

#From np_aw_t, select the last row. This contains the endpoint of all 500
# random walks you've simulated. Store this Numpy array as ends

# Select last row from np_aw_t: ends
ends = np_aw_t[-1]

# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()

# =============================================================================
# Well then, what's the estimated chance that you'll reach 60 steps high if you 
# play this Empire State Building game? 
# =============================================================================

np.mean(ends >= 60)

###Alternatively

len(ends[ends >= 60]) / len(ends)




















