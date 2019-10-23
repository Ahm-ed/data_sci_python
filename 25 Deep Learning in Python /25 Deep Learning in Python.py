#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:55:27 2019

@author: amin
"""

# =============================================================================
# Interactions
# ● Neural networks account for interactions really well
# ● Deep learning uses especially powerful neural networks
# ● Text
# ● Images
# ● Videos
# ● Audio
# ● Source code
# =============================================================================

# =============================================================================
# Forward propagation
# 
# Multiply - add process
# ● Dot product
# ● Forward propagation for one data point at a time
# ● Output is the prediction for that data point
# =============================================================================

import numpy as np

input_data = np.array([2, 3]) # These are the data points for 1 person. 2 for predictor1 and 3 for predictor 2

weights = {'node_0': np.array([1, 1]),
           'node_1': np.array([-1, 1]),
           'output': np.array([2, -1])}

# the weights are the numbers going to the node

node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()

hidden_layer_values = np.array([node_0_value, node_1_value])
print(hidden_layer_values)

output = (hidden_layer_values * weights['output']).sum()
print(output) 

# so the prediction will be 9 

# -------------------------another example-------------------------------------

input_data = np.array([3, 5])

weights = {'node_0': np.array([2, 4]), 
           'node_1': np.array([ 4, -5]), 
           'output': np.array([2, 7])}

# Calculate node 0 value: node_0_value
node_0_value = ( input_data * weights['node_0']).sum()

# Calculate node 1 value: node_1_value
node_1_value = (input_data * weights['node_1']).sum()

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_value, node_1_value])

# Calculate output: output
output = (hidden_layer_outputs * weights['output']).sum()

# Print output
print(output)

# predicted will be -39

# =============================================================================
# Activation functions
# =============================================================================

#For neural networks to achieve their maximum predictive power, we have to apply 
#activation functions in the hidden layer. An activation function allows the linear
#model to capture non-linearities. 
#
#An activation function is something that is applied to the value coming in a
#node which then transforms it into a value stored in that node or the node 
#output. 
#
#For a long time, an s-shaped function called tanh was a popular activation 
#function. If we use the tanh activation function, the first example's first nodes value 
#for example, will not be tanh(5) which is very close to 1. Today, the standard
#in both industry and research applications is something called the ReLU or 
#Rectified Linear Activation function.

# Let's add activation functions - tanh

import numpy as np

input_data = np.array([-1, 2])

weights = {'node_0': np.array([3, 3]),
           'node_1': np.array([1, 5]),
           'output': np.array([2, -1])}

node_0_input = (input_data * weights['node_0']).sum()
node_0_output = np.tanh(node_0_input)

node_1_input = (input_data * weights['node_1']).sum()
node_1_output = np.tanh(node_1_input)

hidden_layer_output = np.array([node_0_output, node_1_output])

output = (hidden_layer_output * weights['output']).sum()
print(output)

# =============================================================================
# The Rectified Linear Activation Function
# 
# An "activation function" is a function applied at each node. 
# It converts the node's input into some output.
# 
# The rectified linear activation function (called ReLU) has been shown to lead 
# to very high-performance networks. This function takes a single number as an 
# input, returning 0 if the input is negative, and the input if the input is positive.
# 
# Here are some examples:
# relu(3) = 3 
# relu(-3) = 0 
# =============================================================================

input_data = np.array([3, 5])

weights = {'node_0': np.array([2, 4]), 
           'node_1': np.array([ 4, -5]), 
           'output': np.array([2, 7])}

def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(input, 0)
    
    # Return the value just calculated
    return(output)

# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output)

#Without this activation function, you would have predicted a negative number! 
#The real power of activation functions will come soon when you start tuning model weights

# ----------------Applying the network to many observations/rows of data------------------
 
input_data = [np.array([3, 5]), 
              np.array([ 1, -1]), 
              np.array([0, 0]), 
              np.array([8, 4])]

weights = {'node_0': np.array([2, 4]), 
           'node_1': np.array([ 4, -5]), 
           'output': np.array([2, 7])}

# predict_with_network() which will generate predictions for multiple data observations
# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    
    # Return model output
    return(model_output)


# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)

#-----------------------------------------------------------------------------
#Deep networks internally build up representations of the patterns in the data 
#that are useful for making predictions. And they find increasingly complex
#patterns as we go through successive hidden layers of the network. 
#
#In this way, neural networks partially replace the need for feature engineering
#or manually creating better predictive features. Deep learning is also sometimes
#called represention learning because subsequent layers build increasingly 
#sophisticated representation of the data until we get to a stage where we can 
#make predictions. This is easiest to understand from an application to images.

# =============================================================================
# Multi-layer neural networks
# =============================================================================

#Forward propagation for a neural network with 2 hidden layers. 
#Each hidden layer has two nodes

input_data = np.array([3, 5])

weights = {'node_0_0': np.array([2, 4]),
           'node_0_1': np.array([ 4, -5]),
           'node_1_0': np.array([-1,  2]),
           'node_1_1': np.array([1, 2]),
           'output': np.array([2, 7])}

def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])

    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])
    
    # Calculate output here: model_output
    model_output = (hidden_1_outputs * weights['output']).sum()
    
    # Return model_output
    return(model_output)

output = predict_with_network(input_data)
print(output)

#------------------------------------------------------------------------------

# The last layers capture the most complex interactions.

# =============================================================================
# The need for optimization
# =============================================================================

# An activation function that returns the input is called an identity function

# =============================================================================
# Coding how weight changes affect accuracy
# =============================================================================

# Now you'll get to change weights in a real network and see how they affect model accuracy!

# The data point you will make a prediction for
input_data = np.array([0, 3])

# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]
            }

# The actual target value, used to calculate the error
target_actual = 3

# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual

# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 0]
            }

# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data, weights_1)

# Calculate error: error_1
error_1 = model_output_1 - target_actual

# Print error_0 and error_1
print(error_0)
print(error_1)

# =============================================================================
# Scaling up to multiple data points
# =============================================================================

# you'll want to measure model accuracy on many points

input_data = [np.array([0, 3]), 
              np.array([1, 2]), 
              np.array([-1, -2]), 
              np.array([4, 0])]

target_actuals = [1, 3, 5, 7]

weights_0 = {'node_0': np.array([2, 1]), 
             'node_1': np.array([1, 2]), 
             'output': np.array([1, 1])}

weights_1 = {'node_0': np.array([2, 1]),
             'node_1': np.array([1. , 1.5]),
             'output': np.array([1. , 1.5])}

def predict_with_network(input_data_point, weights):
    node_0_input = (input_data_point * weights['node_0']).sum()
    node_0_output = relu(node_0_input)
    
    node_1_input = (input_data_point * weights['node_1']).sum()
    node_1_output = relu(node_1_input)
    
    hidden_layer_values = np.array([node_0_output, node_1_output])
    input_to_final_layer = (hidden_layer_values * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    
    return(model_output)

from sklearn.metrics import mean_squared_error

# Create model_output_0 
model_output_0 = []
# Create model_output_1
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))

# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals, model_output_0)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals, model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)

# =============================================================================
# Gradient Descent
# =============================================================================

#Calculating slopes
#
#You're now going to practice calculating slopes. When plotting the mean-squared 
#error loss function against predictions, the slope is 2 * x * (y-xb), or 2 * input_data * error. 
#
#Note that x and b may have multiple numbers (x is a vector for each data point, and b is a vector). 
#In this case, the output will also be a vector, which is exactly what you want.

input_data = np.array([1, 2, 3])
target= 0

weights = np.array([0, 2, 1])

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = target - preds

# Calculate the slope: slope
slope = 2 * input_data * error

# Print the slope
print(slope)

# Improving model weights

#You've just calculated the slopes you need. Now it's time to use those slopes 
#to improve your model. If you add the slopes to your weights, you will move in 
#the right direction. However, it's possible to move too far in that direction. 
#So you will want to take a small step in that direction first, using a lower 
#learning rate, and verify that the model is improving

# Set the learning rate: learning_rate
learning_rate = 0.01

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Update the weights: weights_updated
weights_updated = weights - (slope * learning_rate)

# Get updated predictions: preds_updated
preds_updated = (weights_updated * input_data).sum()

# Calculate updated error: error_updated
error_updated = preds_updated - target

# Print the original error
print(error)

# Print the updated error
print(error_updated)

# =============================================================================
# Making multiple updates to weights
# =============================================================================

import matplotlib.pyplot as plt

def get_error(input_data, target, weights):
    preds = (weights * input_data).sum()
    
    error = preds - target
    return(error)

def get_slope(input_data, target, weights):
    error = get_error(input_data, target, weights)
    slope = 2 * input_data * error
    
    return(slope)
    
input_data = np.array([1, 2, 3])

def get_mse(input_data, target, weights):
    errors = get_error(input_data, target, weights)
    mse = np.mean(errors**2)
    
    return(mse)
    
input_data = np.array([1, 2, 3])
target= 0
weights = np.array([0, 2, 1])

n_updates = 20
mse_hist = []

s = []
w = []
# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)
    
    # Update the weights: weights
    weights = weights - 0.01 * slope
    
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    
    # Append the mse to mse_hist
    mse_hist.append(mse)
    
    s.append(slope)
    
    w.append(weights)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()

#As you can see, the mean squared error decreases as the number of iterations go up.

# =============================================================================
# Creating a keras model
# =============================================================================

import numpy as np
from keras.layers import Dense
from keras.models import Sequential

predictors = np.loadtxt('predictors_data.csv', delimiter=',')

#We always need to specify how many columns are in the input when building 
#a keras model because that's the number of nodes in the input layer. 

n_cols = predictors.shape[1]

#Sequential models require that each layer has weights or connections only to
#the layer coming directly after it in the network diagram. There other exotic 
#models with complex patterns of connection. 

model = Sequential()

#We start adding layers using the add method of the model. We add Dense layers.
#It's called dense because all the nodes in the previous layer connect to all 
#the nodes in the current layer. We then specify the number of nodes as the 
#first positional argument and the activation function we want to use. 

model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))

#In the first layer, we need to specify input shapes which is columns (n_cols)
#followed by comma only meaning any number of rows/datapoint. 

#The last layer has 1 node. This is the output layer. 
model.add(Dense(1))

# -----------------------------------------------------------------------------


# Import necessary modules
import pandas as pd
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential


auto = pd.read_csv('data/auto-mpg.csv', na_values = '?').dropna()
X = auto.drop(['car name', 'mpg'], axis = 1).values
y = auto['mpg']

# Save the number of columns in predictors: n_cols
n_cols = X.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation = 'relu'))

# Add the output layer
model.add(Dense(1))

 
# =============================================================================
# Compiling and fitting a model
# =============================================================================

#After you specify a model, the next step is to compile it. Which sets up 
#the network for optimization, for instance creating an internal function to do
#backpropagation efficiently. 

#The compile argument has two different arguments for you to choose. FIrst, 
#specify which optimizer to use, which controls the learning rate. Adam is 
#an excellent choice as a go-to optimizer. 
#
#Adam adjusts the learning rate as it does gradient descent, to ensure 
#reasonable values through out the weight optimization process. 
#
#The second thing you specify is the loss function. MSE is the most common
#choice for regression problems 

model.compile(optimizer='adam', loss='mean_squared_error')

#After compiling, you can fit it. That is applying backpropagation and 
#gradient descent with your data to update the weights. 

#The fit step looks similar to sci-kit's learn fit, though it has more options

# Even with Adam optimizer, it can improve your optimization if you scale all 
# the data so that each feature, is on average, about similar sized values. 

#One common approach is to subtract each feature by the feature's mean and divide
#by it's standard deviation. 

model.fit(X, y)

# When you run this, you will see some output showing the optimization's progress
# as it fit's the data. 
# It's like a log showing model performaance on the tranining data as we update 
# the model weights 

# -----------------------------------------------------------------------------
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

# Fit the model
model.fit(predictors, target)

# =============================================================================
# Classification models
# =============================================================================

#For categorical problems 
#1. You can set the loss function to 'categorical_crossentropy' instead of 'mean_squared_error' 
#Similar to log loss, small is better. It's a bit hard to intepret. So you can 
#add the argument metric = ['accuracy'] to compile for easy to understand diagnostics. 
#This will basically print out the accuracy score at the end of each epoch, 
#which makes it easier to understand the models progress. 
#
#2. Need to modify the last layer so that it has a separate node for each 
#potential outcome. You also change the activation funtion to 'softmax'. 
#The softmax activation function ensures that the predictions sum to 1, so 
#they can be interpreted like probabilities. 
#
#Outcomes(target) in a single column is common. But in general we will want to
#convert categoricals in Keras to a format with a separate column for each output. 
#Keras has a function to do that. 

#This format is consistent with the fact that your model will have a separate 
#node for in the output for each possible class. 

# so it will have a dummy variable structure (one-hot encoding)

from keras.utils import to_categorical

data = pd.read_csv('data/diabetes.csv')
predictors = data.drop(['Outcome'], axis=1).as_matrix()
target = to_categorical(data.Outcome)

n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(predictors, target)

# =============================================================================
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

df = pd.read_csv('data/titanickaggle.csv').drop(['Name', 'Cabin','Ticket', 'PassengerId','Sex','Embarked'], axis=1)

predictors = df.drop(['Survived'], axis=1).as_matrix()
n_cols = predictors.shape[1]
# Convert the target to categorical: target
target = to_categorical(df.Survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fit the model
model.fit(predictors, target)

# =============================================================================
# Saving, reloading and using your Model
# =============================================================================
from keras.models import load_model

model.save('data/models/model_file.h5')

my_model = load_model('data/models/model_file.h5')

predictions = my_model.predict(data_to_predict_with)

probability_true = predictions[:,1]

#Verifying model structure

my_model.summary()

# =============================================================================
# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(predictors, target)

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)



















