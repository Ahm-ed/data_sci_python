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

input_data = np.array([2, 3])

weights = {'node_0': np.array([1, 1]),
           'node_1': np.array([-1, 1]),
           'output': np.array([2, -1])}

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

