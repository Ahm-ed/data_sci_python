#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 14:06:12 2019

@author: amin
"""
import numpy as np

# =============================================================================
# np.random.choice()
# In this exercise, you will be introduced to the np.random.choice() function. 
# This is a remarkably useful function for simulations and you will be making 
# extensive use of it later in the course
# =============================================================================

# =============================================================================
# Poisson random variable
# 
# The numpy.random module also has a number of useful probability distributions 
# for both discrete and continuous random variables. In this exercise, you will 
# learn how to draw samples from a probability distribution.
# 
# In particular, you will draw samples from a very important discrete probability 
# distribution, the Poisson distribution, which is typically used for modeling 
# the average rate at which events occur.
# 
# Following the exercise, you should be able to apply these steps to any of the 
# probability distributions found in numpy.random. In addition, you will also 
# see how the sample mean changes as we draw more samples from a distribution.
# =============================================================================

# Initialize seed and parameters
np.random.seed(123) 
lam, size_1, size_2 = 5, 3, 1000  

# Draw samples & calculate absolute difference between lambda and sample mean
samples_1 = np.random.poisson(lam, size_1)
samples_2 = np.random.poisson(lam, size_2)
answer_1 = abs(lam - samples_1.mean())
answer_2 = abs(lam - samples_2.mean()) 

print("|Lambda - sample mean| with {} samples is {} and with {} samples is {}. ".format(size_1, \
      answer_1, size_2, answer_2))

# Why do you think the larger size gives us a better result?

# =============================================================================
# Shuffling a deck of cards
# 
# Often times we are interested in randomizing the order of a set of items. 
# Consider a game of cards where you first shuffle the deck of cards or a game 
# of scrabble where the letters are first mixed in a bag. As the final exercise 
# of this section, you will learn another useful function - np.random.shuffle(). 
# This function allows you to randomly shuffle a sequence in place
# =============================================================================

deck_of_cards = [('Heart', 0),
 ('Heart', 1),
 ('Heart', 2),
 ('Heart', 3),
 ('Heart', 4),
 ('Heart', 5),
 ('Heart', 6),
 ('Heart', 7),
 ('Heart', 8),
 ('Heart', 9),
 ('Heart', 10),
 ('Heart', 11),
 ('Heart', 12),
 ('Club', 0),
 ('Club', 1),
 ('Club', 2),
 ('Club', 3),
 ('Club', 4),
 ('Club', 5),
 ('Club', 6),
 ('Club', 7),
 ('Club', 8),
 ('Club', 9),
 ('Club', 10),
 ('Club', 11),
 ('Club', 12),
 ('Spade', 0),
 ('Spade', 1),
 ('Spade', 2),
 ('Spade', 3),
 ('Spade', 4),
 ('Spade', 5),
 ('Spade', 6),
 ('Spade', 7),
 ('Spade', 8),
 ('Spade', 9),
 ('Spade', 10),
 ('Spade', 11),
 ('Spade', 12),
 ('Diamond', 0),
 ('Diamond', 1),
 ('Diamond', 2),
 ('Diamond', 3),
 ('Diamond', 4),
 ('Diamond', 5),
 ('Diamond', 6),
 ('Diamond', 7),
 ('Diamond', 8),
 ('Diamond', 9),
 ('Diamond', 10),
 ('Diamond', 11),
 ('Diamond', 12)]

# Shuffle the deck
np.random.shuffle(deck_of_cards) 

# Print out the top three cards
card_choices_after_shuffle = deck_of_cards[0:3]
print(card_choices_after_shuffle)



# =============================================================================
# Simulation steps
# 1. Define possible outcomes for random variables.
# 2. Assign probabilities.
# 3. Define relationships between random variables.
# 4. Get multiple outcomes by repeated random sampling.
# 5. Analyze sample outcomes
# =============================================================================

# =============================================================================
# Throwing a fair die
# 
# Once you grasp the basics of designing a simulation, you can apply it to any system or process. 
# Next, we will learn how each step is implemented using some basic examples.
# 
# As we have learned, simulation involves repeated random sampling. 
# The first step then is to get one random sample. Once we have that, 
# all we do is repeat the process multiple times. This exercise will focus on 
# understanding how we get one random sample. We will study this in the context of 
# throwing a fair six-sided die.
# =============================================================================














