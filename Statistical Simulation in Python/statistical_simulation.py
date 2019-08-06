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

# Define die outcomes and probabilities
die, probabilities, throws = [1,2,3,4,5,6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], 1

# Use np.random.choice to throw the die once and record the outcome
outcome = np.random.choice(die, size=throws, p=probabilities)
print("Outcome of the throw: {}".format(outcome[0]))

# =============================================================================
# Throwing two fair dice
# 
# We now know how to implement the first two steps of a simulation. 
# 
# Now let's implement the next step - defining the relationship between random variables.
# 
# Often times, our simulation will involve not just one, but multiple 
# random variables. Consider a game where throw you two dice and win if 
# each die shows the same number. Here we have two random variables - the 
# two dice - and a relationship between each of them - we win if they show 
# the same number, lose if they don't. In reality, the relationship between 
# random variables can be much more complex, especially when simulating things 
# like weather patterns.
# =============================================================================

# Initialize number of dice, simulate & record outcome
die, probabilities, num_dice = [1,2,3,4,5,6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], 2
outcomes = np.random.choice(die, size= num_dice, p=probabilities) 

# Win if the two dice show the same number
if outcomes[0]  == outcomes[1]: 
    answer = 'win' 
else:
    answer = 'lose'

print("The dice show {} and {}. You {}!".format(outcomes[0], outcomes[1], answer))

# =============================================================================
# Simulating the dice game
# 
# We now know how to implement the first three steps of a simulation. 
# 
# Now let's consider the next step - repeated random sampling.
# 
# Simulating an outcome once doesn't tell us much about how often we can expect 
# to see that outcome. In the case of the dice game from the previous exercise, 
# it's great that we won once. But suppose we want to see how many times we can 
# expect to win if we played this game multiple times, we need to repeat the 
# random sampling process many times. Repeating the process of random sampling is 
# helpful to understand and visualize inherent uncertainty and deciding next steps.
# =============================================================================


# Initialize model parameters & simulate dice throw
die, probabilities, num_dice = [1,2,3,4,5,6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], 2
sims, wins = 100, 0

for i in range(sims):
    outcomes = np.random.choice(a = die, size = num_dice, p = probabilities) 
    # Increment `wins` by 1 if the dice show same number
    if outcomes[0] == outcomes[1]: 
        wins += 1 

print("In {} games, you win {} times".format(sims, wins))

# =============================================================================
# Simulating one lottery drawing
# 
# In the last three exercises of this chapter, we will be bringing together 
# everything you've learned so far. We will run a complete simulation, take 
# a decision based on our observed outcomes, and learn to modify inputs to the 
# simulation model.
# 
# We will use simulations to figure out whether or not we want to buy a lottery 
# ticket. Suppose you have the opportunity to buy a lottery ticket which gives 
# you a shot at a grand prize of $1 Million. Since there are 1000 tickets in 
# total, your probability of winning is 1 in 1000. Each ticket costs $10. 
# 
# Let's use our understanding of basic simulations to first simulate one 
# drawing of the lottery
# =============================================================================

# Pre-defined constant variables
lottery_ticket_cost, num_tickets, grand_prize = 10, 1000, 1000000

# Probability of winning
chance_of_winning = 1/num_tickets

# Simulate a single drawing of the lottery
gains = [-lottery_ticket_cost, grand_prize-lottery_ticket_cost]
probability = [1-chance_of_winning, chance_of_winning]
outcome = np.random.choice(a=gains, size=1, p=probability, replace=True)

print("Outcome of one drawing of the lottery is {}".format(outcome))

# =============================================================================
# Should we buy?
# 
# In the last exercise, we simulated the random drawing of the lottery ticket once. 
# In this exercise, we complete the simulation process by repeating the process multiple times.
# 
# Repeating the process gives us multiple outcomes. We can think of this as
#  multiple universes where the same lottery drawing occurred. We can then 
#  determine the average winnings across all these universes. 
#  If the average winnings are greater than what we pay for the ticket 
#  then it makes sense to buy it, otherwise, we might not want to buy the ticket.
# 
# This is typically how simulations are used for evaluating business investments. 
# After completing this exercise, you will have the basic tools required to use 
# simulations for decision-making
# =============================================================================

# Initialize size and simulate outcome
lottery_ticket_cost, num_tickets, grand_prize = 10, 1000, 1000000
chance_of_winning = 1/num_tickets
size = 2000
payoffs = [-lottery_ticket_cost, grand_prize-lottery_ticket_cost]
probs = [1-chance_of_winning, chance_of_winning]

outcomes = np.random.choice(a=payoffs, size=size, p=probs, replace=True)

# Mean of outcomes.
answer = outcomes.mean()
print("Average payoff from {} simulations = {}".format(size, answer))

# =============================================================================
# Calculating a break-even lottery price
# 
# Simulations allow us to ask more nuanced questions that might not necessarily 
# have an easy analytical solution. Rather than solving a complex mathematical 
# formula, we directly get multiple sample outcomes. We can run experiments by 
# modifying inputs and studying how those changes impact the system. 
# 
# For example, once we have a moderately reasonable model of global weather 
# patterns, we could evaluate the impact of increased greenhouse gas emissions.
# 
# In the lottery example, we might want to know how expensive the ticket 
# needs to be for it to not make sense to buy it. To understand this, 
# we need to modify the ticket cost to see when the expected payoff is negative
# 
# =============================================================================

lottery_ticket_cost, num_tickets, grand_prize = 10, 1000, 1000000

# Initialize simulations and cost of ticket
sims, lottery_ticket_cost = 3000, 0

# Use a while loop to increment `lottery_ticket_cost` till average value of outcomes falls below zero
while 1:
    outcomes = np.random.choice([-lottery_ticket_cost, grand_prize-lottery_ticket_cost],
                 size=sims, p=[1-chance_of_winning, chance_of_winning], replace=True)
    if outcomes.mean() < 0:
        break
    else:
        lottery_ticket_cost += 1
        
answer = lottery_ticket_cost - 1

print("The highest price at which it makes sense to buy the ticket is {}".format(answer))


# CHAPTER 2

# =============================================================================
# Using Simulation for Probability Estimation
# 
# Steps for Estimating Probability:
#     
# 1. Construct sample space or population.
# 2. Determine how to simulate one outcome.
# 3. Determine rule for success.
# 4. Sample repeatedly and count successes.
# 5. Calculate frequency of successes as an estimate of probability.
# =============================================================================

# =============================================================================
# Two of a kind
# 
# Now let's use simulation to estimate probabilities. 
# 
# Suppose you've been invited to a game of poker at your friend's home. 
# In this variation of the game, you are dealt five cards and the player 
# with the better hand wins. You will use a simulation to estimate the 
# probabilities of getting certain hands. Let's work on estimating the 
# probability of getting at least two of a kind. Two of a kind is when 
# you get two cards of different suites but having the same numeric value 
# (e.g., 2 of hearts, 2 of spades, and 3 other cards).
# =============================================================================

# Shuffle deck & count card occurrences in the hand
n_sims, two_kind = 10000, 0

for i in range(n_sims):
    np.random.shuffle(deck_of_cards)
    hand, cards_in_hand = deck_of_cards[0:5], {}
    
    for card in hand:
        # Use .get() method on cards_in_hand
        cards_in_hand[card[1]] = cards_in_hand.get(card[1], 0) + 1
    
    # Condition for getting at least 2 of a kind
    highest_card = max(cards_in_hand.values())
    if  highest_card>=2: 
        two_kind += 1

print("Probability of seeing at least two of a kind = {} ".format(two_kind/n_sims))












