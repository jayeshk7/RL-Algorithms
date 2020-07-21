# simplified blackjack state space, On policy MC.py has the full solution

import gym
import numpy as np 
import random as rd 
from utils import *

env = gym.make('Blackjack-v0')
env.unwrapped

state_space = env.observation_space
action_space = env.action_space

policy = {}
state_occur = {}
values = {}
for i in range(21): # because 21 is the number of states we can see - [1, 21] 
    
    # the probability for lower value hands is 0, the changes have not been made accordingly to make the code efficient
    
    values.update({i+1 : 0})                            # initialise value fn
    state_occur.update({i+1 : 0})                       # initialise occurence of state
    policy.update({i+1 : int(rd.uniform(0,2))})         # initialise random policy

for j in range(500000):
    
    episodes = generateEpisode(env)     # generates episode using policy same as dealer's
    observed_states = []

    for i in range(len(episodes)):
        if episodes[i][0] < 22:
            observed_states.append(episodes[i][0])
            state_occur[episodes[i][0]] += 1
    rewards = episodes[i][2]

    for state in observed_states:
        values[state] = values[state] + (rewards - values[state]) / state_occur[state]

    if j%500000==0:
        print('iteration number', j)