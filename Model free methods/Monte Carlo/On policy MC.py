import gym
import numpy as np 
import random as rd 
from utils import *

env = gym.make('Blackjack-v0')
env.unwrapped

state_space = env.observation_space
action_space = env.action_space

# This code works only for the blackjack environment, make appropriate changes if environment is not Blackjack

policy = {}
state_occur = {}
values = {}
for i in range(21): # because 21 is the number of states we can see - [1, 21] 
    
    # the probability for lower value hands is 0, the changes has not been made accordingly to make the code efficient
    # it anyway wouldn't matter much it's just a few extra iterations
    
    values.update({i+1 : 0})
    state_occur.update({i+1 : 0})
    policy.update({i+1 : int(rd.uniform(0,2))})


## This is the code for monte carlo on policy evaluation, comment this part of the code if you are interested in MC control

for j in range(500000):
    
    episodes = generateEpisode(env)
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


## This is the code for monte carlo control

temp_qvlaues = np.zeros(2)
epsilon = 0.7

for i in range(1000):
    
    q_values = {}
    for state in range(21):
        for action in range(2):
            q_values.update({(state+1, action) : 0})
    
    for _ in range(10000): # this loop can be thrown away as we don't want to converge to the true value of that policy

        episodes = Policyimproving(env, policy)
        observed_states = []
        for j in range(len(episodes)):
        
            if episodes[j][0] < 22:
                observed_states.append((episodes[j][0],episodes[j][1]))  
                # states observed. Tuple (state, action)
                state_occur[episodes[j][0]] += 1
        rewards = episodes[j][2]

        for state_action in observed_states:
            
            state = state_action[0]     # state_action is tuple (current hand, action takens)
            q_values[state_action] = q_values[state_action] + (rewards - q_values[state_action]) / state_occur[state]
    
    for state in range(21):

        for action in range(2):
            temp_qvlaues[action] = q_values[(state+1, action)]
        
        chance = rd.uniform(0,1)
        if(chance > epsilon):
            policy[state] = np.argmax(temp_qvlaues)
        else:
            policy[state] = int(rd.uniform(0,2))

    if(i%100==0):
        epsilon /= 2
        print('iteration',i)


print('q_values\n', q_values)
