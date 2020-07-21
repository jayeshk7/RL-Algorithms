import gym
import numpy as np 
import random as rd 
from utils import *

env = gym.make('Blackjack-v0')
env.unwrapped

action_space = [0,1]

policy = {}
state_occur = {}
values = {}
q_values = {}

for i in range(4,22): # our card sum can be from [4, 21] 
    
    for j in range(1,11):  # dealer's card value can be [1-10] where 1 is ace

        for ace in [True, False]:  # whether or not we have a usable ace

            values.update({(i, j, ace) : 0 })                           # initialise value fn
            state_occur.update({(i, j, ace) : 0})                       # initialise occurence of state
            policy.update({(i, j, ace) : int(rd.uniform(0,2))})         # initialise random policy
            
            for action in action_space:
                q_values.update({(i, j,  ace, action) : 0})             # initialise Q value lookup table



## This is the code for monte carlo control

temp_qvlaues = np.zeros(2)
epsilon = 0.7

for i in range(100000):
    
    # for _ in range(50):     # this loop can be thrown away if we don't want to converge to the true value of that policy

    episodes = Policyimproving(env, policy)     # episodes = (our hand, dealer hand, action, reward, usable ace)
    observed_states = []

    for j in range(len(episodes)):
    
        if episodes[j][0] < 22:
            observed_states.append((episodes[j][0], episodes[j][1], episodes[j][4], episodes[j][2]))      # states observed. Tuple (our hand, dealer hand, usable ace, action)
            state_occur[(episodes[j][0], episodes[j][1], episodes[j][4])] += 1                            # increasing count of current state

    rewards = episodes[j][3]

    for state_action in observed_states:
        
        state = (state_action[0], state_action[1], state_action[2])
        q_values[state_action] = q_values[state_action] + (rewards - q_values[state_action]) / state_occur[state]
    

    # choosing new policy
    for state in range(4,22):
        
        for dealer in range(1,11):

            for ace in [True, False]:

                for action in range(2):
                    temp_qvlaues[ace] = q_values[(state, dealer, ace, action)]
        
            chance = rd.uniform(0,1)
            if(chance > epsilon):
                policy[(state, dealer, ace)] = np.argmax(temp_qvlaues)
            else:
                policy[(state, dealer, ace)] = int(rd.uniform(0,2))


    if(i%10000==0):
        epsilon /= 2
        print('iteration',i)



print('q_values\n', q_values)
