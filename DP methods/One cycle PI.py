import numpy as np 
import random as rd 
import gym
import time

env = gym.make('FrozenLake8x8-v0') 
#this is a stochastic environment, action taken may be different from what you chose

env = env.unwrapped

state_space = env.nS
action_space = env.nA

policy = [int(rd.uniform(0,env.nA)) for i in range(state_space)]
values = np.zeros(env.nS)
discount = 0.9
delta = 0
theta = 1e-9
i = 0

while(True):

    prev_values = values.copy()
    
    for state in range(state_space):

        temp_value = 0
        nextstates = len(env.P[state][policy[state]]) 
        # multiple nextstates because environment throws you anywhere randomly
        # there is equal probability assigned to every action, doesn't really matter what action you take lmao
        
        for next in range(nextstates):
            
            # because there are multiple nextstates, so iterating over all possible states we could end up in

            probability, next_state, reward, _ = env.P[state][policy[state]][next] 
            temp_value += reward + discount*probability*values[next_state]
        values[state] = temp_value / nextstates # value of the current state = average over the values of states we can end up in

    for state in range(state_space):

        temp_value = 0
        temp_values = []
        
        for action in range(action_space):

            for next in range(len(env.P[state][action])): # same reason for iterating here
                
                probability, next_state, reward, _ = env.P[state][action][next]
                temp_value += reward + discount*probability*values[next_state]
    
            temp_value /= len(env.P[state][action])
            temp_values.append(temp_value)

        values[state] = np.max(temp_values)
        policy[state] = np.argmax(temp_values)

    i += 1
    delta = max(0, abs(np.sum(prev_values) - np.sum(values)))

    if(delta < theta):
        break

print('trained agent in', i, 'episdode(s)', '\n')
print('value function\n', values, '\n')
print('policy\n', policy, '\n')

nextstate = 0

while(1):
    time.sleep(0.5)
    action = policy[nextstate]
    nextstate, reward, done, info = env.step(int(action))
    env.render()
    if(done):
        break
