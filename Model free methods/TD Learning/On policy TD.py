import numpy as np 
import gym
import time

env = gym.make('Taxi-v2')

# reward : +20 for successful dropoff and -1 for each timestep. -10 for illegal pickup and dropoff actions
# actions : {0,1,2,3,4,5} = {south, north, east, west, pickup, dropoff}

action_values = {}
policy = []

# Initialise all action values since using tabular methods 

for state in range(env.env.nS):

    policy.append(0)
    for action in range(env.env.nA):

        action_values[(state, action)] = 0

# TD learning of action values

iterations = 1000000
state = env.reset()
alpha = 0.1 
discount = 0.9
i = 0
epsilon = 0.7

while(1):

    # policy evaluation
    nextstate1, reward1, done1, _ = env.step(policy[state])
    nextstate2, reward2, done2, _ = env.step(policy[nextstate1])

    TD_error = reward1 + discount * action_values[(nextstate1, policy[nextstate1])] - action_values[(state, policy[state])]
    action_values[(state, action)] += alpha * TD_error  

    # policy improvement
    for state in range(env.env.nS):

        temp_actionvalue = []
        for action in range(env.env.nA):

            temp_actionvalue.append(action_values[(state, action)])
        
        # e-greedy
        if epsilon > np.random.uniform(0,1):
            policy[state] = np.argmax(temp_actionvalue)
        else:
            policy[state] = int(np.random.uniform(0,6))

    if i%1000 == 0:
        epsilon /= 2
        print('iteration number', i)

    state = nextstate1

    i += 1
    if i>iterations:
        break 

state = env.reset()
while(1):
    
    nextstate1, _, done, _ = env.step(policy[state])
    if done:
        break 

    env.render()
    time.sleep(0.5)
    state = nextstate1
