import numpy as np
import gym
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

env = gym.make('Taxi-v2')
env.unwrapped
# reward : +20 for successful dropoff and -1 for each timestep. -10 for illegal pickup and dropoff actions
# actions : {0,1,2,3,4,5} = {south, north, east, west, pickup, dropoff}


# Observation space = Discrete(500)
# Action space = Discrete(6)
action_space = [0,1,2,3,4,5]

q_values = {}
policy = {}

for i in range(env.observation_space.n):                        
    policy.update({i : np.random.choice(action_space)})     # initialise policy
    for j in range(env.action_space.n):     
        q_values.update({(i,j) : 0})                        # initialise q_value table

epsilon = 0.9
alpha = 0.1

rewards = []
timestep = 0
total_reward = 0
state = env.reset()             # returns the state of the environment 

# since TD is online learning algo, no need to generate episode like in MC control
for iteration in range(100000):

    # policy evaluation
    nextstate, reward, done, _ = env.step(policy[state])
    timestep += 1
    total_reward += reward
    q_values[(state, policy[state])] += alpha * (reward + q_values[(nextstate, policy[nextstate])] - q_values[(state, policy[state])])
    state = nextstate

    # (epsilon greedy) policy improvement
    for i in range(env.observation_space.n):
        temp_policy = []
        for j in action_space:
            temp_policy.append(q_values[(i,j)])

        chance = np.random.uniform(0,1)
        if chance < epsilon:
            policy[i] = np.random.choice(action_space)
        else:
            policy[i] = np.argmax(temp_policy)

    if (iteration+1)%10000 == 0:
        print(f'iteration number {iteration+1} : avg reward per timestep = {rewards[-1:]}')
        epsilon /= 1.5

    if done:
        state = env.reset()
        rewards.append(total_reward / timestep)
        total_reward = 0
        timestep = 0


# plot average rewards
plt.title('Rewards per timestep')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.plot(rewards)
plt.plot(gaussian_filter(rewards, sigma = 10))
plt.show()


# try out the policy
state = env.reset()
while(1):
    env.render()
    time.sleep(0.5)
    action = policy[state]
    state, reward, done, _ = env.step(action)
    if done : 
        break