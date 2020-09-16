import numpy as np
import gym
import random as rd
import time
from algorithms import PolicyImprovement, PolicyEvaluation, ValueIteration

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=200,
    reward_threshold=0.78, # optimum = .8196
)

direction = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

env = gym.make('FrozenLakeNotSlippery-v0')
env = env.unwrapped
env.render()

state_space = env.nS
action_space = env.nA

values = np.zeros(state_space)
policy = np.zeros(state_space)
theta = 1e-9
discount = 0.9
i = 0
iterations = 10

def PolicyIteration(env, policy, values, discount, state_space, action_space):
    
    values = PolicyEvaluation(env, policy, values, discount, state_space, action_space)
    policy = PolicyImprovement(env, policy, values, discount, state_space, action_space)

    return values, policy

while(1):

    # depending on which algorithm you want to run comment one of the two lines below
    values, policy = PolicyIteration(env, policy, values, discount, state_space, action_space)
    values = ValueIteration(env, values, discount, state_space, action_space)
    
    i += 1
    if(i>iterations):
        break
    
# comment this if you ran policy iteration.
# This finds the policy because in value iteration you don't change the policy, 
# you can find it using the value function returned from the algorithm
for state in range(state_space):

    temp = []
    for action in range(action_space):

        probability, nextstate, reward, _ = env.P[state][action][0]
        temp.append(reward + discount * probability * values[nextstate])
    policy[state] = np.argmax(temp)


print('value function\n', values,'\n')
print('policy\n',policy, '\n')
print('agent trained for', i, 'episodes')
nextstate = 0


while(1):
    time.sleep(0.5)
    action = policy[nextstate]
    nextstate, reward, done, info = env.step(int(action))
    env.render()
    if(done):
        break

# even if you don't uncomment anything, the code will still work