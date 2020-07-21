import numpy as np
import gym

env = gym.make('NChain-v0')
env.unwrapped

observation = env.reset()
print(observation)
while(1):
    action = env.action_space.sample()
    print('action taken',action)
    observation, reward, done, _ = env.step(action)
    print(observation, reward, done, '\n')
    if done:
        break

