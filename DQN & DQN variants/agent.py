import torch
import torch.nn as nn
import torch.nn.functional as F 
import gym
import numpy as np
from DQN import DQN, choose_action

env = gym.make('CartPole-v0')
env.unwrapped
action_space = env.action_space.n 
state_space = env.observation_space.shape[0]

behaviour = DQN(state_space, action_space)
target = DQN(state_space, action_space)
target.eval()
target.load_state_dict(behaviour.state_dict())

alpha = 0.003
optimizer = torch.optim.Adam(behaviour.parameters(), lr = alpha)
lossfn = nn.MSELoss()

state_replay = []
action_replay = []
nextstate_replay = []
reward_replay = []
episodes = 1000
epsilon = 0.9

state = env.reset()
for episode in range(episodes):

    action = choose_action(state, epsilon, env, target)
    nextstate, reward, done, _ = env.step(action)
    state_replay.append(state)
    reward_replay.append(reward)
    nextstate_replay.append(nextstate)
    action_replay.append(action)

    if len(states) < 64:
        continue
    batch = np.random.choice(len(states), 64, replace=False)
    batch_index = np.arange(32, dtype = np.int32)

     