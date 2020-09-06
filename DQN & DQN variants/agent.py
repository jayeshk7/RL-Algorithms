import torch
import torch.nn as nn
import torch.nn.functional as F 
import gym
import numpy as np
from DQN import DQN, choose_action
from collections import deque
import random

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

replay_buffer = deque([])
episode_reward = []
episodes = 1000
epsilon = 0.9
MEMORY = 10000


for episode in range(episodes):

    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = choose_action(state, epsilon, env, target)
        nextstate, reward, done, _ = env.step(action)
        experience = (state, action, reward, nextstate)
        total_reward += reward
        
        # ADDING EXPERIENCE
        if len(replay_buffer) < MEMORY:
            replay_buffer.append(experience)
        else:
            replay_buffer.popleft()
            replay_buffer.append(experience)

        # TRAINING NETWORK
        if len(replay_buffer) >= 32:
            batch_buffer = random.sample(replay_buffer, 32)
            s, a, r, ns = map(np.stack, zip(*batch_buffer))             # SHAPE = 32x4

            target_qvalue, _ = torch.max(target(ns), 1)
            target_qvalue = torch.tensor(r) + target_qvalue
            prediction = torch.max(behaviour(s), 1)
            print(prediction, target_qvalue)
            loss = lossfn(prediction, target_qvalue)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        state = nextstate
    
    if (episode+1)%100 == 0:
        print(f'episode number {episode + 1}; average reward of last 100 episodes = {np.mean(episode_reward[-100:])}')
        target.load_state_dict(behaviour.state_dict())

            

     