import torch
import torch.nn as nn
import gym
import numpy as np
from DQN import DQN, choose_action
from collections import deque
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

env = gym.make('CartPole-v0')
env.unwrapped
action_space = env.action_space.n 
state_space = env.observation_space.shape[0]

behaviour = DQN(state_space, action_space).cuda()
target = DQN(state_space, action_space).cuda()
target.eval()
target.load_state_dict(behaviour.state_dict())

alpha = 0.003
optimizer = torch.optim.Adam(behaviour.parameters(), lr = alpha)
lossfn = nn.MSELoss()

replay_buffer = deque([])
episode_reward = []
episodes = 2000
epsilon = 0.99
MEMORY = 10000
BATCH_SIZE = 64


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
        if len(replay_buffer) >= BATCH_SIZE:
            batch_buffer = random.sample(replay_buffer, BATCH_SIZE)
            s, a, r, ns = map(np.stack, zip(*batch_buffer))             # SHAPE = 32x4
            s = torch.FloatTensor(s)
            a = torch.FloatTensor(a).view(-1,1).cuda()
            r = torch.FloatTensor(r).view(-1,1).cuda()
            ns = torch.FloatTensor(ns)
            # print(s.shape, a.shape, r.shape, ns.shape)

            target_qvalue, _ = torch.max(target(ns), 1)
            target_qvalue = r + target_qvalue.view(-1,1)
            prediction, _ = torch.max(behaviour(s), 1) 
            prediction = prediction.view(-1,1)
            loss = lossfn(target_qvalue, prediction)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        state = nextstate
        episode_reward.append(total_reward)
    
    if (episode+1)%100 == 0:
        print(f'episode number {episode + 1}; average reward of last 100 episodes = {np.mean(episode_reward[-100:])}')
        target.load_state_dict(behaviour.state_dict())
        epsilon = epsilon/1.5

plt.plot(episode_reward)
plt.plot(gaussian_filter(episode_reward, 25))
plt.show()