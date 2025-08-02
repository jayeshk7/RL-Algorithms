import torch
import torch.nn as nn
import gym
from collections import deque
from DQN import *
import numpy as np
import random as rd

env = gym.make('CartPole-v0')
state_space = env.observation_space.shape[0]
action_space = env.action_space.n 

behaviour_q = DQN(state_space, action_space)
target_q = DQN(state_space, action_space)
target_q.load_state_dict(behaviour_q.state_dict())
target_q.eval()

optimizer = torch.optim.Adam(behaviour_q.parameters(), lr = 0.0003)
lossfn = nn.MSELoss()

episodes = 1000
replay_buffer = deque([])
MAXBUFFER = 100000
BATCHSIZE = 128
epsilon = 0.9
total_rewards = []

for episode in range(episodes):

    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = behaviour_q(state).detach()
        if np.random.uniform(0,1) > epsilon:
            action = torch.argmax(action).item()
        else :
            action = env.action_space.sample()
        nextstate, reward, done, _ = env.step(action)
        episode_reward += reward

        experience = (state, action, reward, nextstate)
        if len(replay_buffer) < MAXBUFFER :
            replay_buffer.append(experience)
        else:
            replay_buffer.popleft()
            replay_buffer.append(experience)

        if len(replay_buffer) >= BATCHSIZE:
            batch_buffer = rd.sample(replay_buffer, BATCHSIZE)
            s, a, r, ns = map(np.stack, zip(*batch_buffer))
            a = torch.FloatTensor(a)
            r = torch.FloatTensor(r)

            target_value,_ = torch.max(target_q(ns), 0)
            target_value = r + target_value
            predicted_value = behaviour_q(s)
            loss = lossfn(predicted_value, target_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    total_rewards.append(episode_reward)
    if (episode+1)%100 == 0:
        print(f'avg reward of past 50 episodes : {np.mean(total_rewards[-50:])}')


