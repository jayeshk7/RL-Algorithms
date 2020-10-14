import torch
import torch.nn as nn
import torch.nn.functional as f 
import numpy as np
import gym 
from td3 import *

env = gym.make('Pendulum-v0')
env.unwrapped
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

critic1 = Critic(state_space, action_space).cuda()
target_critic1 = Critic(state_space, action_space)
critic2 = Critic(state_space, action_space)
target_critic2 = Critic(state_space, action_space)

actor = Actor(state_space, action_space)
target_actor = Actor(state_space, action_space)

