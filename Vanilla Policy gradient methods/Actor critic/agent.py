import torch
import torch.nn as nn
import gym
from model import *

env = gym.make('CartPole-v0')
state_space = env.observation_space.shape[0]
action_space = env.action_space.n 

policy = Actor(state_space, action_space)
