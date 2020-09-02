import torch
import torch.nn as nn
from model import Policynetwork
from utils import generate_episode, plot
import numpy as np
import gym

env = gym.make('CartPole-v0')

state_space = env.observation_space.shape[0]
action_space = env.action_space.n

alpha = 0.001
policy_network = Policynetwork(state_space, action_space)
optimizer = torch.optim.Adam(policy_network.parameters(), lr = alpha)

## TRAIN

episodes = 1000
rewards = []

for episode in range(episodes):
    
    episode_experience, reward_list = generate_episode(env, policy_network)          # RETURNS (S,A,R,S') TUPLES OF THE EPISODE AND LIST OF REWARDS OBTAINED AT EACH STEP
    total_reward = np.sum(reward_list)                  # TOTAL REWARD OF THE EPISODE
    rewards.append(total_reward)                        # STORING TOTAL REWARD OF EACH EPISODE
    loss = 0

    for i,sars in enumerate(episode_experience):
        
        state, action, _, nextstate = sars
        target_reward = np.sum(reward_list[i:-1]) - np.mean(reward_list)
        reward_weight = torch.tensor(target_reward)                             # using monte carlo estimate for target, have i incorporated causality here??
        action_logprob = -torch.log(policy_network(state))[action]
        loss += action_logprob*reward_weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (episode+1)%100 == 0:
      print(f'{episode+1}th episode; average reward of past 100 episodes :', np.mean(rewards[-100:]))              # PRINTING AVG OF LAST 100 EPISODES


plot(rewards)

# TESTING THE NETWORK

done = False
state = env.reset()
policy_network.eval()
while not done:
	env.render()
	action_probs = policy_network(state)
	action = int(np.random.choice(np.arange(action_space), p=action_probs.detach().numpy()))
	nextstate, reward, done, _ = env.step(action)
	if done:
		break
	state = nextstate

