import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

env = gym.make('CartPole-v0')

state_space = env.observation_space.shape[0]
action_space = env.action_space.n

class Policynetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(Policynetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_space, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, action_space)
        
    def forward(self, state):
        
        state = torch.FloatTensor(state)
        hidden1 = nn.functional.relu(self.linear1(state))
        hidden2 = nn.functional.relu(self.linear2(hidden1))
        action_preds = nn.functional.softmax(self.linear3(hidden2), dim=0)
        
        return action_preds

def generate_episode(env, policy):

    experience = []
    rewards = []
    state = env.reset()
    action_probs = policy.forward(state)       
    action = int(np.random.choice(np.arange(action_space), p=action_probs.detach().numpy()))     # SELECTING ACTION ACCORDING TO THE PROBABILITY GIVEN BY NETWORK

    done = False
    while not done:
        nextstate, reward, done, _ = env.step(action)
        rewards.append(reward)                                   # APPENDING REWARD AT EACH STEP
        experience.append((state, action, reward, nextstate))    # APPENDING (S,A,R,S') TUPLE
        state = nextstate
        action_probs = policy.forward(state) 
        action = int(np.random.choice(np.arange(action_space), p=action_probs.detach().numpy()))     # SELECTING ACTION
    
    return experience, rewards


alpha = 0.001
policy_network = Policynetwork(state_space, action_space)
optimizer = torch.optim.Adam(policy_network.parameters(), lr = alpha)

## TRAIN

episodes = 800
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


plt.plot(rewards)
plt.xlabel(Episodes)
plt.ylabel(Rewards)
plt.plot(gaussian_filter(rewards, sigma=50))
plt.show()

done = False
state = env.reset()
policy_network.eval()
while not done:
	env.render()
	time.sleep(0.1)
	action_probs = policy_network(state)
	action = int(np.random.choice(np.arange(action_space), p=action_probs.detach().numpy()))
	nextstate, reward, done, _ = env.step(action)
	if done:
		break
	state = nextstate

