import torch
import torch.nn as nn 
import numpy as np
import gym 
from ddpg import *
from collections import deque
import random
 
env = gym.make('Pendulum-v0')
env.unwrapped
action_space = env.action_space.shape[0]
state_space = env.observation_space.shape[0]

behaviour_critic = ddpg_critic(state_space, action_space).cuda()
target_critic = ddpg_critic(state_space, action_space).cuda()
target_critic.load_state_dict(behaviour_critic.state_dict())

behaviour_actor = ddpg_actor(state_space, action_space).cuda()
target_actor = ddpg_actor(state_space, action_space).cuda()
target_actor.load_state_dict(behaviour_actor.state_dict())


episodes = 4000
BATCH_SIZE = 32                   
MEMORY = 10000                    # REPLAY MEMORY CAPACITY
TAU = 0.001                       # FOR POLYAK AVERAGING
replay_buffer = deque([])         # INITIALISED REPLAY BUFFER
episode_reward = []

alpha = 0.003
lossfn = nn.MSELoss()
critic_optimizer = torch.optim.Adam(behaviour_critic.parameters(), lr = alpha)
actor_optimizer = torch.optim.Adam(behaviour_actor.parameters(), lr = alpha)


for episode in range(episodes):

    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        # SELECT ACTION
        action = behaviour_actor.actor_forward(state)
        action = action + torch.randn(1).cuda()              # ADDING NOISE FOR EXPLORATION
        nextstate, reward, done, _ = env.step([action.item()])
        total_reward += reward
        
        # STORE THE TRANSITION
        experience = (state, action.item(), reward, nextstate)
        if len(replay_buffer) < MEMORY :
            replay_buffer.append(experience)
        else:
            replay_buffer.popleft()
            replay_buffer.append(experience)

        # TRAINING NETWORK
        if len(replay_buffer) >= BATCH_SIZE :
            batch_buffer = random.sample(replay_buffer, BATCH_SIZE)
            s, a, r, ns = map(np.stack, zip(*batch_buffer))              # LEARNT SOMETHING NEW HERE

            # UPDATE THE CRITIC
            ns_actions = target_actor.actor_forward(ns)
            target_qvalues = torch.FloatTensor(r).view(-1,1).cuda() + target_critic.critic_forward(ns, ns_actions)
            predicted_qvalues = behaviour_critic.critic_forward(s, torch.unsqueeze(torch.FloatTensor(a),1))
            loss_critic = lossfn(predicted_qvalues, target_qvalues)

            critic_optimizer.zero_grad()
            loss_critic.backward()
            critic_optimizer.step()

            # UPDATE THE ACTOR 
            loss_actor = behaviour_critic.critic_forward(s, torch.unsqueeze(torch.FloatTensor(a),1))
            loss_actor = -loss_actor.mean()
            actor_optimizer.zero_grad()
            loss_actor.backward()
            actor_optimizer.step()

            # POLYAK AVERAGING
            for target_param, param in zip(target_critic.parameters(), behaviour_critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - TAU) + param.data * TAU
                )

            for target_param, param in zip(target_actor.parameters(), behaviour_actor.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - TAU) + param.data * TAU
                )

        state = nextstate


    episode_reward.append(total_reward)
    if (episode+1)%500 == 0:
        print(f'episode number {episode+1}; average reward of last 100 episodes = {np.mean(episode_reward[-100:])}')