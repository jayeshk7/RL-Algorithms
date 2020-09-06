import torch
import torch.nn as nn 
import torch.nn.functional as f 

class ddpg_actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(ddpg_actor, self).__init__()

        self.actor1 = nn.Linear(state_space, 256)
        self.actor2 = nn.Linear(256, 128)
        self.actor3 = nn.Linear(128, action_space)

    def actor_forward(self, state):
        
        state = torch.FloatTensor(state).cuda()
        hidden1 = f.relu(self.actor1(state))
        hidden2 = f.relu(self.actor2(hidden1))
        out = self.actor3(hidden2)

        return out 

class ddpg_critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(ddpg_critic, self).__init__()
        
        self.critic1 = nn.Linear(state_space + action_space, 256)
        self.critic2 = nn.Linear(256, 128)
        self.critic3 = nn.Linear(128, 1)

    def critic_forward(self, state, action):          # ACTION IS ALREADY A TENSOR, STATE IS NOT A TENSOR
        
        state = torch.FloatTensor(state).cuda()
        action = action.cuda()
        state = torch.cat((state, action), 1).cuda()
        hidden1 = f.relu(self.critic1(state))
        hidden2 = f.relu(self.critic2(hidden1))
        out = self.critic3(hidden2)

        return out