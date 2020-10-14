import torch
import torch.nn as nn
import torch.nn.functional as f 

class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__

        self.actor1 = nn.Linear(state_space, 256)
        self.actor2 = nn.Linear(256, 128)
        self.actor3 = nn.Linear(128, action_space)

    def forward(self, state):

        state = torch.FloatTensor(state).cuda()
        hid1 = f.relu(self.actor1(state))
        hid2 = f.relu(self.actor2(hid1))
        action_probs = f.softmax(self.actor3(hid2))

        return action_probs

class Critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__

        self.critic1 = nn.Linear(state_space+action_space, 256)
        self.critic2 = nn.Linear(256, 128)
        self.critic3 = nn.Linear(128, 1)

    def forward(self, state):

        state = torch.FloatTensor(state).cuda
        hid1 = f.relu(self.critic1(state))
        hid2 = f.relu(self.critic2(hid1))
        state_value = self.critic3(hid2)

        return state_value