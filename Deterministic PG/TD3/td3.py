import torch
import torch.nn as nn 
import torch.nn.functional as f 

class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__

        self.linear1 = nn.Linear(state_space, 400)
        self.linear2 = nn.Linear(400, 200)
        self.linear3 = nn.Linear(200, 1)

    def forward(self, state) :

        state = torch.FloatTensor(state).cuda()
        hidden1 = f.relu(self.linear1(state))
        hidden2 = f.relu(self.linear2(hidden1))
        action = self.linear3(hidden2)

        return action 

class Critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__

        self.linear1 = nn.Linear(state_space+action_space, 400)
        self.linear2 = nn.Linear(400, 200)
        self.linear3 = nn.Linear(200, 1)

    def forward(self, state, action):
        
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        x = torch.cat([state, action]).cuda()
        hidden1 = f.relu(self.linear1(x))
        hidden2 = f.relu(self.linear2(hidden1))
        q_value = self.linear3(hidden2)

        return q_value