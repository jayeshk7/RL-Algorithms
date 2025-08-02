import torch 
import torch.nn as nn 
import torch.nn.functional as f 

class DQN(nn.Module):
    def __init__(self, state_space, action_space):
        super(DQN, self).__init__()

        self.dqn1 = nn.Linear(state_space, 256)
        self.dqn2 = nn.Linear(256, 128)
        self.dqn3 = nn.Linear(128, action_space)

    def forward(self, state):

        state = torch.FloatTensor(state)
        hid1 = f.relu(self.dqn1(state))
        hid2 = f.relu(self.dqn2(hid1))
        q_values = self.dqn3(hid2)

        return q_values