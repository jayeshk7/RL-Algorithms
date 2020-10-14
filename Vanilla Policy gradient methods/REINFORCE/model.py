import torch
import torch.nn as nn

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