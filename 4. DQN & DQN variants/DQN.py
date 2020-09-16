# import torch 
# import torch.nn as nn
# import numpy as np
# import torch.nn.functional as F 


# def choose_action(state, epsilon, env, target):
#     chance = np.random.uniform(0,1)
#     if chance > epsilon :
#         action = target.forward(state)
#         action = torch.argmax(action).detach()
#         return action.item()
#     else:
#         return env.action_space.sample()


# class DQN(nn.Module):
#     def __init__(self, state_space, action_space):
#         super(DQN, self).__init__()

#         self.linear1 = nn.Linear(state_space, 256)
#         self.linear2 = nn.Linear(256, 128)
#         self.out = nn.Linear(128, action_space)

#     def forward(self, state):

#         state = torch.FloatTensor(state).cuda()
#         hidden = F.relu(self.linear1(state))
#         hidden2 = F.relu(self.linear2(hidden))
#         output = self.out(hidden2)

#         return output


