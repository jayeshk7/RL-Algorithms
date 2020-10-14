import torch
import torch.nn as nn 
import torch.nn.functional as f 

class ddpg_actor(nn.Module):                           # TAKES STATE AS INPUT AND OUTPUTS ACTION
    def __init__(self, state_space, action_space):
        super(ddpg_actor, self).__init__()

        self.actor1 = nn.Linear(state_space, 400)
        self.actor2 = nn.Linear(400, 200)
        self.actor3 = nn.Linear(200, action_space)

    def actor_forward(self, state):
        
        state = torch.FloatTensor(state).cuda()
        hidden1 = f.relu(self.actor1(state))
        hidden2 = f.relu(self.actor2(hidden1))
        out = self.actor3(hidden2)

        return out 

class ddpg_critic(nn.Module):                         # TAKES STATE AND ACTION AS INPUT, OUTPUTS Q VALUE
    def __init__(self, state_space, action_space):
        super(ddpg_critic, self).__init__()
        
        self.critic1 = nn.Linear(state_space + action_space, 400)
        self.critic2 = nn.Linear(400, 200)
        self.critic3 = nn.Linear(200, 1)

    def critic_forward(self, state, action):          # ACTION IS ALREADY A TENSOR, STATE IS NOT A TENSOR
        
        state = torch.FloatTensor(state).cuda()
        # action = action.cuda()
        state = torch.cat((state, action), 1).cuda()
        hidden1 = f.relu(self.critic1(state))
        hidden2 = f.relu(self.critic2(hidden1))
        out = self.critic3(hidden2)

        return out