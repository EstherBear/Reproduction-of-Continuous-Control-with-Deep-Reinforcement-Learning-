import torch
import torch.nn.functional as F
import torch.nn as nn

class Actor(torch.nn.Module):

    def __init__(self, state_dimension, action_dimension, max_action):
        super(Actor, self).__init__()
        self.linear1 = torch.nn.Linear(state_dimension, 400)
        self.linear2 = torch.nn.Linear(400, 300)
        self.linear3 = torch.nn.Linear(300, action_dimension)
        self.bn1 = nn.BatchNorm1d(400)
        self.bn2 = nn.BatchNorm1d(300)
        self.max_action = max_action

    def forward(self, state, mode):
        if mode == 'bn' or mode == 'bntn':
            a = F.relu(self.bn1(self.linear1(state)))
            a = F.relu(self.bn2(self.linear2(a)))
        else:
            a = F.relu(self.linear1(state))
            a = F.relu(self.linear2(a))
        return self.max_action * torch.tanh(self.linear3(a))


class Critic(torch.nn.Module):

    def __init__(self, state_dimension, action_dimension):
        super(Critic, self).__init__()
        self.linear1 = torch.nn.Linear(state_dimension + action_dimension, 400)
        self.linear2 = torch.nn.Linear(400, 300)
        self.linear3 = torch.nn.Linear(300, 1)
        self.bn1 = nn.BatchNorm1d(400)
        self.bn2 = nn.BatchNorm1d(300)

    def forward(self, state, action, mode):
        if mode == 'bn' or mode == 'bntn':
            q = F.relu(self.bn1(self.linear1(torch.cat([state, action], 1))))
            q = F.relu(self.bn2(self.linear2(q)))
        else:
            q = F.relu(self.linear1(torch.cat([state, action], 1)))
            q = F.relu(self.linear2(q))
        return self.linear3(q)
