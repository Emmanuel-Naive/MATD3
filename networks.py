"""
Function for building networks

For each part(actor or critic), there are two network.
All four networks have similar structure.
Different parameter values, like learning rates(alpha and beta), could be implemented in networks.
For the critic part, the critic network would have the same parameter values with the critic target network.
The same situation happens in the actor part.

Using:
pytroch: 1.10.2
os: Built-in package of Python
Python: 3.9
"""
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.001, b=0.001)
        nn.init.constant_(layer.bias, 0)


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, chkpt_dir):
        """
        :param beta: learning rate of critic network
        :param input_dims: number of dimensions for inputs
        :param fc1_dims: number of dimensions for first layer
        :param fc2_dims: number of dimensions for second layer
        :param n_agents: number of agents
        :param n_actions: number of actions
        :param name: name of network
        :param chkpt_dir: check point directory
        """
        super(CriticNetwork, self).__init__()  # call the superclass(nn.Module) constructor

        self.chkpt_file = os.path.join(chkpt_dir, name)
        # network architecture of q1
        self.fc1 = nn.Linear(input_dims + n_agents * n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q1 = nn.Linear(fc2_dims, 1)

        # network architecture of q2
        self.fc3 = nn.Linear(input_dims + n_agents * n_actions, fc1_dims)
        self.fc4 = nn.Linear(fc1_dims, fc2_dims)
        self.q2 = nn.Linear(fc2_dims, 1)
        # optimization method
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        # if possible, use GPU to train
        self.apply(init_weights)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        """
        :param state:
        :param action:
        :return: result of the network
        """
        x1 = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x1 = F.relu(self.fc2(x1))
        q1 = self.q1(x1)

        x2 = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x2 = F.relu(self.fc2(x2))
        q2 = self.q2(x2)
        return q1, q2

    def Q1(self, state, action):
        """
        :param state:
        :param action:
        :return: result of the network
        """
        x1 = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x1 = F.relu(self.fc2(x1))
        q1 = self.q1(x1)
        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir):
        """
        :param alpha: learning rate of actor network
        :param input_dims: number of dimensions for inputs
        :param fc1_dims: number of dimensions for first layer
        :param fc2_dims: number of dimensions for second layer
        :param n_actions: number of actions
        :param name: name of network
        :param chkpt_dir: check point directory
        """
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # self.fc2 = nn.Linear(input_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.apply(init_weights)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        """
        :param state:
        :return: result of the network
        """
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        # x = F.relu(self.fc2(state))
        # output range (-1,1)
        # pi = nn.Tanh()(self.pi(x))
        pi = nn.Softsign()(self.pi(x))

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
