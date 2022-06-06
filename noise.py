"""
Codes for creating noise

Using:
numpy: 1.21.5
random: Built-in package of Python
Python: 3.9
"""
import random
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt


class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        """
        Create Ornstein-Uhlenbeck noise
        :param action_dimension: number of actions
        :param mu: mean value of OU noise, default value is 0
        :param theta: a constant which should be bigger than 0: theta > 0
        :param sigma: standard deviation of OU noise, default value is 0.05
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state


class GaussNoise:
    def __init__(self, action_dimension, mu=0, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.sigma = sigma

    def noise(self):
        """
        Create Gauss white noise
        :param action_dimension: number of actions
        :param mu: mean value of OU noise, default value is 0
        :param sigma: standard deviation of OU noise, default value is 0.2
        """
        noises = []
        for i in range(self.action_dimension):
            noise = random.gauss(self.mu, self.sigma)
            noises.append(noise)
        return noises


if __name__ == '__main__':
    time = 500
    ou = OUNoise(1)
    ou_states = []
    for i in range(time):
        ou_states.append(ou.noise())

    ga = GaussNoise(1)
    ga_states = []
    for i in range(time):
        ga_states.append(ga.noise())

    n_l = 0.2
    ou_states_c = np.clip(ou_states, -n_l, n_l)
    ga_states_c = np.clip(ga_states, -n_l, n_l)
    # plt.plot(ou_states, label='OUNoise')
    # plt.plot(ga_states, label='GaussNoise')
    # plt.legend()
    # plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(ou_states, label='Ornstein-Uhlenbeck noise')
    plt.title('OUNoise')
    plt.subplot(2, 1, 1)
    plt.plot(ou_states_c, label='Clipped Ornstein-Uhlenbeck noise')
    plt.legend(bbox_to_anchor=(0, 0), loc=3, borderaxespad=0)
    plt.subplot(2, 1, 2)
    plt.plot(ga_states, label='Gauss white noise')
    plt.title('GaussNoise')
    plt.subplot(2, 1, 2)
    plt.plot(ga_states_c, label='Clipped Gauss white noise')
    plt.legend(bbox_to_anchor=(0, 0), loc=3, borderaxespad=0)
    plt.tight_layout()
    plt.show()
