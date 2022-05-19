"""
Function for building buffer, where some trained data would be saved.

Using:
numpy: 1.21.5
"""
import numpy as np


class MultiAgentReplayBuffer:
    """
    Replay Buffer for Multi agents
    """
    def __init__(self, max_size, actor_dims, critic_dims,
                 n_agents, n_actions, batch_size):
        """
        :param max_size: number for max size for storing transition
        :param critic_dims: number of dimensions for the critic
        :param actor_dims: number of dimensions for the actor
        :param n_actions: number of actions
        :param n_agents: number of agents
        :param batch_size: number of batch size
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        # These memories focus on critic
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        # These memories focus on reward
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)
        # These memories focus on actor
        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs, state, action, reward,
                         raw_obs_, state_, done):
        """
        :param raw_obs: state raw observations
        :param state:
        :param action:
        :param reward:
        :param raw_obs_: new state raw observations
        :param state_: new states
        :param done: terminal flags
        """
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        """
        :return:  appropriate memories
            actor_states: individual arrays of states
            states: flattened combination of state arrays
            actions: flattened combination of action arrays
            rewards: individual arrays of rewards
            actor_new_states: flattened combination of new action arrays
            states_: individual arrays of new states
            terminal: individual arrays of terminal flags
        """
        max_mem = min(self.mem_cntr, self.mem_size)  # current memory size
        # memories could not be selected multiple times (replace=False)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal

    def ready(self):
        """
        :return: memory state
            Ture:  fill up the batch size
        """
        if self.mem_cntr >= self.batch_size:
            return True
