"""
Code for creating a multi-agent environment with one of the scenarios listed in ./scenarios/.
"""
from functions import *


class MultiAgentEnv:
    """
    Environment for multi agents
    """

    def __init__(self, scenario_name):
        """
        :param scenario_name: the file name, str
            example: scenario_name = '2Ships_Cross'
        """
        self.case = get_data(scenario_name)

        self.ships_num = self.case.ships_num

        self.ships_pos = self.case.ships_init.copy()
        self.ships_speed = self.case.ships_speed.copy()
        self.ships_length = self.case.ships_length.copy()
        self.ships_head = warp_to_360(self.case.ships_head.copy(), self.ships_num)
        self.ships_done_term = [False] * self.ships_num

        self.ships_term = self.case.ships_goal.copy()
        self.ship_action_space = self.case.ship_action_space
        self.angle_limit = self.case.angle_limit
        # ships_obs_space = 2: position
        # ship__obs_space = 3: position + heading
        self.ship_obs_space = 3
        self.ships_obs = np.column_stack((self.ships_pos, self.ships_head))
        self.ships_obs_space = []
        for ship_idx in range(self.ships_num):
            self.ships_obs_space.append(self.ship_obs_space)
        self.ships_given_pos = self.case.ships_given_pos
        self.ships_pos_min = self.case.ships_pos_min
        self.ships_pos_max = self.case.ships_pos_max
        self.ships_x_min = self.case.ships_x_min
        self.ships_y_min = self.case.ships_y_min
        self.ships_x_max = self.case.ships_x_max
        self.ships_y_max = self.case.ships_y_max
        self.ships_dis_max = self.case.ships_dis_max
        self.ships_vel_min = self.case.ships_vel_min

    def reset(self):
        """
        Function for resetting
        :return: the initial position(state)
        """
        self.ships_pos = self.case.ships_init.copy()
        self.ships_speed = self.case.ships_speed.copy()
        self.ships_head = warp_to_360(self.case.ships_head.copy(), self.ships_num)
        self.ships_done_term = [False] * self.ships_num
        self.ships_obs = np.column_stack((self.ships_pos, self.ships_head))
        return self.ships_obs

    def step(self, actions):
        """
        Function for ships to move
        :param actions:
        :return: positions after moving
        """
        for ship_idx in range(self.ships_num):
            if not self.ships_done_term[ship_idx]:
                self.ships_head[ship_idx] += actions[ship_idx] * self.angle_limit
                self.ships_pos[ship_idx, 0] += (self.ships_speed[ship_idx] *
                                                math.cos(math.radians(self.ships_head[ship_idx])))
                self.ships_pos[ship_idx, 1] += (self.ships_speed[ship_idx] *
                                                math.sin(math.radians(self.ships_head[ship_idx])))
        self.ships_head = warp_to_360(self.ships_head, self.ships_num)
        self.ships_obs = np.column_stack((self.ships_pos, self.ships_head))
        return self.ships_obs


if __name__ == '__main__':
    ships = MultiAgentEnv('2Ships_Cross')
    # print(ships.ships_done_term)
    action = [5, 10]
    positions = ships.step(action)
    print(positions)
    # print(len(ships.ship_actions))
    # print(sum(ships.ships_obs_space))
