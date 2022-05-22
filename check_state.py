"""
Code for checking ship states, return reward.
"""
from functions import *
from make_env import MultiAgentEnv


class CheckState:
    def __init__(self, num_agent, pos_init, pos_term, head_init, vel_init, dis_goal, dis_collision,
                 dis_CPA1, dis_CPA2):
        """
        :param num_agent: number of agents
        :param pos_init: initial positions of ships
        :param pos_term: terminal positions of ships
        :param head_init: initial heading angles of ships
        :param vel_init: initial velocities of ships
        :param dis_goal: redundant distance for checking goal
        :param dis_collision: safe distance for checking collision
        :param dis_CPA1: redundant distance for checking CPA,
        :param dis_CPA2: redundant distance for checking CPA,
        """
        self.agents_num = num_agent
        self.pos_init = pos_init
        self.pos_term = pos_term
        self.heads = head_init
        self.speeds = vel_init
        self.dis_r = dis_goal
        self.dis_s = dis_collision
        self.dis_c1 = dis_CPA1
        self.dis_c2 = dis_CPA2

        if num_agent > 1:
            distance = []
            for ship_i in range(self.agents_num):
                for ship_j in range(ship_i + 1, self.agents_num):
                    distance.append(euc_dist(self.pos_init[ship_i, 0], self.pos_init[ship_j, 0],
                                             self.pos_init[ship_i, 1], self.pos_init[ship_j, 1]))
            self.dis_closest = min(distance)

            self.rules_list = ['Null'] * self.agents_num * self.agents_num
            self.rules_table = np.array(self.rules_list).reshape(self.agents_num, self.agents_num)
            for ship_i in range(self.agents_num):
                for ship_j in range(self.agents_num):
                    self.rules_table[ship_i, ship_j] = 'Null'

        self.max_reward_term = 10
        self.reward_alive = 1
        self.reward_max = self.max_reward_term + self.reward_alive
        if num_agent > 1:
            self.max_reward_CPA = 5
            self.max_reward_COLREGs = 20
            self.reward_max = self.reward_max + self.max_reward_CPA + self.max_reward_COLREGs

    def check_done(self, next_state, done_term):
        """
        Function for checking goal states
        :param next_state:
        :param done_term: flag of done state
        :return: reward_done: reward according to done states
        """
        reward_done = np.zeros(self.agents_num)
        for ship_idx in range(self.agents_num):
            dis_to_goal = euc_dist(next_state[ship_idx, 0], self.pos_term[ship_idx, 0],
                                   next_state[ship_idx, 1], self.pos_term[ship_idx, 1])

            ang_to_term = true_bearing(next_state[ship_idx, 0], next_state[ship_idx, 1],
                                       self.pos_term[ship_idx, 0], self.pos_term[ship_idx, 1])
            dif_ang = abs(next_state[ship_idx, 2] - ang_to_term)
            if dif_ang > 360:
                dif_ang -= 360

            if not done_term[ship_idx]:
                if dis_to_goal < self.dis_r:
                    done_term[ship_idx] = True
                    if dif_ang <= 2.5:
                        # reward for reaching goal
                        reward_done[ship_idx] = 100
                    elif 2.5 < dif_ang < 5:
                        reward_done[ship_idx] = 50
                    else:
                        reward_done[ship_idx] = 10
                else:
                    done_term[ship_idx] = False
                    # punishment for living
                    reward_done[ship_idx] = -self.reward_alive
        return reward_done, done_term

    # def check_term(self, state, next_state):
    #     """
    #     Function for checking relative distance to destination
    #     :param state:
    #     :param next_state:
    #     :return: reward_term and done_term states
    #     """
    #     reward_term = np.zeros(self.agents_num)
    #     for ship_idx in range(self.agents_num):
    #         dis_to_goal = euc_dist(next_state[ship_idx, 0], self.pos_term[ship_idx, 0],
    #                                next_state[ship_idx, 1], self.pos_term[ship_idx, 1])
    #         dis_last = euc_dist(state[ship_idx, 0], self.pos_term[ship_idx, 0],
    #                             state[ship_idx, 1], self.pos_term[ship_idx, 1])
    #         # reward_term[ship_idx] = dis_last - dis_to_goal
    #         if dis_last > dis_to_goal:
    #             reward_term[ship_idx] = dis_last - dis_to_goal
    #         else:
    #             reward_term[ship_idx] = dis_last - dis_to_goal - 5
    #     return reward_term

    def check_term(self, next_state):
        """
        Function for checking relative angle to destination
        :param next_state:
        :return: reward_term: reward according to heading angele
        """
        reward_term = np.zeros(self.agents_num)
        for ship_idx in range(self.agents_num):
            ang_to_term = true_bearing(next_state[ship_idx, 0], next_state[ship_idx, 1],
                                       self.pos_term[ship_idx, 0], self.pos_term[ship_idx, 1])
            dif_ang = abs(next_state[ship_idx, 2] - ang_to_term)
            if dif_ang > 360:
                dif_ang -= 360

            if dif_ang < 5:
                reward_term[ship_idx] = self.max_reward_term
            elif 5 <= dif_ang < 90:
                reward_term[ship_idx] = 0
            else:
                reward_term[ship_idx] = -self.max_reward_term
        return reward_term

    # def check_coll(self, state, next_state):
    #     """
    #     Function for checking collision
    #     :param state:
    #     :param next_state:
    #     :return: reward_coll and done_coll states
    #              reward_coll: reward according to collision states
    #     """
    #     reward_coll = np.zeros(self.agents_num)
    #     done_coll = False
    #     for ship_i in range(self.agents_num):
    #         for ship_j in range(ship_i+1, self.agents_num):
    #             dis_coll = euc_dist(next_state[ship_i, 0], next_state[ship_j, 0],
    #                                 next_state[ship_i, 1], next_state[ship_j, 1])
    #             dis_last = euc_dist(state[ship_i, 0], state[ship_j, 0],
    #                                 state[ship_i, 1], state[ship_j, 1])
    #             if dis_coll < self.dis_closest:
    #                 self.dis_closest = dis_coll
    #
    #             if dis_coll < self.dis_s:
    #                 done_coll = True
    #                 reward_coll[ship_i] = reward_coll[ship_i] + (dis_coll - dis_last)/20 - 100
    #                 reward_coll[ship_j] = reward_coll[ship_j] + (dis_coll - dis_last)/20 - 100
    #             else:
    #                 reward_coll[ship_i] = reward_coll[ship_i] + (dis_coll - dis_last)/20
    #                 reward_coll[ship_j] = reward_coll[ship_j] + (dis_coll - dis_last)/20
    #     return reward_coll, done_coll

    def check_coll(self, next_state):
        """
        Function for checking collision
        :param next_state:
        :return: reward_coll and done_coll states
                 reward_coll: reward according to collision states
        """
        reward_coll = np.zeros(self.agents_num)
        done_coll = False
        for ship_i in range(self.agents_num):
            for ship_j in range(ship_i + 1, self.agents_num):
                dis_coll = euc_dist(next_state[ship_i, 0], next_state[ship_j, 0],
                                    next_state[ship_i, 1], next_state[ship_j, 1])
                if dis_coll < self.dis_s:
                    done_coll = True
                    reward_coll[ship_i] = reward_coll[ship_i] - 100
                    reward_coll[ship_j] = reward_coll[ship_j] - 100
        return reward_coll, done_coll, dis_coll

    def check_CPA(self, next_state):
        reward_CPA = np.zeros(self.agents_num)
        for ship_i in range(self.agents_num):
            for ship_j in range(ship_i + 1, self.agents_num):
                DCPA, TCPA = CPA(next_state[ship_i, 0], next_state[ship_i, 1], next_state[ship_i, 2],
                                 self.speeds[ship_i],
                                 next_state[ship_j, 0], next_state[ship_j, 1], next_state[ship_j, 2],
                                 self.speeds[ship_j])
                if DCPA >= self.dis_c1:
                    reward_CPA[ship_i] = reward_CPA[ship_i] + self.max_reward_CPA
                elif self.dis_c2 < DCPA < self.dis_c1:
                    reward_CPA[ship_i] += 0
                else:
                    reward_CPA[ship_i] = reward_CPA[ship_i] - self.max_reward_CPA
        return reward_CPA

    def check_CORLEGs(self, state, next_state):
        """
        :param state:
        :param next_state:
        :return: reward_CORLEGs: reward according to CORLEGs states
        """
        pos = state[:, 0:2]
        head = state[:, 2]
        # pos_ = next_state[:, 0:1]
        head_ = next_state[:, 2]
        head_diff = warp_to_180(head_ - head, self.agents_num)

        reward_CORLEGs = np.zeros(self.agents_num)
        for ship_i in range(self.agents_num):
            for ship_j in range(self.agents_num):
                # update the CORLEGs table
                if ship_i == ship_j:
                    self.rules_table[ship_i, ship_j] = 'Null'
                else:
                    self.rules_table[ship_i, ship_j] = colregs_rule(
                        pos[ship_i, 0], pos[ship_i, 1],
                        head[ship_i], self.speeds[ship_i],
                        pos[ship_j, 0], pos[ship_j, 1],
                        head[ship_j], self.speeds[ship_j])
                    # get reward according heading angles
                    if self.rules_table[ship_i, ship_j] == 'HO-GW' or 'OT-GW' or 'CR-GW':
                        # Ship steered starboard (negative head_diff)
                        reward_CORLEGs[ship_i] -= head_diff[ship_i] * self.max_reward_COLREGs
                    if self.rules_table[ship_i, ship_j] == 'OT-SO' or 'CR-SO':
                        # stand-on: The smaller heading angles change, the better rewards
                        if abs(head_diff[ship_i]) < 0.1:
                            reward_CORLEGs[ship_i] += self.max_reward_COLREGs
                        else:
                            reward_CORLEGs[ship_i] -= abs(head_diff[ship_i]) * self.max_reward_COLREGs
        return reward_CORLEGs, self.rules_table


if __name__ == '__main__':
    ships = MultiAgentEnv('2Ships_Cross')
    # obs = ships.ships_pos
    # actions = [5, 10]
    # obs_ = ships.step(actions)
    dis_r = 10
    dis_s = 15
    # check_env = CheckState(ships.ships_num, ships.ships_pos, ships.ships_term, ships.ships_head, ships.ships_speed,
    #                        dis_r, dis_s)

    # done_term = [False] * ships.ships_num
    # reward_term, done_term = check_env.check_term(obs, obs_, done_term)
    # reward_coll, done_coll = check_env.check_coll(obs, obs_)
    # print(check_env.rules_table)

    # ships = MultiAgentEnv('1Ship')
    obs = ships.ships_pos
    actions = [10, 0]
    obs_ = ships.step(actions)
    # dis_r = 10
    # dis_s = 15
    done = [False] * ships.ships_num
    check_env = CheckState(ships.ships_num, ships.ships_pos, ships.ships_term, ships.ships_head, ships.ships_speed,
                           dis_r, dis_s)
    print(obs_)
    reward_term = check_env.check_term(obs_)
    reward_done, done = check_env.check_done(obs_, done)
    print(reward_term)
    print(reward_done)
