"""
Scenario: 2Ships_Cross
2 ships are in the cross rule

Definition for data in this scenario:
    ships_num: number of ships, int
    ships_init: initial positions of ships, array[x1,y1; x2, y2; ...]
    ships_goal: target positions of ships, array[x1,y1; x2, y2; ...]
    ships_speed(m/s): (constant) speeds of ships, array[v1; v2; ...]
    ships_head(degree): initial heading angles of ships, array[h1; h2; ...]
    (assume the spaces for all ships are same)
    ship_actions: action spaces of each ship, array[action1, action2, ...]
"""
from functions import *
ships_num = 2

ships_init = np.zeros((ships_num, 2))
ships_goal = np.zeros((ships_num, 2))
ships_speed = np.zeros((ships_num, 1))
ships_head = np.zeros((ships_num, 1))

ships_init[0, :] = np.array([5000, 0])
ships_goal[0, :] = np.array([5000, 10000])
ships_speed[0] = 20
ships_head[0] = 90

ships_init[1, :] = np.array([0, 5000])
ships_goal[1, :] = np.array([10000, 5000])
ships_speed[1] = 20
ships_head[1] = 0
# actions of ships
ship_action_space = 1 # heading angle
angle_limit = 2   # heading angle changing range (-2,2)


# calculate below data based on given data
ships_given_pos = np.vstack((ships_init.reshape(-1), ships_goal.reshape(-1)))
ships_pos_min = ships_given_pos.min(0)
ships_pos_max = ships_given_pos.max(0)
ships_x_min = []
ships_y_min = []
ships_x_max = []
ships_y_max = []
ships_dis = []
for ship_idx in range(ships_num):
    ships_x_min.append(ships_pos_min[ship_idx * 2])
    ships_y_min.append(ships_pos_min[ship_idx * 2 + 1])
    ships_x_max.append(ships_pos_max[ship_idx * 2])
    ships_y_max.append(ships_pos_max[ship_idx * 2 + 1])
    ships_dis.append(euc_dist(ships_init[ship_idx, 0], ships_goal[ship_idx, 0],
                              ships_init[ship_idx, 1], ships_goal[ship_idx, 1]))
ships_dis_max = np.array(ships_dis).max(-1)
ships_vel_min = ships_speed.min(0)

if __name__ == '__main__':
    # print(ships_init)
    # print(ships_head)
    # obs = np.column_stack((ships_init, ships_head))
    # # obs = np.concatenate((ships_init, ships_head.T), axis=0)
    # print(obs)

    print(ships_given_pos)
    print(ships_x_min)
    print(ships_y_min)
    print(ships_dis_max)
    print(ships_vel_min)
