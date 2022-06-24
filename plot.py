"""
Codes for plotting paths

Using:
matplotlib: 3.4.1
"""
import matplotlib.pyplot as plt
from functions import *


def draw_path_all(num_ships, ships_init, ships_goal, states, x_l, x_h, y_l, y_h, time):
    """
    Function for drawing planned path (all waypoints) from reinforcement learning algorithm
    :param num_ships: number of ships
    :param ships_init: initial positions of ships
    :param ships_goal: target positions of ships
    :param states: states of all ships(x, y, heading)
    :param x_l: value range of coordinate axis: minimum value of x axis
    :param x_h: value range of coordinate axis: maximum value of x axis
    :param y_l: value range of coordinate axis: minimum value of y axis
    :param y_h: value range of coordinate axis: maximum value of y axis
    :param time: last time
    :return: planned paths
    """
    # more ships need more colors, in which case please add more colors to this set
    colorset = ['b', 'r', 'g', 'y', 'c', 'm']

    fig, ax = plt.subplots()
    plt.xlabel('East')
    plt.ylabel('North')
    ax.set(xlim=(x_l, x_h),
           ylim=(y_l, y_h))
    for i in range(num_ships):
        plt.scatter(ships_init[i, 0], ships_init[i, 1], 20, marker='.', color=colorset[i], label='initial point')
        plt.scatter(ships_goal[i, 0], ships_goal[i, 1], 20, marker='x', color=colorset[i], label='goal point')
        # ax.step(states[:, 0, 3 * i], states[:, 0, 3 * i + 1], dashes=[2, 1], alpha=0.6,
        #         color=colorset[i], label='ship{}'.format(i + 1))
        ax.step(states[0:time-1, 0, 3 * i], states[0:time-1, 0, 3 * i + 1], color=colorset[i],
                label='ship{index}(S{ind})'.format(index=i + 1, ind=i + 1))
        ax.arrow(states[time-1, 0, 3 * i], states[time-1, 0, 3 * i + 1],
                 states[time-1, 0, 3 * i] - states[time-2, 0, 3 * i],
                 states[time-1, 0, 3 * i + 1] - states[time-2, 0, 3 * i + 1],
                 head_width=100, head_length=150, fc=colorset[i], ec=colorset[i])
    if env.ships_num == 1:
        pass
    else:
        title_template = 'Distance between ships\n at Time {t}s:'
        plt.text(x_max + 70, y_min + 500, title_template.format(t=time))
        idx = 0
        for i in range(num_ships):
            for j in range(i + 1, num_ships):
                dis = euc_dist(states[time-1, 0, 3 * i], states[time-1, 0, 3 * i + 1],
                               states[time-1, 0, 3 * j], states[time-1, 0, 3 * j + 1])
                dis = round(dis, 3)
                dis_template = 'S{ship1} and S{ship2}: {dis}m'
                plt.text(x_max + 70, y_min + 300 + idx, dis_template.format(dis=dis, ship1=i+1, ship2=j+1))
                idx -= 180
        ax.legend(title='Vessel list:', loc='upper left', bbox_to_anchor=(1, 1))
        # plt.scatter(obst_center[:, 0], obst_center[:, 1], 20, 'black', label='obstacle')
        plt.tight_layout()
    plt.show()


def draw_headings(num_ships, states):
    for i in range(num_ships):
        plt.figure()
        plt.xlabel('time(s)')
        plt.ylabel('Heading angle(deg)')
        plt.plot(states[:, 0, 3 * i + 2])
        plt.title('Heading angles of ship{index}'.format(index=i + 1))
    plt.show()


def draw_headings_speed(num_ships, states):
    speed = np.zeros((num_ships, len(states) - 1))
    time = np.linspace(0, len(states)-1, len(states)-1)
    for i in range(num_ships):
        for j in range(1, len(states)):
            speed[i, j-1] = states[j, 0, 3 * i + 2] - states[j-1, 0, 3 * i + 2]
    for i in range(num_ships):
        plt.figure()
        plt.xlabel('time(s)')
        plt.ylabel('Heading angular velocities(deg/s)')
        # plt.plot(speed[i,:])
        plt.step(time, speed[i,:])
        plt.title('Heading angular velocities of ship{index}'.format(index=i + 1))
    plt.show()


def draw_actions(num_ships, acts):
    time = np.linspace(0, len(acts), len(acts))
    for i in range(num_ships):
        plt.figure()
        plt.xlabel('time(s)')
        plt.ylabel('Heading angular velocities(deg/s)')
        plt.step(time, acts[:, i])
        plt.title('Heading angular velocities of ship{index}'.format(index=i + 1))
    plt.show()


def dra_score(score, weight):
    scalar = scores
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    plt.plot(smoothed, color='cornflowerblue', label='Smooth score curve')
    plt.plot(score, alpha=0.3, color='cornflowerblue', label='True score curve')
    plt.legend()
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    result_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedResult'
    ship_states = np.load(result_dir + '/path_global.npy')
    # ship_states = np.load(result_dir + '/path_test.npy')
    # rewards = np.load(result_dir + '/rewards_global.npy')
    scores = np.load(result_dir + '/score_history.npy')
    ship_actions = np.load(result_dir + '/speed_global.npy')

    # scenario = '1Ship'
    scenario = '2Ships_C2m'
    # scenario = '3Ships_C3H2'
    # scenario = '4Ships_C4H2O2'

    env = get_data(scenario)

    x = np.r_[env.ships_init[:, 0], env.ships_goal[:, 0]]
    y = np.r_[env.ships_init[:, 1], env.ships_goal[:, 1]]
    x_min = np.min(x) - 500
    x_max = np.max(x) + 500
    y_min = np.min(y) - 500
    y_max = np.max(y) + 500
    if (x_max - x_min) < (y_max - y_min):
        x_max_ = x_max
        x_min_ = x_min
        x_max += (y_max - y_min - x_max_ + x_min_) / 2
        x_min -= (y_max - y_min - x_max_ + x_min_) / 2
    else:
        y_max_ = y_max
        y_min_ = y_min
        y_max += (x_max - x_min - y_max_ + y_min_) / 2
        y_min -= (x_max - x_min - y_max_ + y_min_) / 2

    if env.ships_num >= 1:
        dis_min, time_min = np.load(result_dir + '/info_closest_most_global.npy')
        time_min = int(time_min) - 1
        # for i in range(len(ship_states)):
        #     for j in range(env.ships_num):
        #         for k in range(i + 1, env.ships_num):
        #             if j == k:
        #                 continue
        #             dis = euc_dist(ship_states[i, 0, 3 * j], ship_states[i, 0, 3 * j + 1],
        #                            ship_states[i, 0, 3 * k], ship_states[i, 0, 3 * k + 1])
        #             print(i, j, k)
        #             if dis_min == 0:
        #                 dis_min = dis
        #             elif dis_min > dis:
        #                 dis_min = dis
        #                 time_min = i

    t_begin = 5
    t_last = 0
    # t = t_begin
    # t = time_min
    t = t_last
    # draw_path_all(env.ships_num, env.ships_init, env.ships_goal, ship_states, x_min, x_max, y_min, y_max, time_min)
    # draw_headings(env.ships_num, ship_states)
    # draw_headings_speed(env.ships_num, ship_states)
    draw_actions(env.ships_num, ship_actions)
    # dra_score(scores, weight=0.99)
