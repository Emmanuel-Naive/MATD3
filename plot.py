"""
Codes for plotting paths

Using:
matplotlib: 3.4.1
"""
import matplotlib.pyplot as plt
from functions import *
from noise import *


def draw_path_all(num_ships, ships_init, ships_goal, states, x_l, x_h, y_l, y_h, time, obstacle_index, lw, NED):
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
    :param obstacle_index: obstacle ship index
    :param lw: width of trajectories
    :param NED: NED flag state
    :return: plots: planned paths at given times
    """
    # more ships need more colors, in which case please add more colors to this set
    colorbatch = ['b', 'r', 'g', 'y', 'c', 'm']

    fig, ax = plt.subplots()
    if NED:
        plt.xlabel('East')
        plt.ylabel('North')
    else:
        plt.xlabel('Position X')
        plt.ylabel('Position Y')
    # ax.set(xlim=(x_l, x_h))
    # ax.set(ylim=(y_l, y_h))
    # ax.set(ylim=(-1500, 1500))
    # y_min = -1500
    # ax.set(xlim=(-1000, x_h),
    #        ylim=(y_l, y_h))
    ax.set(xlim=(x_l, x_h),
           ylim=(y_l, y_h))
    for i in range(num_ships):
        plt.scatter(ships_init[i, 0], ships_init[i, 1], 20, marker='.', color=colorbatch[i], label='initial point')
        plt.scatter(ships_goal[i, 0], ships_goal[i, 1], 20, marker='x', color=colorbatch[i], label='goal point')
        # ax.step(states[:, 0, 3 * i], states[:, 0, 3 * i + 1], dashes=[2, 1], alpha=0.6,
        #         color=colorset[i], label='ship{}'.format(i + 1))
        if obstacle_index >= 0:
            if i == obstacle_index:
                ax.plot(states[0:time - 1, 0, 3 * i], states[0:time - 1, 0, 3 * i + 1], color=colorbatch[i],
                        linestyle='--', linewidth=lw, label='ship{index}(S{ind})'.format(index=i + 1, ind=i + 1))
            else:
                ax.plot(states[0:time - 1, 0, 3 * i], states[0:time - 1, 0, 3 * i + 1], color=colorbatch[i],
                        linewidth=lw, label='ship{index}(S{ind})'.format(index=i + 1, ind=i + 1))
        else:
            ax.plot(states[0:time - 1, 0, 3 * i], states[0:time - 1, 0, 3 * i + 1], color=colorbatch[i], linewidth=lw,
                    label='ship{index}(S{ind})'.format(index=i + 1, ind=i + 1))
        ax.arrow(states[time - 1, 0, 3 * i], states[time - 1, 0, 3 * i + 1],
                 states[time - 1, 0, 3 * i] - states[time - 2, 0, 3 * i],
                 states[time - 1, 0, 3 * i + 1] - states[time - 2, 0, 3 * i + 1],
                 head_width=100, head_length=150, fc=colorbatch[i], ec=colorbatch[i])
    if env.ships_num == 1:
        pass
    else:
        title_template = 'Distance between ships\n at Time {t}s:'
        plt.text(x_max + 70, y_min + 800, title_template.format(t=time))
        idx = 0
        for i in range(num_ships):
            for j in range(i + 1, num_ships):
                dis = euc_dist(states[time - 1, 0, 3 * i], states[time - 1, 0, 3 * j],
                               states[time - 1, 0, 3 * i + 1], states[time - 1, 0, 3 * j + 1])
                dis = round(dis, 3)
                dis_template = 'S{ship1} and S{ship2}: {dis}m'
                plt.text(x_max + 70, y_min + 600 + idx, dis_template.format(dis=dis, ship1=i + 1, ship2=j + 1))
                idx -= 180
        # plt.scatter(obst_center[:, 0], obst_center[:, 1], 20, 'black', label='obstacle')
    if env.ships_num == 1:
        plt.title('Trajectory at Time {t}s:'.format(t=time))
    else:
        plt.title('Trajectories')
    ax.legend(title='Vessel list:', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def draw_dis(num_ships, states):
    """
    Function for drawing distances between ships
    :param num_ships: number of ships
    :param states: states of all ships(x, y, heading)
    :return: plot: distances between ships
    """
    # more ships need more colors, in which case please add more colors to this set
    dis = np.zeros((1, len(states)))
    x = len(states) + 15
    y = 0
    for i in range(num_ships):
        for j in range(i + 1, num_ships):
            dis_init = euc_dist(states[0, 0, 3 * i], states[0, 0, 3 * j],
                                states[0, 0, 3 * i + 1], states[0, 0, 3 * j + 1])
            if dis_init > y:
                y = dis_init

    plt.figure(figsize=(8, 5))
    plt.xlabel('time(s)')
    plt.ylabel('Distance(m)')
    title_template = 'Shortest distance between ships:'
    plt.text(x, y, title_template)
    idx = -180
    for i in range(num_ships):
        for j in range(i + 1, num_ships):
            for k in range(len(states)):
                dis[0, k] = euc_dist(states[k, 0, 3 * i], states[k, 0, 3 * j],
                                     states[k, 0, 3 * i + 1], states[k, 0, 3 * j + 1])
                dis[0, k] = round(dis[0, k], 3)
            dis_min = np.min(dis)
            dis_min_index = np.where(dis[0, :] == dis_min)
            dis_min_index = dis_min_index[0]
            dis_min_index = dis_min_index[0]
            dis_template = 'S{ship1} and S{ship2}: {dis}m (Time: {time}s)'

            plt.plot(dis[0, :], label='Distance between S{ship1} and S{ship2}'.format(ship1=i + 1, ship2=j + 1))
            plt.text(x, y + idx, dis_template.format(dis=dis_min, ship1=i + 1, ship2=j + 1, time=dis_min_index + 1))
            idx -= 180
    plt.title('Distance between ships')
    plt.legend()
    plt.tight_layout()
    plt.show()


def draw_headings(num_ships, states, smooth=True, order=25):
    """
    Function for drawing headings, value range [0,360)
    :param num_ships: number of ships
    :param states: states of all ships(x, y, heading)
    :param smooth: smooth function
    :param order: order or degree of polynomial(the smooth function)
    :return: plot: (smooth) headings, value range [0,360)
    """
    time = np.linspace(0, len(states), len(states))
    for i in range(num_ships):
        plt.figure()
        plt.xlabel('time(s)')
        plt.ylabel('Heading angle(deg)')
        if smooth:
            fit_headings = np.poly1d(np.polyfit(time, states[:, 0, 3 * i + 2], order))
            plt.plot(time, fit_headings(time), color='cornflowerblue', label='Smooth Heading angle')
            plt.plot(states[:, 0, 3 * i + 2], alpha=0.3, color='cornflowerblue', label='True Heading angle')
        else:
            plt.plot(states[:, 0, 3 * i + 2], color='cornflowerblue', label='True Heading angle')
        # plt.scatter(time, states[:, 0, 3 * i + 2], s=1)
        # if i == 0:
        #     plt.ylim(85, 95)
        plt.title('Heading angles of ship{index}, value range [0,360)'.format(index=i + 1))
        plt.legend()
        plt.tight_layout()
    plt.show()


def draw_headings_180(num_ships, states, smooth=True, order=25):
    """
    Function for drawing headings, value range [-180, 180)
    :param num_ships: number of ships
    :param states: states of all ships(x, y, heading)
    :param smooth: smooth function
    :param order: order or degree of polynomial(the smooth function)
    :return: plot: (smooth) headings, value range [-180, 180)
    """
    time = np.linspace(0, len(states), len(states))
    for i in range(num_ships):
        states_m = warp_to_180(states[:, 0, 3 * i + 2], len(states))
        plt.figure()
        plt.xlabel('time(s)')
        plt.ylabel('Heading angle(deg)')
        if smooth:
            fit_headings = np.poly1d(np.polyfit(time, states_m, order))
            plt.plot(time, fit_headings(time), color='cornflowerblue', label='Smooth Heading angle')
            plt.plot(states_m, alpha=0.3, color='cornflowerblue', label='True Heading angle')
        else:
            plt.plot(states_m, color='cornflowerblue', label='True Heading angle')
        # plt.scatter(time, states[:, 0, 3 * i + 2], s=1)
        # if i == 0:
        #     plt.ylim(130, 140)
        plt.title('Heading angles of ship{index}, value range [-180,180)'.format(index=i + 1))
        plt.legend()
        plt.tight_layout()
    plt.show()


def draw_all_headings(num_ships, states, order=25):
    """
    Function for drawing all headings in one plot
    :param num_ships: number of ships
    :param states: states of all ships(x, y, heading)
    :param order: order or degree of polynomial(the smooth function)
    :return: general plot: all (smooth) headings, value range [0,360)
    """
    colorbatch = ['b', 'r', 'g', 'y', 'c', 'm']
    plt.figure()
    time = np.linspace(1, len(states) - 1, len(states) - 2)
    for i in range(num_ships):
        plt.xlabel('time(s)')
        plt.ylabel('Heading angle(deg)')
        fit_headings = np.poly1d(np.polyfit(time, states[1:len(states) - 1, 0, 3 * i + 2], order))
        plt.plot(time, fit_headings(time), color=colorbatch[i], label='ship{index})'.format(index=i + 1))
        plt.plot(states[:, 0, 3 * i + 2], alpha=0.3, color=colorbatch[i])
        # plt.scatter(time, states[:, 0, 3 * i + 2], s=1)
        # if i == 0:
        #     plt.ylim(130, 140)
        plt.title('Smoothed heading angles of ships')
        plt.legend()
        # plt.tight_layout()
    plt.show()


def draw_headings_speed(num_ships, states):
    """
    Function for drawing all heading speeds by calculating states
    :param num_ships: number of ships
    :param states: states of all ships(x, y, heading)
    :return: plot: heading speed (calculated based on states)
    """
    speed = np.zeros((num_ships, len(states) - 1))
    time = np.linspace(0, len(states) - 1, len(states) - 1)
    for i in range(num_ships):
        for j in range(1, len(states)):
            speed[i, j - 1] = states[j, 0, 3 * i + 2] - states[j - 1, 0, 3 * i + 2]
    for i in range(num_ships):
        plt.figure()
        plt.xlabel('time(s)')
        plt.ylabel('Heading angular velocities(deg/s)')
        # plt.plot(speed[i,:])
        plt.step(time, speed[i, :])
        plt.title('Heading angular velocities of ship{index}'.format(index=i + 1))
    plt.show()


def draw_actions(num_ships, acts):
    """
    Function for drawing all heading speeds by loading saved files
    :param num_ships: number of ships
    :param acts: actions of ships
    :return: plot: heading speed (draw based on the saved file)
    """
    time = np.linspace(0, len(acts), len(acts))
    for i in range(num_ships):
        plt.figure()
        plt.xlabel('time(s)')
        plt.ylabel('Heading angular velocities(deg/s)')
        plt.step(time, acts[:, i])
        plt.title('Heading angular velocities of ship{index}'.format(index=i + 1))
    plt.show()


def dra_score(score, weight):
    """
    Function for drawing all heading speeds by loading saved files
    :param score: the sum of cumulative rewards of all ships in one episode
    :param weight: weight of smooth function
    :return: plot: (smooth) score
    """
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
    # ship_states = np.load(result_dir + '/path_first.npy')
    ship_states = np.load(result_dir + '/path_global.npy')
    # ship_states = np.load(result_dir + '/path_last.npy')
    # rewards = np.load(result_dir + '/rewards_global.npy')
    scores = np.load(result_dir + '/score_history.npy')
    ship_actions = np.load(result_dir + '/speed_global.npy')

    # scenario = '1ShipM'
    # scenario = '2Ships_O2'
    # scenario = '3Ships_H3O2m'
    scenario = '4Ships_C4H4'

    env = get_data(scenario)

    NED = True
    # NED = False
    if NED:
        ship_states = XoYtoNED(ship_states)

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
        t_min = int(time_min)
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

    # t_begin = 10
    t_begin = 40
    t_last = len(ship_states)

    # t = t_begin
    # t = t_last
    # t = t_min
    w = 1
    # draw_dis(env.ships_num, ship_states)

    draw_path_all(env.ships_num, env.ships_init, env.ships_goal, ship_states, x_min, x_max, y_min, y_max, 30, -1, w, NED)
    draw_path_all(env.ships_num, env.ships_init, env.ships_goal, ship_states, x_min, x_max, y_min, y_max, 70, -1, w, NED)
    draw_path_all(env.ships_num, env.ships_init, env.ships_goal, ship_states, x_min, x_max, y_min, y_max, 102, -1, w, NED)
    draw_path_all(env.ships_num, env.ships_init, env.ships_goal, ship_states, x_min, x_max, y_min, y_max, 197, -1, w, NED)
    # draw_path_all(env.ships_num, env.ships_init, env.ships_goal, ship_states, x_min, x_max, y_min, y_max, t_begin, -1, w, NED)
    # draw_path_all(env.ships_num, env.ships_init, env.ships_goal, ship_states, x_min, x_max, y_min, y_max, t_min, -1, w, NED)
    # draw_path_all(env.ships_num, env.ships_init, env.ships_goal, ship_states, x_min, x_max, y_min, y_max, t_last, -1, w, NED)

    # smooth_flag = False
    smooth_flag = True
    smooth_order = 15
    draw_headings(env.ships_num, ship_states, smooth_flag, smooth_order)
    draw_headings_180(env.ships_num, ship_states, smooth_flag, smooth_order)
    # draw_all_headings(env.ships_num, ship_states, 10)
    # draw_headings_speed(env.ships_num, ship_states)
    # draw_actions(env.ships_num, ship_actions)
    # dra_score(scores, weight=0.95)
