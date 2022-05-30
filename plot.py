"""
Codes for plotting paths

Using:
matplotlib: 3.4.1
"""
import matplotlib.pyplot as plt
from functions import *


# def drawPlanedPath(num_ships, states, x_min, x_max, y_min, y_max, obst_center):
# :param x_min: value range of coordinate axis: minimum value of x axis
# :param x_max: value range of coordinate axis: maximum value of x axis
# :param y_min: value range of coordinate axis: minimum value of y axis
# :param y_max: value range of coordinate axis: maximum value of y axis
# :param obst_center: coordinates of obstacles
def draw_path(num_ships, ships_init, ships_goal, states, x_min, x_max, y_min, y_max):
    """
    Function for drawing planned path from reinforcement learning algorithm
    :param num_ships: number of ships
    :param ships_init: initial positions of ships
    :param ships_goal: target positions of ships
    :param states: states of all ships(x, y, heading)
    :return: dashed lines of planned paths
    """
    # more ships need more colors, in which case please add more colors to this set
    colorset = ['b', 'r', 'g', 'y', 'c', 'm']

    fig, ax = plt.subplots()
    plt.xlabel('East')
    plt.ylabel('North')
    ax.set(xlim=(x_min, x_max),
           ylim=(y_min, y_max))
    for i in range(num_ships):
        plt.scatter(ships_init[i, 0], ships_init[i, 1], 20, marker='s', color=colorset[i], label='initial point')
        plt.scatter(ships_goal[i, 0], ships_goal[i, 1], 20, marker='x', color=colorset[i], label='goal point')
        # ax.step(states[:, 0, 3 * i], states[:, 0, 3 * i + 1], dashes=[2, 1], alpha=0.6,
        #         color=colorset[i], label='ship{}'.format(i + 1))
        ax.step(states[:, 0, 3 * i], states[:, 0, 3 * i + 1], color=colorset[i],
                label='ship{index}(S{ind})'.format(index=i + 1, ind=i + 1))
        ax.arrow(states[-1, 0, 3 * i], states[-1, 0, 3 * i + 1],
                 states[-1, 0, 3 * i] - states[-2, 0, 3 * i], states[-1, 0, 3 * i + 1] - states[-2, 0, 3 * i + 1],
                 head_width=300, head_length=500, fc=colorset[i], ec=colorset[i])
    if env.ships_num == 1:
        pass
    else:
        dis_infos = np.load(result_dir + '/info_closest_local_last.npy')
        dis_info = dis_infos[i]
        dis = round(dis_info[0, 0], 1)
        ship1 = int(dis_info[0, 1] + 1)
        ship2 = int(dis_info[0, 2] + 1)
        dis_template = 'Time:{t}s \nClosest distance:{dis}m \n(distance between \n S{ship1} and S{ship2})'
        plt.text(x_max + 70, y_min + 25, dis_template.format(t=len(states), dis=dis, ship1=ship1, ship2=ship2))
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
    states = np.load(result_dir + '/path_global.npy')
    # states = np.load(result_dir + '/path_test.npy')
    # rewards = np.load(result_dir + '/rewards_global.npy')
    scores = np.load(result_dir + '/score_history.npy')

    # scenario = '1Ship'
    # scenario = '2Ships_Cross'
    # scenario = '2Ships_Headon'
    # scenario = '3Ships_Cross&Headon'
    scenario = '2Ships_H2'
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

    # draw_path(env.ships_num, env.ships_init, env.ships_goal, states, x_min, x_max, y_min, y_max)
    # draw_headings(env.ships_num, states)
    dra_score(scores, weight=0.99)
