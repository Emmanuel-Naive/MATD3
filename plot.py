"""
Codes for plotting paths and rewards

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
def draw_path(num_ships, ships_init, ships_goal, states):
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
    # ax.set(xlim=(x_min, x_max),
    #        ylim=(y_min, y_max))
    for i in range(num_ships):
        plt.scatter(ships_init[i, 0], ships_init[i, 1], 20, marker='s', color=colorset[i], label='initial point')
        plt.scatter(ships_goal[i, 0], ships_goal[i, 1], 20, marker='x', color=colorset[i], label='goal point')
        ax.step(states[:, 0, 3 * i], states[:, 0, 3 * i + 1], dashes=[8, 4],
                color=colorset[i], label='ship{}'.format(i + 1))
    ax.legend(title='Vessel list:')
    # plt.scatter(obst_center[:, 0], obst_center[:, 1], 20, 'black', label='obstacle')
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
    # states = np.load(result_dir + '/path_global.npy')
    # states = np.load(result_dir + '/path_test.npy')
    rewards = np.load(result_dir + '/rewards_global.npy')
    scores = np.load(result_dir + '/score_history.npy')

    scenario = '1Ship'
    # scenario = '2Ships_Cross'
    # scenario = '2Ships_Headon'
    # scenario = '3Ships_Cross&Headon'
    env = get_data(scenario)

    x = np.r_[env.ships_init[:, 0], env.ships_goal[:, 0]]
    y = np.r_[env.ships_init[:, 1], env.ships_goal[:, 1]]
    # x_min = np.min(x) - 20
    # x_max = np.max(x) + 20
    # y_min = np.min(y) - 20
    # y_max = np.max(y) + 20

    # draw_path(env.ships_num, env.ships_init, env.ships_goal, states)
    dra_score(scores, weight=0.999)
