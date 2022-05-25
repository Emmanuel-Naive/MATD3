"""
Codes for animation

Using:
matplotlib: 3.4.1
ffmpeg: 2.7.0
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functions import *

def animate1(i):
    time_text.set_text(time_template % (i * dt))

    for n in range(env.ships_num):
        if i > 0:
            headings[n] = ax.patches.remove(headings[n])
        past_trajectory[n].set_xdata(states[:i, 0, 3 * n])
        past_trajectory[n].set_ydata(states[:i, 0, 3 * n + 1])
        if i > 0:
            dx = states[i, 0, 3 * n] - states[i-1, 0, 3 * n]
            dy = states[i, 0, 3 * n + 1] - states[i-1, 0, 3 * n + 1]
            headings[n] = ax.arrow(states[i, 0, 3 * n], states[i, 0, 3 * n + 1], dx, dy,
                                   head_width=500, head_length=500, fc=colorset[n], ec=colorset[n])
    # return time_text, ship_markers, past_trajectory

def animate2(i):
    time_text.set_text(time_template % (i * dt))
    if i != 0:
        dis_info = dis_infos[i]
        dis = round(dis_info[0, 0],1)
        ship1 = int(dis_info[0, 1] + 1)
        ship2 = int(dis_info[0, 2] + 1)
        dis_text.set_text(dis_template.format(dis=dis, ship1=ship1, ship2=ship2))
    for n in range(env.ships_num):
        if i > 0:
            headings[n] = ax.patches.remove(headings[n])
        past_trajectory[n].set_xdata(states[:i, 0, 3 * n])
        past_trajectory[n].set_ydata(states[:i, 0, 3 * n + 1])
        if i > 0:
            dx = states[i, 0, 3 * n] - states[i-1, 0, 3 * n]
            dy = states[i, 0, 3 * n + 1] - states[i-1, 0, 3 * n + 1]
            headings[n] = ax.arrow(states[i, 0, 3 * n], states[i, 0, 3 * n + 1], dx, dy,
                                   head_width=300, head_length=500, fc=colorset[n], ec=colorset[n])
    # return time_text, ship_markers, past_trajectory


if __name__ == '__main__':
    result_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedResult'
    states = np.load(result_dir + '/path_global.npy')
    # states = np.load(result_dir + '/path_last.npy')
    # states = np.load(result_dir + '/path_test.npy')

    # scenario = '1Ship'
    scenario = '2Ships_Cross'
    # scenario = '2Ships_Headon'
    # scenario = '2Ships_Overtaking'
    # scenario = '3Ships_Cross&Headon'
    env = get_data(scenario)
    if env.ships_num == 1:
        pass
    else:
        dis_infos = np.load(result_dir + '/info_closest_all_global.npy')
        # dis_infos = np.load(result_dir + '/info_closest_local_last.npy')

    dt = 1
    t_step = len(states)

    x = env.ships_goal[:, 0]
    y = env.ships_goal[:, 1]

    for i in range(0, env.ships_num):
        x = np.r_[x, states[:, 0, 3 * i]]
        y = np.r_[y, states[:, 0, 3 * i + 1]]
    x_min = np.min(x) - 500
    x_max = np.max(x) + 500
    y_min = np.min(y) - 500
    y_max = np.max(y) + 500
    if (x_max - x_min) < (y_max - y_min):
        x_max_ = x_max
        x_min_ = x_min
        x_max += (y_max - y_min - x_max_ + x_min_)/2
        x_min -= (y_max - y_min - x_max_ + x_min_)/2
    else:
        y_max_ = y_max
        y_min_ = y_min
        y_max += (x_max - x_min - y_max_ + y_min_)/2
        y_min -= (x_max - x_min - y_max_ + y_min_)/2

    past_trajectory = []
    headings = []
    colorset = ['b', 'r', 'g', 'y', 'c', 'm']
    # colors = ['blue', 'purple', 'darkolivegreen', 'teal', 'darkorange', 'saddlebrown']

    fig = plt.figure()
    ax = fig.add_subplot(autoscale_on=False, xlim=(x_min, x_max), ylim=(y_min, y_max))
    ax.set_aspect('equal')

    for i in range(env.ships_num):
        past_trajectory.append(ax.plot([], [], c=colorset[i], dashes=[8, 4], alpha=0.8)[0])
        headings.append(ax.arrow([], [], [], []))

        plt.scatter(env.ships_goal[i, 0], env.ships_goal[i, 1], 20, marker='x', color=colorset[i],
                    label='ship{index}(S{ind})'.format(index=i+1, ind=i+1))
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    if env.ships_num == 1:
        pass
    else:
        dis_template = 'closest_distance = {dis}m \n (distance between S{ship1} and S{ship2})'
        dis_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)

    frequency = 1  # when this value is set lower than 1, the
    if env.ships_num == 1:
        ani_path = animation.FuncAnimation(fig, animate1, len(states), interval=frequency, blit=False)
    else:
        ani_path = animation.FuncAnimation(fig, animate2, len(states), interval=frequency, blit=False)
    plt.legend()
    plt.show()
    # ani.save("result.gif", writer='pillow')
    ani_path.save(result_dir + '/animation.mp4', writer='ffmpeg', fps=50)