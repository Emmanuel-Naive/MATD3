"""
Code for some utility functions.

Using:
numpy: 1.21.5
os: Built-in package of Python
math: Built-in package of Python
Python: 3.9
"""
import os
import math
import numpy as np
import Scenarios as Scen


def get_data(scenario_name):
    """
    Get data from the saved file (scenario)
    :param scenario_name:
    :return: data in this given scenario, which includes:
        ships_num: number of ships
        ships_init: initial positions of ships
        ships_term: target positions of ships
        ships_speed: (constant) speeds of ships
        ships_head: initial heading angles of ships
        ship_actions: action spaces of each ship (assume the spaces for all ships are same)
    """
    scenario = Scen.load(scenario_name + ".py")
    return scenario

def wrap_to_pi(angle):
    """
    wraps the angle to [-pi,pi)
    :param angle: (radians)
    :return:
    """
    res = math.fmod(angle + 2 * math.pi, 2 * math.pi)
    if res >= math.pi:
        res -= 2*math.pi
    return res


def wrap_to_2pi(angle):
    """
    wraps the angle to [0,2*pi)
    :param angle: (radians)
    :return:
    """
    res = math.fmod(angle + 2 * math.pi, 2 * math.pi)
    return res


def warp_to_180(degrees, n):
    """
    wraps angles to [-180,180)
    :param degrees:
    :param n: number of angles(degrees)
    :return:
    """
    res = np.zeros(n)
    for i in range(n):
        res[i] = math.radians(degrees[i])
        res[i] = wrap_to_pi(res[i])
        res[i] = round(math.degrees(res[i]), 2)
        if res[i] >= 180:
            res[i] -= 360
        if res[i] < -180:
            res[i] += 360
    return res


def warp_to_360(degrees, n):
    """
    wraps the angle to [0,360)
    :param degrees: number of angles(degrees)
    :param n:
    :return:
    """
    res = np.zeros(n)
    for i in range(n):
        res[i] = math.radians(degrees[i])
        res[i] = wrap_to_2pi(res[i])
        res[i] = round(math.degrees(res[i]), 2)
        if res[i] >= 360:
            res[i] -= 360
        if res[i] < 0:
            res[i] += 360
    return res


def euc_dist(x_1, x_2, y_1, y_2):
    """
    calculate the Euclidean distance
    :param x_1:
    :param y_1:
    :param x_2:
    :param y_2:
    :return:
    """
    distance = math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
    return distance


def true_bearing(x_os, y_os, x_cn, y_cn):
    """
    calculate the true bearing
    :param x_os:
    :param y_os:
    :param x_cn:
    :param y_cn:
    :return:
    """
    # result in radians between -pi and pi
    bearing_ture = math.atan2((y_cn - y_os), (x_cn - x_os))
    # result in radians between 0 and 2*pi
    # true_bearing = wrap_to_2pi(true_bearing)
    # result in degrees between 0 and 360
    bearing_ture = round(math.degrees(bearing_ture), 2)
    while bearing_ture >= 180:
        bearing_ture -= 360
    while bearing_ture < -180:
        bearing_ture += 360
    return bearing_ture


def relative_bearing(x_os, y_os, theta_os, x_cn, y_cn):
    """
    calculate the relative bearing
    :param x_os:
    :param y_os:
    :param theta_os:
    :param x_cn:
    :param y_cn:
    :return:
    """
    rel_bearing = true_bearing(x_os, y_os, x_cn, y_cn) - theta_os
    # Relative bearing is between -pi, pi
    # rel_bearing = wrap_to_pi(math.radians(rel_bearing))
    # result in degrees between 0 and 360
    # rel_bearing = round(math.degrees(rel_bearing), 2)
    while rel_bearing >= 180:
        rel_bearing -= 360
    while rel_bearing < -180:
        rel_bearing += 360
    return rel_bearing


def colregs_rule(ship1_x, ship1_y, ship1_psi, ship1_u, ship2_x, ship2_y, ship2_psi, ship2_u):
    """
    check COLREGs
    :param ship1_x:
    :param ship1_y:
    :param ship1_psi:
    :param ship1_u:
    :param ship2_x:
    :param ship2_y:
    :param ship2_psi:
    :param ship2_u:
    :return: state according COLREGs
    """
    # RB_os_ts: Relative bearing of TS from OS
    RB_os_ts = relative_bearing(ship1_x, ship1_y, ship1_psi, ship2_x, ship2_y)
    # RB_ts_os: Relative bearing of OS from TS
    RB_ts_os = relative_bearing(ship2_x, ship2_y, ship2_psi, ship1_x, ship1_y)
    # Head on
    if abs(RB_os_ts) < 13 and abs(RB_ts_os) < 13:
        rule = 'HO-GW'
    # Overtaking, stand on
    elif abs(RB_os_ts) > 112.5 and abs(RB_ts_os) < 45 and (ship2_u > (ship1_u * 1.1)):
        rule = 'OT-SO'
    # Overtaking, give way
    elif abs(RB_ts_os) > 112.5 and abs(RB_os_ts) < 45 and (ship1_u > (ship2_u * 1.1)):
        rule = 'OT-GW'
    # Crossing, give way
    elif 0 < RB_os_ts < 112.5 and 10 > RB_ts_os > -112.5:
        rule = 'CR-SO'
    # Crossing, stand on
    elif 10 > RB_os_ts > -112.5 and 0 < RB_ts_os < 112.5:
        rule = 'CR-GW'
    else:
        rule = 'Null'
    return rule

# if __name__ == '__main__':
    # ship1_x = -5000
    # ship1_y = 0
    # ship1_psi = 0
    # ship1_u = 20
    # ship2_x = 0
    # ship2_y = 5000
    # ship2_psi = 220
    # ship2_u = 20

    ship1_x = 0
    ship1_y = 500
    ship1_psi = 270
    ship1_u = 25
    ship2_x = -500
    ship2_y = 0
    ship2_psi = 0
    ship2_u = 30

    print(true_bearing(ship1_x, ship1_y, ship2_x, ship2_y))
    print(true_bearing(ship2_x, ship2_y, ship1_x, ship1_y))
    print(relative_bearing(ship1_x, ship1_y, ship1_psi, ship2_x, ship2_y))
    print(relative_bearing(ship2_x, ship2_y, ship2_psi, ship1_x, ship1_y))
    print(colregs_rule(ship1_x, ship1_y, ship1_psi, ship1_u, ship2_x, ship2_y, ship2_psi, ship2_u))
    print(colregs_rule(ship2_x, ship2_y, ship2_psi, ship2_u, ship1_x, ship1_y, ship1_psi, ship1_u))

    # ship2_psi = 210
    # print(relative_bearing(ship1_x, ship1_y, ship1_psi, ship2_x, ship2_y))
    # print(relative_bearing(ship2_x, ship2_y, ship2_psi, ship1_x, ship1_y))
    # print(colregs_rule(ship1_x, ship1_y, ship1_psi, ship1_u, ship2_x, ship2_y, ship2_psi, ship2_u))
    # print(colregs_rule(ship2_x, ship2_y, ship2_psi, ship2_u, ship1_x, ship1_y, ship1_psi, ship1_u))

def delete_files(dir_path):
    """
    delete all files in the given folder
    :param dir_path: the path of the given folder
    :return:
    """
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))  # delete files

if __name__ == '__main__':
    dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedTrainingProcess'
    delete_files(dir)