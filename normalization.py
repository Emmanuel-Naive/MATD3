"""
Code for normalizing different data to the interval [-1, 1]
"""


class NormalizeData:
    def __init__(self, dis_max=5000):
        self.dis_max = dis_max

    def nmlz_pos(self, pos_c, pos_min, pos_max):
        """
        Function for normalizing the given position(x, y) to the interval [-1, 1]
            e.g.； if pos_c == pos_min, return: n_pos_c = 0
        :param pos_c: the given position
        :param pos_min: the minimum value of initial and goal positions
        :param pos_max: the maximum value of initial and goal positions
        :return:
        """
        if pos_min == pos_max:
            n_pos_c = 2 * (pos_c - pos_min) / self.dis_max
        else:
            n_pos_c = 2 * (pos_c - pos_min) / (pos_max - pos_min)
        return n_pos_c

    def nmlz_ang(self, angle_c):
        """
        Function for normalizing the given heading angle to the interval [-1, 1]
            e.g.: if angle_c = 180, return: n_ang_c = 0
        :param angle_c: the interval [0, 360]
        :return:
        """
        n_ang_c = (180 - angle_c) / 180
        return n_ang_c

    def nmlz_r(self, r_c, r_max):
        """
        Function for normalizing the given reward to the interval [-1, 1]
            e.g.: if r_c = 0, return: n_r_c = 0
        :param r_c: the given reward
        :param r_max: the maximum reward
        :return:
        """
        n_r_c = r_c / r_max
        return n_r_c


if __name__ == '__main__':
    nmlz = NormalizeData(10000)
    print(nmlz.nmlz_pos(5000, 5000, 5000))
    print(nmlz.nmlz_ang(90))
    print(nmlz.nmlz_r(20, 20))
