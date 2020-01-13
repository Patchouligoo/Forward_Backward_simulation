import numpy as np
from math import *
import matplotlib.pyplot as plt


def get_rand_ini(theta_index, phi_index):
    f = open("./outputfile/cos_" + str(theta_index) + "/phi_" + str(phi_index) + ".txt", 'r')

    theta = []
    phi = []
    azimuth = []
    zenith = []
    energy_list = []

    while 1:
        temp = f.readline().split()

        if len(temp) < 1:
            break

        theta.append(float(temp[0]))
        phi.append(float(temp[1]))
        zenith.append(float(temp[2]))
        azimuth.append(float(temp[3]))
        energy_list.append(float(temp[4]))

    theta = np.array(theta)
    phi = np.array(phi)
    azimuth = np.array(azimuth)
    zenith = np.array(zenith)
    energy_list = np.array(energy_list)

    theta_min = np.min(theta)
    theta_max = np.max(theta)
    theta_pos = np.random.random() * (theta_max - theta_min) + theta_min

    azimuth_min = np.min(azimuth)
    azimuth_max = np.max(azimuth)
    azimuth_dir = np.random.random() * (azimuth_max - azimuth_min) + azimuth_min

    """------------------phi-------------------"""
    if phi_index != 0 and phi_index != 19:
        phi_min = np.min(phi)
        phi_max = np.max(phi)
    else:
        phi_pos = []
        phi_neg = []
        for ang in phi:
            if ang > 6:
                phi_neg.append(ang - 2 * pi)
            else:
                phi_pos.append(ang)

        phi_pos = np.array(phi_pos)
        phi_neg = np.array(phi_neg)

        phi_min = np.min(phi_neg)
        phi_max = np.max(phi_pos)
    phi_pos = np.random.random() * (phi_max - phi_min) + phi_min
    if phi_pos < 0:
        phi_pos = phi_pos + 2 * pi

    """-----------------zenith and energy ------------------"""
    e_max = np.max(energy_list)
    e_min = np.min(energy_list)
    zenith_min = np.min(zenith)
    zenith_max = np.max(zenith)

    """
    zenith_list = np.linspace(zenith_min, zenith_max, 100)
    e_list = np.linspace(e_min, e_max, 10000)

    factor = []

    for i in range(10000):  # e
        for j in range(100):  # theta
            temp = 1 / (1 + 1.1 * e_list[i] * cos(zenith_list[j]) / (115 * pow(10, 9))) + 0.054 / (
                        1 + 1.1 * e_list[i] * cos(zenith_list[j]) / (850 * pow(10, 9)))
            temp = temp * pow(e_list[i], -2.7)
            factor.append(temp)

    factor = np.array(factor)

    factor = factor / np.sum(factor)

    index = np.arange(1000000)

    res = np.random.choice(index, p=factor)

    # print(res)

    res_i = res // 100

    # print(res_i)

    res_j = res - res_i * 100

    energy = e_list[res_i]

    zenith_dir = zenith_list[res_j]
    
    """

    for i in range(20):
        temp_arr = np.linspace(e_min, e_max, 1000)
        temp_index = np.arange(999)
        p_list = []
        for j in range(1, 1000):
            # -2.7 * E^-2.7
            temp_p = -2.7 * pow(temp_arr[j], -2.7) - (-2.7 * pow(temp_arr[j - 1], -2.7))
            p_list.append(temp_p)
        p_list = np.array(p_list)/np.sum(np.array(p_list))
        res = np.random.choice(temp_index, p=p_list)
        e_min = temp_arr[res]
        e_max = temp_arr[res + 1]

    energy = (e_min + e_max)/2

    zenith_list = np.linspace(zenith_min, zenith_max, 100)

    factor = 1 / (1 + 1.1 * energy * np.cos(zenith_list) / (115 * pow(10, 9))) + 0.054 / (
                        1 + 1.1 * energy * np.cos(zenith_list) / (850 * pow(10, 9)))
    factor = np.array(factor)/np.sum(np.array(factor))

    zenith_dir = np.random.choice(zenith_list, p=factor)

    """
    m = []
    for i in range(10000):
        zenith_dir = np.random.choice(zenith_list, p=factor)
        m.append(zenith_dir)

    plt.hist(m, bins=100)

    plt.show()
    exit()
    """
    #print(e_max)
    #print(e_min)
    print(energy)

    return theta_pos, phi_pos, azimuth_dir, zenith_dir, energy

#print(get_rand_ini(4, 0))

def get_range(theta_index, phi_index):
    f = open("./outputfile/cos_" + str(theta_index) + "/phi_" + str(phi_index) + ".txt", 'r')

    theta = []
    phi = []
    azimuth = []
    zenith = []
    energy_list = []

    while 1:
        temp = f.readline().split()

        if len(temp) < 1:
            break

        theta.append(float(temp[0]))
        phi.append(float(temp[1]))
        zenith.append(float(temp[2]))
        azimuth.append(float(temp[3]))
        energy_list.append(float(temp[4]))

    theta = np.array(theta)
    phi = np.array(phi)
    azimuth = np.array(azimuth)
    zenith = np.array(zenith)

    theta_min = np.min(theta)
    theta_max = np.max(theta)

    azimuth_min = np.min(azimuth)
    azimuth_max = np.max(azimuth)

    """------------------phi-------------------"""
    if phi_index != 0 and phi_index != 19:
        phi_min = np.min(phi)
        phi_max = np.max(phi)
    else:
        phi_pos = []
        phi_neg = []
        for ang in phi:
            if ang > 6:
                phi_neg.append(ang - 2 * pi)
            else:
                phi_pos.append(ang)

        phi_pos = np.array(phi_pos)
        phi_neg = np.array(phi_neg)

        phi_min = np.min(phi_neg)
        phi_max = np.max(phi_pos)
        phi_min = phi_min + 2 * pi

    """-----------------zenith------------------"""
    zenith_min = np.min(zenith)
    zenith_max = np.max(zenith)

    return theta_min, theta_max, phi_min, phi_max, zenith_min, zenith_max, azimuth_min, azimuth_max
