from math import *
import numpy as np
np.set_printoptions(threshold=np.nan)
import os
import matplotlib.pyplot as plt
from decimal import *
import mpmath as mmp
from random_generator import get_range


def changecord(theta, phi):
    R = 6371000
    x = R * sin(theta) * cos(phi)
    y = R * sin(theta) * sin(phi)
    z = R * cos(theta)
    return np.array([x, y, z])


"""
this method returns the area between 4 points on sphere
"""
def get_each(pos):
    mmp.mp.dps = 30
    return mmp.mpf(pos[0]), mmp.mpf(pos[1]), mmp.mpf(pos[2])
# put in 1, 2, 3, get angle of 1
def get_ang(pos_1, pos_2, pos_3):

    mmp.mp.dps = 30
    x1, y1, z1 = get_each(pos_1)
    x2, y2, z2 = get_each(pos_2)
    x3, y3, z3 = get_each(pos_3)
    cosb = (x1 * x2 + y1 * y2 + z1 * z2)/(mmp.sqrt(x1**2 + y1**2 + z1**2) * mmp.sqrt(x2**2 + y2**2 + z2**2))
    cosc = (x1 * x3 + y1 * y3 + z1 * z3)/(mmp.sqrt(x1**2 + y1**2 + z1**2) * mmp.sqrt(x3**2 + y3**2 + z3**2))
    cosa = (x3 * x2 + y3 * y2 + z3 * z2)/(mmp.sqrt(x3**2 + y3**2 + z3**2) * mmp.sqrt(x2**2 + y2**2 + z2**2))

    sinb = mmp.sin(mmp.acos(cosb))
    sinc = mmp.sin(mmp.acos(cosc))

    cosA = (cosa - cosb * cosc) /(sinb * sinc)

    return mmp.acos(cosA)

def get_area(pos_1, pos_2, pos_3, pos_4):
    mmp.mp.dps = 30

    ang1 = get_ang(pos_1, pos_2, pos_3)
    ang2 = get_ang(pos_2, pos_4, pos_1)
    ang4 = get_ang(pos_4, pos_3, pos_2)
    ang3 = get_ang(pos_3, pos_1, pos_4)
    return float(mmp.power(6371000, 2) * (ang1 + ang2 + ang3 + ang4 - 2*pi))

def get_solid_ang(theta1, theta2, phi1, phi2):
    if abs(phi1 - phi2) < pi:
        return abs(cos(theta1) - cos(theta2)) * abs(phi1 - phi2)
    else:
        return abs(cos(theta1) - cos(theta2)) * abs(abs(phi1 - phi2) - 2*pi)


def get_factor(theta_index, phi_index):
    theta_min, theta_max, phi_min, phi_max, zenith_min, zenith_max, azimuth_min, azimuth_max = get_range(theta_index, phi_index)


    # 1,2,3,4 from left top to right top
    pos1 = changecord(theta_min, phi_min)
    pos2 = changecord(theta_max, phi_min)
    pos3 = changecord(theta_max, phi_max)
    pos4 = changecord(theta_min, phi_max)

    area = get_area(pos1, pos4, pos2, pos3)

    solid_ang = get_solid_ang(zenith_min, zenith_max, azimuth_min, azimuth_max)

    return area * solid_ang, theta_max, theta_min

list = []
theta_min_l = []
theta_max_l = []
for i in range(20):
    a, t_max, t_min = get_factor(3, i)
    theta_min_l.append(t_min)
    theta_max_l.append(t_max)
    list.append(a)
    print(a)

list = np.array(list)/np.average(np.array(list))

phi_list = np.linspace(0, 2*pi, 20)

plt.plot(phi_list, list)
plt.xlabel('phi list from 0 to 2pi')
plt.ylabel('normalized A*sigma')
plt.show()

f2 = plt.figure()
plt.plot(phi_list, theta_min_l)
plt.title('min theta')
f = plt.figure()
plt.plot(phi_list, theta_max_l)
plt.title('max theta')
plt.show()
