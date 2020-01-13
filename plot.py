from math import *
import numpy as np
from scipy.interpolate import UnivariateSpline
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt


"""------------------------------------set front size----------------------------------------"""
SMALL_SIZE = 8 * 2
MEDIUM_SIZE = 10 * 2
BIGGER_SIZE = 12 * 2

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


"""------------------------------------------------------------------------------------------"""

im2D = np.zeros(20)

theta_index = 3

temp_list = []

for pos in [0, 20, 60, 100, 120]:

    f = open("./outputfile_2/pos_" + str(pos), "r")

    counter = 0
    while 1:
        if counter != theta_index:
            f.readline()
            counter = counter + 1
            continue
        temp = f.readline().split(', ')

        temp[0] = (temp[0])[1:]
        temp[len(temp) - 1] = (temp[len(temp) - 1][:-1])[:-1]
        t = []
        for i in range(len(temp)):
            t.append(float(temp[i]))
        temp = np.array(t)
        break
    print(temp)
    temp_list.append(temp)
    im2D = im2D + temp



im2D = im2D/np.average(im2D)

f = plt.figure()

plt.plot(range(20), im2D, label='total')

for i in range(len(temp_list)):
    plt.plot(range(20), temp_list[i]/np.average(temp_list[i]), label=str(i))

plt.xlabel('phi')
plt.ylabel('cos theta')
plt.legend()
plt.show()

"""
x = np.linspace(0, 2*pi, 20)
cost = np.linspace(0.2, 0, 5)

f0 = plt.figure()
print(im2D[0])
plt.plot(x, im2D[0])
plt.xlabel('phi')
plt.ylabel('area')
plt.title('cos = ' + str(round(cost[0], 3)))

f1 = plt.figure()
print(im2D[1])
plt.plot(x, im2D[1])
plt.xlabel('phi')
plt.ylabel('area')
plt.title('cos = ' + str(round(cost[1], 3)))

f2 = plt.figure()
print(im2D[2])
plt.plot(x, im2D[2])
plt.xlabel('phi')
plt.ylabel('area')
plt.title('cos = ' + str(round(cost[2], 3)))

f3 = plt.figure()
print(im2D[3])
plt.plot(x, im2D[3])
plt.xlabel('phi')
plt.ylabel('area')
plt.title('cos = ' + str(round(cost[3], 3)))

f4 = plt.figure()
print(im2D[4])
plt.plot(x, im2D[4])
plt.xlabel('phi')
plt.ylabel('area')
plt.title('cos = ' + str(round(cost[4], 3)))
plt.show()
"""