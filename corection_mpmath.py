from math import *
import numpy as np
from scipy.interpolate import UnivariateSpline
np.set_printoptions(threshold=np.nan)
import os
import matplotlib.pyplot as plt
from decimal import *
import mpmath as mmp

"""
===================================file header======================================
The file generated by the simulation has only position, direction, energy info, and 
so on and the final position is highly inaccurate, because in the last step the muon
will go far away beyond earth surface which causes error in E and distance

So this file will read all files generated by simulator, doing correction on final
position and energy, then for each file (each file corresponds to an energy level
with 28 rows and 73 columns representing different initial angle), the final dir and
energy will be converted to flux

The result for each file is a 28 x 73 matrix representing flux corresponds to the 
total flux map of a certain initial energy level. for energy step 0.1, 31 files will
be generated.

Then they will be used to do numerical integration in integration.py
=======================================end==========================================
"""

# phi1 is center, phi2 is deflected
def dPhi(phi1, phi2):
    if abs(phi2 - phi1) <= pi:
        return phi2 - phi1
    else:
        if phi2 > phi1:
            return -(2*pi - abs(phi2 - phi1))
        else:
            return 2*pi - abs(phi2 - phi1)


def changecord(theta, phi):
    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)

    return np.array([x, y, z])


def convertcord(dir):
    x = dir[0]
    y = dir[1]
    z = dir[2]

    R = sqrt(pow(x, 2) + pow(y,2) + pow(z,2))
    theta = acos(z/R)

    phi = acos(x/sqrt(x**2 + y**2))

    if y < 0:
        phi = 2 * pi - phi

    return theta, phi

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
    print(ang1 + ang2 + ang3 + ang4 - 2 * pi)
    return float(mmp.power(6371000, 2) * (ang1 + ang2 + ang3 + ang4 - 2*pi))


"""
gget energy loss per cm : dE/dx using :
dE/dx = 0.259GeV/wme + 0.363/wme * E(GeV)
dE/dx is eV per cm
"""
def getEloss(E):

    E = E / pow(10, 9)
    totalloss = (0.000363 * E + 0.259) / 100 * 0.9167 * pow(10, 9)


    return totalloss



"""
This method return the new energy for initial energy E and new distance traveled forward as dis

the energy loss per cm is dE/dx, which we get from getEloss(E)
a simple forward euler method is performed to get new energy after dis
since the dis must be less than 1000m due to our set up in simulator, it should be accurate
"""
def getnewE_p(E, dis):

    deltax = 0.1
    num = int(dis/deltax) + 10
    deltax = dis/num


    for i in range(0, num):
        E_loss1 = getEloss(E)
        E_temp = E + 100 * deltax * E_loss1
        E_loss2 = getEloss(E_temp)
        E_loss = (E_loss1 + E_loss2)/2

        E = E + 100 * deltax * E_loss
    return E

"""
Same as previous one, but this time we get PREVIOUS !!! energy for energy E and dis traveled BACKWARD !!!

which means we over run during the correction and need to go back
"""
def getnewE_m(E, dis):
    deltax = 0.1
    num = int(dis / deltax) + 10
    deltax = dis / num


    for i in range(0, num):
        E_loss1 = getEloss(E)
        E_temp = E - 100 * deltax * E_loss1
        E_loss2 = getEloss(E_temp)
        E_loss = (E_loss1 + E_loss2)/2

        E = E - 100 * deltax * E_loss


    return E



def correction(pos, prepos, preenergy, predis):

    """-----------------------correction of final step--------------------------"""
    norm = np.linalg.norm(pos - prepos)  # length of final step
    dir = (pos - prepos) / norm  # direction unit vector of final step

    try:
        # this algorithm works like a binary search. Until the difference between radius r of final
        # position and radius of earth is less than 0.001m, the algorithm will use current step size / 2
        # as step for next time. If difference is greater than 0, then we go back, if it is less than 0
        # then we go forward.
        step = norm / 2

        while abs(np.linalg.norm(prepos) - 6371000) > 0.001:

            # go back! we get out of earth surface too far
            if np.linalg.norm(prepos) - 6371000 > 0:
                prepos = prepos - dir * step
                predis = predis - step
                preenergy = getnewE_m(preenergy, step)  # decrease energy using FIRST method
                step = step / 2

            # keep going! we are still inside the earth
            else:
                prepos = prepos + dir * step
                predis = predis + step
                preenergy = getnewE_p(preenergy, step)  # increase energy using SECOND method
                step = step / 2

        # in the end, prepos is the correct position and preenergy is the correct energy
        pos = prepos
        energy = preenergy
    except:
        energy = 0
        print("error!")
        exit()

    return energy, pos



"""============================================main body==============================================="""
def handle(current, num):

    numphi = 20
    R = 6371000

    f = open("./cos_" + str(current) + "/pos_" + str(num) + ".txt", "r")

    count = 0

    area_list = []

    while count < numphi:

        temp1 = f.readline().split()
        temp2 = f.readline().split()
        temp3 = f.readline().split()
        temp4 = f.readline().split()
        f.readline()

        """for point 1"""
        dis1 = float(temp1[3])
        predis1 = float(temp1[11])
        pos1 = np.array([float(temp1[5]), float(temp1[6]), float(temp1[7])])
        prepos1 = np.array([float(temp1[8]), float(temp1[9]), float(temp1[10])])
        preenergy1 = float(temp1[12])

        energy_1, pos_1 = correction(pos1, prepos1, preenergy1, predis1)
        # theta1, phi1 = convertcord(pos_1)
        v_dir1 = (pos1 - prepos1)/np.linalg.norm(pos1 - prepos1)
        zenith1 = acos(np.dot(v_dir1, pos_1)/(np.linalg.norm(pos_1) * np.linalg.norm(v_dir1)))

        """for point 2"""
        dis2 = float(temp2[3])
        predis2 = float(temp2[11])
        pos2 = np.array([float(temp2[5]), float(temp2[6]), float(temp2[7])])
        prepos2 = np.array([float(temp2[8]), float(temp2[9]), float(temp2[10])])
        preenergy2 = float(temp2[12])

        energy_2, pos_2 = correction(pos2, prepos2, preenergy2, predis2)
        # theta2, phi2 = convertcord(pos_2)
        v_dir2 = (pos2 - prepos2) / np.linalg.norm(pos2 - prepos2)
        zenith2 = acos(np.dot(v_dir2, pos_2) / (np.linalg.norm(pos_2) * np.linalg.norm(v_dir2)))

        """for point 3"""
        dis3 = float(temp3[3])
        predis3 = float(temp3[11])
        pos3 = np.array([float(temp3[5]), float(temp3[6]), float(temp3[7])])
        prepos3 = np.array([float(temp3[8]), float(temp3[9]), float(temp3[10])])
        preenergy3 = float(temp3[12])

        energy_3, pos_3 = correction(pos3, prepos3, preenergy3, predis3)
        # theta3, phi3 = convertcord(pos_3)
        v_dir3 = (pos3 - prepos3) / np.linalg.norm(pos3 - prepos3)
        zenith3 = acos(np.dot(v_dir3, pos_3) / (np.linalg.norm(pos_3) * np.linalg.norm(v_dir3)))

        """for point 4"""
        dis4 = float(temp4[3])
        predis4 = float(temp4[11])
        pos4 = np.array([float(temp4[5]), float(temp4[6]), float(temp4[7])])
        prepos4 = np.array([float(temp4[8]), float(temp4[9]), float(temp4[10])])
        preenergy4 = float(temp4[12])

        energy_4, pos_4 = correction(pos4, prepos4, preenergy4, predis4)
        # theta4, phi4 = convertcord(pos_4)
        v_dir4 = (pos4 - prepos4) / np.linalg.norm(pos4 - prepos4)
        zenith4 = acos(np.dot(v_dir4, pos_4) / (np.linalg.norm(pos_4) * np.linalg.norm(v_dir4)))

        #todo: not been used !!!!!!!!!
        #print(str(degrees(zenith1)) + " " + str(degrees(zenith2)) + " " + str(degrees(zenith3)) + " " + str(degrees(zenith4)))

        area = get_area(pos_1, pos_2, pos_3, pos_4)

        area_list.append(area)

        count = count + 1


    return area_list


# current = 4     # 0 to 4
# num = 0         # 0 to 63

def output():
    if not os.path.exists("./outputfile_2"):
        os.mkdir("./outputfile_2")

    for j in range(125):
        f = open("./outputfile_2/pos_" + str(j), "w")
        for i in [0, 1, 2, 3, 4]:
            area_list = handle(i, j)
            f.write(str(area_list))
            f.write('\n')
            print()
        f.close()
        print("finish " + str(j))



output()