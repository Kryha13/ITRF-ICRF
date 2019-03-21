from math import pi, cos, sin
from scipy.interpolate import interp1d
import numpy as np
from numpy import transpose
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def deg2rad(st, min, sec):
    deg = st + min/60 + sec/3600
    rad = deg * pi/180
    return rad


def mjd(day, month, year):
    if month <= 2:
        year = year - 1
        month = month + 12
    else:
        pass
    i = int(year / 100)
    k = 2 - i + int(i / 4)
    mjd = int(365.25 * year) - 679006 + int(30.6001 * (month + 1)) + k + day
    return mjd


def interpolation(to_interp):
    igs = np.loadtxt('igs')
    mjd = igs[:, 0]
    f = interp1d(mjd, to_interp, kind='cubic')

    return f()

def interpolationbieg(to_interp):
    nut = np.loadtxt('nutation.txt')
    mjd = nut[:, 0]
    f = interp1d(mjd, to_interp, kind='cubic')

    return f(mjdTT)


def interpolationtgw(to_interp):
    nut = np.loadtxt('nutation.txt')
    mjd = nut[:, 0]
    f = interp1d(mjd, to_interp, kind='cubic')

    return f(mjdUT1)

