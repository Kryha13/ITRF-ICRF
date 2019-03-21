from math import pi, cos, sin
from scipy.interpolate import interp1d
import numpy as np
from numpy import transpose
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# przeliczanie stopni na radiany


def deg2rad(st, min, sec):
    deg = st + min/60 + sec/3600
    rad = deg * pi/180
    return rad

# MJD


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


today = mjd(14, 9, 2018)  # tu zmienic dzien
ep = np.loadtxt('epoki')
h = ep[:, 0]
m = ep[:, 1]
tGPS = np.array(h) + np.array(m)/60
mjdGPS = today + np.array(tGPS)/24

#  interpolacja parametrów ruchu obrotowego Ziemi

igs = np.loadtxt('igs')
x = igs[:, 1]
y = igs[:, 2]
ut1_utc = igs[:, 3]


def interpolation(to_interp):
    igs = np.loadtxt('igs')
    mjd = igs[:, 0]
    f = interp1d(mjd, to_interp, kind='cubic')

    return f(mjdGPS)

# różnica ut1 utc


ut1utc = interpolation(ut1_utc)


# RUCH BIEGUNA

#  współrzędne bieguna

e = interpolation(x)
f = interpolation(y)

xrad = deg2rad(0, 0, np.array(e)*10**-6)
yrad = deg2rad(0, 0, np.array(f)*10**-6)

RM = np.array([[1, 0, xrad[1]], [0, 1, -yrad[1]], [-xrad[1], yrad[1], 1]])
RM = transpose(RM)

# czasy na podstawie wyinterpolowanego ut1_utc

tUT1 = np.array(tGPS) - 18/3600 + (np.array(ut1utc)*10**-7)/3600  # przeliczenie tgps na tut1
tTT = np.array(tGPS) + 19/3600 + 32.184/3600  # przeliczenie tgps na tt

#  NUTACJA

mjdTT = today + np.array(tTT)/24

nut = np.loadtxt('nutation.txt')
dps = nut[:, 1]
dep = nut[:, 2]


def interpolationbieg(to_interp):
    nut = np.loadtxt('nutation.txt')
    mjd = nut[:, 0]
    f = interp1d(mjd, to_interp, kind='cubic')

    return f(mjdTT)


dpsi = deg2rad(0, 0, interpolationbieg(dps))
deps = deg2rad(0, 0, interpolationbieg(dep))

mjdj2000 = 51544.5
T = (np.array(mjdTT) - mjdj2000) / 36525
E = deg2rad(23, 26, 21.448) - deg2rad(0, 0, 46.8150)*np.array(T) - deg2rad(0, 0, 0.00059)*np.array(T)**2 + deg2rad(0, 0, 0.001813)*np.array(T)**3
cosE = np.cos(E)


# czas gwiazdowy

mjdUT1 = today + np.array(tUT1)/24  # mjdgps


def interpolationtgw(to_interp):
    nut = np.loadtxt('nutation.txt')
    mjd = nut[:, 0]
    f = interp1d(mjd, to_interp, kind='cubic')

    return f(mjdUT1)


dpsi_ut1 = interpolationtgw(dps)
deps_ut1 = interpolationtgw(dep)

Ttgw = (np.array(mjdUT1) - mjdj2000) / 36525
Etgw = deg2rad(23, 26, 21.448) - deg2rad(0, 0, 46.8150)*np.array(Ttgw) - deg2rad(0, 0, 0.00059)*np.array(Ttgw)**2 + deg2rad(0, 0, 0.001813)*np.array(Ttgw)**3
cosEtgw = np.cos(Etgw)


def tgwsr(): # czas gwiazdowy średni - policzone w UT1
    mjdj2000 = 51544.5
    # mjdUT1 = today + tUT1/24  # 0 UT1
    T = (mjdUT1 - mjdj2000)/36525
    M = (24110.54841 + 8640184.812866 * T + 0.093104 * T**2 - (6.2*10**-6)*T**3)/3600
    db = np.array(M)/24
    Mp = M - np.floor(np.array(db))*24  # po odjeciu wielokrotnosci 24, w godzinach
    return Mp


b = tgwsr()


def tgw0():
    tgw =np.array(tUT1) + np.array(tgwsr()) + np.array((((dpsi_ut1/3600)/360)*24) * cosEtgw) #1.0027379093 *
    db = np.array(tgw)/24
    tgw0 = np.array(tgw) - np.floor(np.array(db))*24
    tgw0 = (tgw0/24) * 2*pi   #     tgw0 = ((tgw0/) * 360) *pi/180   #
    return tgw0


tgw = tgw0()

# PRECESJA - macierz dla kazdej epoki


mjdj2000 = 51544.5
mjdTT = today + np.array(tTT) / 24
T = (np.array(mjdTT) - mjdj2000) / 36525
zeta = deg2rad(0, 0, 2306.2182)*np.array(T)+ deg2rad(0, 0, 0.30188)*np.array(T)**2 + deg2rad(0, 0, 0.017998)*np.array(T)**3
z = deg2rad(0, 0, 2306.2182)*np.array(T) + deg2rad(0, 0, 1.09468)*np.array(T)**2 + deg2rad(0, 0, 0.018203)*np.array(T)**3
teta = deg2rad(0, 0, 2004.3109)*np.array(T) + deg2rad(0, 0, 0.42665)*np.array(T)**2 + deg2rad(0, 0, 0.041833)*np.array(T)**3


wsp = np.loadtxt('danegot.txt')  # tu zmienic XYZ ITRF
X = wsp[:, 0]
Y = wsp[:, 1]
Z = wsp[:, 2]
XYZ = np.array([X, Y, Z])
XYZ = transpose(XYZ)


# wykresy torów ruchów satelity w układzie ziemskim

# plt.plot(Y, X)
# plt.xlabel('Ytrf [km]')
# plt.ylabel('Xtrf [km]', labelpad=-10)
# plt.title('Tor ruchu satelity w układzie ziemskim - płaszczyzna XY')
# plt.grid()
# plt.savefig('torziemskixy.png')
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(Y, X, Z, c='r', marker='o')
# ax.tick_params(labelrotation=0)
# ax.set_xlabel('Ytrf [km]', labelpad=10)
# ax.set_ylabel('Xtrf [km]', labelpad=10)
# ax.set_zlabel('Ztrf [km]', labelpad=10)
# ax.set_title('Tor ruchu satelity w układzie ziemskim - 3D', pad=10)
# plt.savefig('torziemski3d.png')
# fig.show()

XYZcrf =np.array([])

for i  in range(0, len(X)):
    R2 = np.array([[cos(teta[i]), 0, -sin(teta[i])], [0, 1, 0], [sin(teta[i]), 0, cos(teta[i])]])
    R31 = np.array([[cos(-z[i]), sin(-z[i]), 0], [-sin(-z[i]), cos(-z[i]), 0], [0, 0, 1]])
    R32 = np.array([[cos(-zeta[i]), sin(-zeta[i]), 0], [-sin(-zeta[i]), cos(-zeta[i]), 0], [0, 0, 1]])
    RP = R31 @ R2 @ R32
    RP = transpose(RP)

    R1n = np.array([[1, 0 , 0], [0, cos(-(E[i]+deps[i])), sin(-(E[i]+deps[i]))], [0, -sin(-(E[i]+deps[i])), cos(-(E[i]+deps[i]))]])
    R3n = np.array([[cos(-dpsi[i]), sin(-dpsi[i]), 0], [-sin(-dpsi[i]), cos(-dpsi[i]), 0], [0, 0, 1]])
    R1n2 =  np.array([[1, 0, 0], [0, cos(E[i]), sin(E[i])], [0, -sin(E[i]), cos(E[i])]])
    RN = R1n @ R3n @ R1n2
    RN = transpose(RN)

    RS = np.array([[cos(tgw[i]), sin(tgw[i]), 0], [-sin(tgw[i]), cos(tgw[i]), 0], [0, 0, 1]])
    RS = transpose(RS)

    RM = np.array([[1, 0, xrad[i]], [0, 1, -yrad[i]], [-xrad[i], yrad[i], 1]])
    RM = transpose(RM)

    XYZcrf = np.append(XYZcrf, RP @ RN @ RS @ RM @ np.array(XYZ[i, :]))  #RP @ RN @ RS @

XYZcrf1 = np.reshape(XYZcrf, (96, 3))

# wykresy torów ruchów satelity w układzie niebieskim

plt.plot(XYZcrf1[:, 1], XYZcrf1[:, 0])
plt.xlabel('Ycrf [km]')
plt.ylabel('Xcrf [km]', labelpad=-10)
plt.title('Tor ruchu satelity w układzie niebieskim - płaszczyzna XY')
plt.grid()
plt.savefig('torxy.png')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(XYZcrf1[:, 1], XYZcrf1[:, 0], XYZcrf1[:, 2], c='r', marker='o')
ax.tick_params(labelrotation=0)
ax.set_xlabel('Ycrf [km]', labelpad=10)
ax.set_ylabel('Xcrf [km]', labelpad=10)
ax.set_zlabel('Zcrf [km]', labelpad=10)
ax.set_title('Tor ruchu satelity w układzie niebieskim - 3D', pad=10)
plt.savefig('tor3d.png')
fig.show()

# zapis wyników

np.savetxt('XYZcrf', XYZcrf1)
np.savetxt('mjdgps', mjdGPS)
np.savetxt('mjdut1', mjdUT1)
np.savetxt('mjdtt', mjdTT)
np.savetxt('zeta', zeta)
np.savetxt('teta', teta)
np.savetxt('z', z)
np.savetxt('ut1utc', ut1utc)
np.savetxt('dpsi', dpsi)
np.savetxt('deps', deps)
np.savetxt('xrad', xrad)
np.savetxt('yrad', yrad)
np.savetxt('tgw', tgw)

#  AZYMUT I WYSOKOŚĆ NAD JOZE

XJ = 3664.9400728
YJ = 1409.1539407
ZJ = 5009.5714312
fij = deg2rad(52.097, 0, 0)
lamj = deg2rad(21.032, 0, 0)

XYZJ = np.array([XJ, YJ, ZJ])

XS = XYZ - XYZJ

Rneu = np.array([[-sin(fij)*cos(lamj), -sin(lamj), cos(fij)*cos(lamj)], [-sin(fij)*sin(lamj), cos(lamj),
                                                                         cos(fij)*sin(lamj)],
                 [cos(fij), 0, sin(fij)]])

xneu = XS @ transpose(Rneu)
n = xneu[:, 0]
e = xneu[:, 1]
u = xneu[:, 2]

Az = np.array([])


for i in range(0, len(n)):
    if n[i] > 0 and e[i] > 0:
        Az = np.append(Az, np.arctan(np.divide(e[i], n[i])) * 180/pi)
    elif n[i] < 0 and e[i] > 0:
        Az = np.append(Az, 180 - abs((np.arctan(np.divide(e[i], n[i])) * 180 / pi)))
    elif n[i] < 0 and e[i] < 0:
        Az = np.append(Az, 180 + abs((np.arctan(np.divide(e[i], n[i])) * 180 / pi)))
    elif n[i] > 0 and e[i] < 0:
        Az = np.append(Az, 360 - abs((np.arctan(np.divide(e[i], n[i])) * 180 / pi)))

np.savetxt('azymut', Az)

EL = np.array([])

for i in range(0, len(u)):
    if u[i] > 0:
        EL =np.append(EL, 90 - np.arccos(np.divide(u[i], np.sqrt(n[i]**2 + e[i]**2 + u[i]**2))) *180/pi)
    elif u[i] < 0:
        EL = np.append(EL, 90 - np.arccos(np.divide(u[i], np.sqrt(n[i]**2 + e[i]**2 + u[i]**2))) * 180 / pi)

np.savetxt('elewacja', EL)

Az1 = np.ma.masked_where((Az < 150), Az)
Az2 = np.ma.masked_where((Az > 150), Az)

# wykresy azymut elewacja

plt.plot(tGPS, Az1)
plt.plot(tGPS, Az2)
plt.xlabel('Czas GPS [h]')
plt.ylabel('Azymut do satelity [°]')
plt.title('Zmiany azymutu do satelity PG06 w Józefosławiu w ciągu doby')
plt.grid()
plt.savefig('azymut.png')
plt.show()

EL2 = np.ma.masked_where((EL < 0), EL)
EL3 = np.ma.masked_where((EL > 0), EL)
plt.plot(tGPS, EL3)

plt.plot(tGPS, EL)
plt.xlabel('Czas GPS [h]')
plt.ylabel('Elewacja satelity [°]')
plt.title('Zmiany elewacji satelity PG06 nad Józefosławiem w ciągu doby')
plt.grid()
plt.savefig('elewacja.png')
plt.show()

