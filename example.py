# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:21:22 2020

@author: balin
"""

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import random
# for each experiment value of l1,l2,m1,m2 and th1,th2,w1,w2 are same so explicitely add these features after training.

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
               M2*G*sin(state[2])*cos(del_) +
               M2*L2*state[3]*state[3]*sin(del_) -
               (M1 + M2)*G*sin(state[0]))/den1

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
               (M1 + M2)*G*sin(state[0])*cos(del_) -
               (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
               (M1 + M2)*G*sin(state[2]))/den2

    return dydx

def velo_sq(x,y):
    v = lambda q: np.array([q[i+1]-q[i] for i in range(len(q)-1)]) /dt
    v_sq = v(x)**2 + v(y)**2
    return v_sq

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.01
t = np.arange(0.0, 100, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)

th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])

#print("x1 : ",x1)
#print("y1 : ",y1)

x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

v1_sq = np.concatenate([[0],velo_sq(x1,y1)])
v2_sq = np.concatenate([[0],velo_sq(x1,y1)])

def energy(v1_sq,v2_sq,y1,y2):
    kin = 0.5 * (v1_sq + v2_sq)
    pot = G*(y1 + y2)
    '''kin = 0.5*y[:,1]**2 + 0.5 * (y[:,1]**2+ y[:,3]**2 + 2* y[:,1] * y[:,3]*cos(y[:,0]-y[:,2]))
    pot = -cos(y[:,0]) * G - G* cos(y[:,2])'''
    return kin+pot

energy_arr = energy(v1_sq, v2_sq, y1, y2)


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=1)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
energy_text = ax.text(0.05, 0.83, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    return line, time_text, energy_text

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    energy_text.set_text("energy: %.2f J" % round(energy_arr[i],2))
    return line, time_text, energy_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=15, blit=True, init_func=init)

plt.plot(np.arange(len(energy_arr)),energy_arr)
#ani.save('double_pendulum.mp4', fps=15)

#plt.show()