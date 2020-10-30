# -*- coding: utf-8 -*-

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import random

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
#%%

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

def initial_cond():
    th1 = np.radians(np.random.ranf(1)[0]*360)
    w1 = 0.0
    th2 = np.radians(np.random.ranf(1)[0]*360)
    w2 = 0.0
    return [th1, w1, th2, w2]

def velo_sq(x,y):
    v = lambda q: np.array([q[i+1]-q[i] for i in range(len(q)-1)]) /dt
    v_sq = v(x)**2 + v(y)**2
    return v_sq

def solve(fname):
     state = initial_cond()
     y = integrate.odeint(derivs, state, t)
     
     #cartesian coordinates of the bobs
     x1 = L1*sin(y[:, 0])
     y1 = -L1*cos(y[:, 0])

     x2 = L2*sin(y[:, 2]) + x1
     y2 = -L2*cos(y[:, 2]) + y1
     
     #velocity square of the bobs - needed to calculate energy later on
     v1_sq = np.concatenate([[0],velo_sq(x1,y1)])
     v2_sq = np.concatenate([[0],velo_sq(x1,y1)])
     
     with open(r"C:\Users\balin\Desktop\7. félév\compsim\proj1\data\training\n{:0>3d}".format(fname)+'.txt',mode='w') as f:
         for i in range(len(v1_sq)):
             f.write("{},{},{},{},{},{},{},{}\n".format(x1[i], y1[i], y[i,1], x2[i], y2[i], y[i,3], v1_sq[i], v2_sq[i]))
         
     with open(r"C:\Users\balin\Desktop\7. félév\compsim\proj1\data\training\init.txt", mode = "a") as g:
        g.write("{},{},{},{}\n".format(*state))
        #write data into file: cartesian coordinates, angular velocities, linear velocities squared
        
def solve_no_file():
    state = initial_cond()
    y = integrate.odeint(derivs, state, t)
     
    #cartesian coordinates of the bobs
    x1 = L1*sin(y[:, 0])
    y1 = -L1*cos(y[:, 0])

    x2 = L2*sin(y[:, 2]) + x1
    y2 = -L2*cos(y[:, 2]) + y1
    
    #velocity square of the bobs - needed to calculate energy later on
    v1_sq = np.concatenate([[0],velo_sq(x1,y1)])
    v2_sq = np.concatenate([[0],velo_sq(x1,y1)])
    
    return np.array([x1, y1, y[:,1], x2, y2, y[:,3]],dtype = np.longdouble).T
    
             
#%%
#integration parameters
dt = 0.01
t = np.arange(0.0, 100, dt)