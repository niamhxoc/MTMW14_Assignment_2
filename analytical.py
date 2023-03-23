# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:51:42 2023

@author: iq828562
"""

import numpy as np
from analytical_functions import *
import matplotlib.pyplot as plt

# Physical parameters
L = 1e6
f0 = 1e-4
beta = 1e-11
g = 10
gamma = 1e-6
rho = 1000
H = 1000
tau_0 = 0.2
epsilon = gamma/(L * beta)
a = (-1 - np.sqrt(1 + (2 * np.pi * epsilon)**2))/(2 * epsilon)
b = (-1 + np.sqrt(1 + (2 * np.pi * epsilon)**2))/(2 * epsilon)

def analytical_plot():
    # Set up domain
    nx = 40
    dx = 2.5E4
    ny = 40
    dy = 2.5E4
    #x_a = np.linspace(0,L,nx)
    #y_a = np.linspace(0,L,ny)
    
    # Set up arrays
    u_a = np.zeros((ny,nx+1))
    v_a = np.zeros((ny+1,nx))
    eta_a = np.zeros((ny,nx))
    x_au = np.tile(np.arange(nx+1)*dx,(nx,1))
    y_au = np.flipud(np.transpose(np.tile((0.5+np.arange(ny))*dy,(nx+1,1))))
    x_av = np.tile((0.5+np.arange(nx))*dx,(nx+1,1))
    y_av = np.flipud(np.transpose(np.tile(np.arange(ny+1)*dy,(nx,1))))
    x_an = np.tile((0.5+np.arange(nx))*dx,(nx,1))
    y_an = np.flipud(np.transpose(np.tile((0.5+np.arange(ny))*dy,(nx,1))))
    
    u_a = u_analytical(x_au,y_au,u_a)
    v_a = v_analytical(x_av,y_av,v_a)
    eta_a = eta_analytical(x_an,y_an,0,eta_a)
    
    
    #print(eta_analytical(x_a,y_a,0))
    
    xplt = (0.5 + np.arange(nx)) * dx
    yplt = (0.5 + np.arange(nx)) * dy
    X, Y = np.meshgrid(xplt,yplt)
    fig, ax = plt.subplots()
    eta_a_plot = ax.contourf(X,Y,eta_a)
    plt.colorbar(eta_a_plot)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    plt.title('Analytical Eta contour plot')
    #X_a, Y_a = np.meshgrid(x_an,y_an)
    #plt.contourf(X_a,Y_a,eta_a)
