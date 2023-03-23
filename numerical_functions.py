#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:47:45 2023

@author: niamhocallaghan
"""

import numpy as np

def coriolis_u(y):
    
    f0 = 1e-4
    beta = 1e-11
    
    f = f0 + beta * y
    
    return f

def coriolis_v(y):
    
    f0 = 1e-4
    beta = 1e-11
    
    f = f0 + beta * y
    
    return f

def tau_x(tau_0,y,L):
    
    tau = - tau_0 * np.cos(np.pi * y/L)
    
    return tau

def tau_y(y):
    
    tau = 0
    
    return tau

def deta_x(eta):
    
    dn = np.diff(eta,axis=1)
    
    return dn

def deta_y(eta):
    
    dn = np.diff(eta,axis=0)
    
    return dn
    
def v_on_u(v):
    
    v = (v[1:,1:] + v[1:,:-1] + v[:-1,1:] + v[:-1,:-1])/4
    
    return v

def u_on_v(u):
    
    u = (u[:-1,:-1] + u[1:,:-1] + u[:-1,1:] + u[1:,1:])/4
    
    return u

def u_on_eta(u):
    
    u = (u[:,:-1] + u[:,1:])/2
    
    return u

def v_on_eta(v):
    
    v = (v[1:,:] + v[:-1,:])/2
    
    return v

def energy(rho,H,u,v,g,eta,dx,dy):
    
    e = np.sum(1/2 * rho * (H * (u**2 + v**2) + g * eta**2)) * dx * dy
    
    return e

def eta_solver(H,dt,u,v,dx,dy):
    
    #
    eta = - H * ((np.diff(u,axis=1))/dx + (np.diff(v,axis=0))/dy)

    return eta

def u_vel_solver(f_u,dt,v,g,eta1,gamma,u,t_x,rho,H,dx):
    
    vel = f_u * v_on_u(v) - g * deta_x(eta1)/dx - gamma * u[:,1:-1] + t_x/(rho * H) 
    
    return vel

def v_vel_solver(f_v,dt,u,g,eta1,gamma,v,y_v,rho,H,dy):
    
    vel = - f_v * u_on_v(u) - g * deta_y(eta1)/dy - gamma * v[1:-1,:]
    
    return vel