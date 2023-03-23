# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:00:36 2023

@author: iq828562
"""
import numpy as np

L = 1e6
f0 = 1e-4
beta = 1e-11
g = 10
gamma = 1e-6
rho = 1e3
H = 1e3
tau_0 = 0.2
epsilon = gamma/(L * beta)
a = (-1 - np.sqrt(1 + (2 * np.pi * epsilon)**2))/(2 * epsilon)
b = (-1 + np.sqrt(1 + (2 * np.pi * epsilon)**2))/(2 * epsilon)

def f1_x(x):
    
    f1 = np.pi * (1 + ((np.exp(a) - 1) * np.exp(b * x) + (1 - np.exp(b)) * 
                       np.exp(a * x))/(np.exp(b) - np.exp(a)))
        
    return f1

def f2_x(x):
    
    f2 = ((np.exp(a) - 1) * b * np.exp(b * x) + (1 - np.exp(b)) * a * 
          np.exp(a * x))/(np.exp(b) - np.exp(a))
        
    return f2

def u_analytical(x,y,u):
    u[:,1:-1] = - tau_0/(np.pi*gamma*rho*H) * f1_x(x[:,1:-1]/L) * np.cos((np.pi * y[:,1:-1])/L)
    
    return u

def v_analytical(x,y,v):
    
    v[1:-1,:] = tau_0/(np.pi*gamma*rho*H) * f2_x(x[1:-1,:]/L) * np.sin((np.pi * y[1:-1,:])/L)
    
    return v

def eta_analytical(x,y,eta_0,eta_a):
    
    eta = eta_0 + tau_0/(np.pi*gamma*rho*H) * (f0 * L)/g * ((gamma/(f0 *
        np.pi) * f2_x(x/L) * np.cos((np.pi * y)/L) + 1/np.pi * f1_x(x/L) * 
            (np.sin((np.pi * y)/L) * (1 + (beta * y)/f0) + (beta * L)/(f0 * 
                np.pi) * np.cos((np.pi * y)/L))))
    
    return eta