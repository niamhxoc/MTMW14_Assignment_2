#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:28:17 2023

@author: niamhocallaghan
"""
from numerical_functions import *
import numpy as np
import matplotlib.pyplot as plt
from analytical_functions import *

# Physical parameters
L = 1e6
f0 = 1e-4
beta = 1e-11
g = 10
gamma = 1e-6
rho = 1000
H = 1000
tau_0 = 0.2

def fwdbwd(task):
    
    # Task time steps
    if task == 'D1' or task == 'E1':
        timed = 86400
    else:
        timed = 40 * 86400
    
    # Spatial steps
    if task == 'E3':
        nx = 80
        ny = 80
        dt = 50
    else: 
        nx = 40
        ny = 40
        dt = 175
    
    # Spatial step size
    dx = L/nx
    dy = L/ny
    nt = int(timed/dt)
    
    # x and y arrays
    x = np.arange(nx+1) * dx
    y = np.arange(nx+1) * dy
    
    # u v and eta arrays
    u = np.zeros((ny,nx+1))
    v = np.zeros((ny+1,nx))
    eta = np.zeros((ny,nx))
    u1 = np.zeros((ny,nx+1))
    v1 = np.zeros((ny+1,nx))
    u2 = np.zeros((ny,nx+1))
    v2 = np.zeros((ny+1,nx))
    
    # Compute coriolis
    y_u = (0.5 + np.arange(ny)) * dy
    f_u = np.array(np.tile(coriolis_u(y_u),(nx-1,1))).T
    y_v = np.arange(1,ny) * dy
    f_v = np.array(np.tile(coriolis_u(y_v),(ny,1))).T
    
    # Compute tau
    t_x = np.array(np.tile(tau_x(tau_0,y_u,L),(nx-1,1))).T
    t_y = np.array(np.tile(tau_y(y_v),(ny,1))).T   
    
    # Energy array
    E = []
    Ed = []
    
    # Analytical Parameters
    epsilon = gamma/(L * beta)
    a = (-1 - np.sqrt(1 + (2 * np.pi * epsilon)**2))/(2 * epsilon)
    b = (-1 + np.sqrt(1 + (2 * np.pi * epsilon)**2))/(2 * epsilon)
    
    # Set up analytical domain
    x_au = np.tile(np.arange(nx+1)*dx,(nx,1))
    y_au = np.flipud(np.transpose(np.tile((0.5+np.arange(ny))*dy,(nx+1,1))))
    x_av = np.tile((0.5+np.arange(nx))*dx,(nx+1,1))
    y_av = np.flipud(np.transpose(np.tile(np.arange(ny+1)*dy,(nx,1))))
    x_an = np.tile((0.5+np.arange(nx))*dx,(nx,1))
    y_an = np.flipud(np.transpose(np.tile((0.5+np.arange(ny))*dy,(nx,1))))
    
    # Set up analytical arrays
    u_a = np.zeros((ny,nx+1))
    v_a = np.zeros((ny+1,nx))
    eta_a = np.zeros((ny,nx))
    
    # Analytical solution
    u_a = u_analytical(x_au,y_au,u_a)
    v_a = v_analytical(x_av,y_av,v_a)
    eta_a = eta_analytical(x_an,y_an,-0.0805128358595952,eta_a)
    
    # Forward-Backward numerical solution
    for i in range(0,nt,2):

        # First run
        eta1 = eta - H * dt * ((np.diff(u,axis=1))/dx + (np.diff(v,axis=0))/dy)
        u1[:,1:-1] = u[:,1:-1] + f_u * dt * v_on_u(v) - g * dt * \
            deta_x(eta1)/dx - gamma * dt * u[:,1:-1] + t_x/(rho * H) * dt 
        v1[1:-1,:] = v[1:-1,:] - f_v * dt * u_on_v(u1) - g * dt * \
            deta_y(eta1)/dy - gamma * dt * v[1:-1,:] + tau_y(y_v)/ (rho * H) * dt
        
        # Second run
        eta2 = eta1 - H * dt * ((np.diff(u1,axis=1))/dx + (np.diff(v1,axis=0))/dy)
        v2[1:-1,:] = v1[1:-1,:] - f_v * dt * u_on_v(u1) - g * dt * \
            deta_y(eta2)/dy - gamma * dt * v1[1:-1,:] + tau_y(y_v) / \
                (rho * H) * dt
        u2[:,1:-1] = u1[:,1:-1] + f_u * dt * v_on_u(v2) - g * dt * \
            deta_x(eta2)/dx - gamma * dt * u1[:,1:-1] + t_x \
                / (rho * H) * dt
        
        # Update arrays
        eta = eta2.copy()
        u = u2.copy()
        v = v2.copy()
        
        # Energy
        E.append(energy(rho,H,u_on_eta(u),v_on_eta(v),g,eta,dx,dy))
        
    # Difference between analytical and numerical
    u_p = u - u_a
    v_p = v - v_a
    eta_p = eta - eta_a
    
    # Energy difference integration
    Ed = energy(rho,H,u_on_eta(u_p),v_on_eta(v_p),g,eta_p,dx,dy)
    
    if task == 'E2' or task == 'E3':
        print\
        ('Energy difference between the analytical and numerical solutions:', \
         "{:e}".format(Ed))
    
    if task == 'D1' or task == 'D2':
        # Plots
        xplt = (0.5 + np.arange(nx)) * dx
        yplt = (0.5 + np.arange(nx)) * dy
        X, Y = np.meshgrid(xplt,yplt)
        fig, [[ax, ax1],[ax2, ax3]] = plt.subplots(nrows=2, ncols=2,figsize=(15, 10))
        
        # Contour Plot
        eta_n_plot = plt.contourf(X,Y,eta)
        plt.colorbar(eta_n_plot).set_label('Height Anomaly (m)')
        ax3.set_xlabel('x (km)')
        ax3.set_ylabel('y (km)')
        
        # u velocity Plot
        ax1.plot((np.arange(nx+1) * dx),u[0,:])
        ax1.set_xlabel('x (km)')
        ax1.set_ylabel('Horizontal velocity (ms^-1)')
 
        # v velocity Plot
        ax2.plot((np.arange(nx+1) * dy),v[:,0])
        ax2.set_xlabel('y (km)')
        ax2.set_ylabel('Vertical velocity (ms^-1)')

        # height anomaly Plot
        ax.plot((np.arange(nx) * dx),eta[int(ny/2),:])
        ax.set_xlabel('x (km)')
        ax.set_ylabel('Height Anomaly (m)')
        
    elif task == 'D3':
        # Plots
        xplt = (0.5 + np.arange(nx)) * dx
        yplt = (0.5 + np.arange(nx)) * dy
        X, Y = np.meshgrid(xplt,yplt)
        fig, [[ax, ax1],[ax2, ax3]] = plt.subplots(nrows=2, ncols=2,figsize=(15, 10))
        
        # Contour Plot
        eta_n_plot = plt.contourf(X,Y,eta_p)
        plt.colorbar(eta_n_plot).set_label('Height Anomaly (m)')
        ax3.set_xlabel('x (km)')
        ax3.set_ylabel('y (km)')
        
        # u velocity Plot
        xplt_u = np.arange(nx+1) * dx
        ax1.plot((np.arange(nx+1) * dx),u_p[0,:])
        ax1.set_xlabel('x (km)')
        ax1.set_ylabel('Horizontal velocity (ms^-1)')
        
        # v velocity Plot
        ax2.plot((np.arange(nx+1) * dy),v_p[:,0])
        ax2.set_xlabel('y (km)')
        ax2.set_ylabel('Vertical velocity (ms^-1)')
        
        # height anomaly Plot
        ax.plot((np.arange(nx) * dx),eta_p[int(ny/2),:])
        ax.set_xlabel('x (km)')
        ax.set_ylabel('Height anomaly (m)')
        
    elif task == 'E1' or task == 'E2':
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        timeline = np.arange(nt/2) * 2 * dt * (1/86400)
        ax3.plot(timeline,E)
        ax3.set_xlabel('Time (days)')
        ax3.set_ylabel('Energy (PJ)')
        plt.title('Energy plot')
        
    elif task == 'C':
        # Plots
        xplt = (0.5 + np.arange(nx)) * dx
        yplt = (0.5 + np.arange(nx)) * dy
        X, Y = np.meshgrid(xplt,yplt)
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        eta_a_plot = ax5.contourf(X,Y,eta_a)
        plt.colorbar(eta_a_plot)
        ax5.set_xlabel('x (km)')
        ax5.set_ylabel('y (km)')
        plt.title('Analytical Eta contour plot')
    
def RK4():
    
    # Task setup
    nx = 40
    ny = 40
    dx = L/nx
    dy = L/ny
    dt = 200
    timed = 40*86400
    nt = int(timed/dt)
    
    # x and y arrays
    x = np.arange(nx+1) * dx
    y = np.arange(nx+1) * dy
    
    # u v and eta arrays
    u = np.zeros((ny,nx+1))
    v = np.zeros((ny+1,nx))
    eta = np.zeros((ny,nx))
    u1 = np.zeros((ny,nx+1))
    v1 = np.zeros((ny+1,nx))
    u2 = np.zeros((ny,nx+1))
    v2 = np.zeros((ny+1,nx))
    eta_k1 = np.zeros((ny,nx))
    eta_k2 = np.zeros((ny,nx))
    eta_k3 = np.zeros((ny,nx))
    eta_k4 = np.zeros((ny,nx))
    u_l1 = np.zeros((ny,nx+1))
    u_l2 = np.zeros((ny,nx+1))
    u_l3 = np.zeros((ny,nx+1))
    u_l4 = np.zeros((ny,nx+1))
    v_m1 = np.zeros((ny+1,nx))
    v_m2 = np.zeros((ny+1,nx))
    v_m3 = np.zeros((ny+1,nx))
    v_m4 = np.zeros((ny+1,nx))
    
    # Compute coriolis
    y_u = (0.5 + np.arange(ny)) * dy
    f_u = np.array(np.tile(coriolis_u(y_u),(nx-1,1))).T
    y_v = np.arange(1,ny) * dy
    f_v = np.array(np.tile(coriolis_u(y_v),(ny,1))).T
    
    # Compute tau
    t_x = np.array(np.tile(tau_x(tau_0,y_u,L),(nx-1,1))).T
    t_y = np.array(np.tile(tau_y(y_v),(ny,1))).T   
    
    
    for i in range(0,nt):
    
        # First run
        eta_k1 = eta_solver(H,dt,u,v,dx,dy)
        u_l1[:,1:-1] = u_vel_solver(f_u,dt,v,g,eta,gamma,u,t_x,rho,H,dx)
        v_m1[1:-1,:] = v_vel_solver(f_v,dt,u,g,eta,gamma,v,y_v,rho,H,dy)
         
        eta_k2 = eta_solver(H,dt,u+(u_l1*dt)/2,v+(v_m1*dt)/2,dx,dy)
        u_l2[:,1:-1] = u_vel_solver(f_u,dt,v+(v_m1*dt)/2,g,eta+(eta_k1*dt)/2,gamma,u+(u_l1*dt)/2,t_x,rho,H,dx)
        v_m2[1:-1,:] = v_vel_solver(f_v,dt,u+(u_l1*dt)/2,g,eta+(eta_k1*dt)/2,gamma,v+(v_m1*dt)/2,y_v,rho,H,dy)
        
        eta_k3 = eta_solver(H,dt,u+(u_l2*dt)/2,v+(v_m2*dt)/2,dx,dy)
        u_l3[:,1:-1] = u_vel_solver(f_u,dt,v+(v_m2*dt)/2,g,eta+(eta_k2*dt)/2,gamma,u+(u_l2*dt)/2,t_x,rho,H,dx)
        v_m3[1:-1,:] = v_vel_solver(f_v,dt,u+(u_l2*dt)/2,g,eta+(eta_k2*dt)/2,gamma,v+(v_m2*dt)/2,y_v,rho,H,dy)
         
        eta_k4 = eta_solver(H,dt,u+(u_l3*dt),v+(v_m3*dt),dx,dy)
        u_l4[:,1:-1] = u_vel_solver(f_u,dt,v+(v_m3*dt),g,eta+(eta_k3*dt),gamma,u+(u_l3*dt),t_x,rho,H,dx)
        v_m4[1:-1,:] = v_vel_solver(f_v,dt,u+(u_l3*dt),g,eta+(eta_k3*dt),gamma,v+(v_m3*dt),y_v,rho,H,dy)        
        
        eta = eta + dt * (1/6) * (eta_k1 + 2 * eta_k2 + 2 * eta_k3 + eta_k4)
        u = u + dt * (1/6) * (u_l1 + 2 * u_l2 + 2 * u_l3 + u_l4)
        v = v + dt * (1/6) * (v_m1 + 2 * v_m2 + 2 * v_m3 + v_m4)
        

    # Plots
    xplt = (0.5 + np.arange(nx)) * dx
    yplt = (0.5 + np.arange(nx)) * dy
    X, Y = np.meshgrid(xplt,yplt)
    fig, [[ax, ax1],[ax2, ax3]] = plt.subplots(nrows=2, ncols=2,figsize=(15, 10))
    
    # Contour Plot
    eta_n_plot = plt.contourf(X,Y,eta)
    plt.colorbar(eta_n_plot).set_label('Height Anomaly (m)')
    ax3.set_xlabel('x (km)')
    ax3.set_ylabel('y (km)')
    
    # u velocity Plot
    ax1.plot((np.arange(nx+1) * dx),u[0,:])
    ax1.set_xlabel('x (km)')
    ax1.set_ylabel('Horizontal velocity (ms^-1)')

    # v velocity Plot
    ax2.plot((np.arange(nx+1) * dy),v[:,0])
    ax2.set_xlabel('y (km)')
    ax2.set_ylabel('Vertical velocity (ms^-1)')

    # height anomaly Plot
    ax.plot((np.arange(nx) * dx),eta[int(ny/2),:])
    ax.set_xlabel('x (km)')
    ax.set_ylabel('Height Anomaly (m)')
   