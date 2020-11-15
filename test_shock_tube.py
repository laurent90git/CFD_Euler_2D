#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:45:27 2020

@author: laurent
"""
from main import modelfun, setupFiniteVolumeMesh, computeT, computeP, getXFromVars, getVarsFromX, getVarsFromX_vectorized, computeOtherVariables
from main import cv ,r, gamma
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy.optimize

#%%
options={'mesh':{}, 'BCs':{'up_down':None, 'left_right':None}}

xmin,xmax = 0,10
ymin,ymax = 0,10

# Config X
# nx,ny = 100,10

#Config Y
nx,ny = 10,100

xfaces = np.linspace(xmin,xmax,nx)
yfaces = np.linspace(ymin,ymax,ny)
options['mesh'] = setupFiniteVolumeMesh(xfaces, yfaces)    
xcells = options['mesh']['cells']['x']
ycells = options['mesh']['cells']['y']

# Config X
#xc = xmax/2.; iselec= xcells<xc; options["BCs"]["up_down"]="periodic"; options["BCs"]["left_right"]="transmissive"

#Config Y
xc = ymax/2.; iselec= ycells<xc; options["BCs"]["up_down"]="transmissive"; options["BCs"]["left_right"]="periodic"



# Sod shock tube (invariant along y)


P_0 = np.zeros_like(xcells)



# xc1,xc2=0.4,0.6
#iselec = np.logical_and(xcells>xc1, xcells<xc2)
not_iselec = np.logical_not(iselec)
P_0[iselec]  =  1.*1e5
P_0[not_iselec] =  0.1*1e5

rho_0 = np.zeros_like(xcells)
rho_0[iselec]  =  1.0
rho_0[not_iselec] =  0.125

u_0 = np.zeros_like(xcells)
u_0[iselec]  =  0.
u_0[not_iselec] =  0.

v_0 = 0.*u_0

T_0 = computeT(P_0, rho_0)
E_0 = cv*T_0 + 0.5*u_0*u_0


X0 = getXFromVars(rho_0, rho_0*u_0, rho_0*v_0, rho_0*E_0)
rho, rhoU, rhoV, rhoE = getVarsFromX(X0, options)

surfaces = options['mesh']['cells']['surfaces']
nx = options['mesh']['cells']['nx']
ny = options['mesh']['cells']['ny']

##### recover conserved variables
rho, rhoU, rhoV, rhoE = getVarsFromX(X0, options)
temp = computeOtherVariables(rho, rhoU, rhoV, rhoE)
u = temp['u']
v = temp['v']
P = temp['P']

#%%
dxdt0 = modelfun(t=0.,x=X0, options=options)
dtrho0, dtrhoU0, dtrhoV0, dtrhoE0 = getVarsFromX(dxdt0, options)

#%% NUMERICAL INTEGRATION
tend= 0.0005
out  = integrate.solve_ivp(fun=lambda t,x: modelfun(t,x,options), t_span=(0.,tend), y0=X0, first_step=1e-9,
                           max_step=np.inf, method='LSODA', atol=1e-10, rtol=1e-10, band=(-6,6))
                           # max_step=np.inf, method='BDF', atol=1e-9, rtol=1e-9)
#%% GATHER RESULTS
rho, rhoU, rhoV, rhoE = getVarsFromX_vectorized(out.y, options)
temp = computeOtherVariables(rho, rhoU, rhoV, rhoE)
u,v,T,P = temp['u'], temp['v'], temp['T'], temp['P']
time = out.t
 
#%% JACOBIAN ANALYSIS
if 0:
    #%%
    testfun= lambda x: modelfun(0., x, options)
    # def testfun(x):
    #     rho, rhoU, rhoV, rhoE = getVarsFromX(x, options)
    #     return rho.flatten()
    import scipy.optimize._numdiff
    Xtest = X0 + np.random.rand(X0.size).reshape(X0.shape)*(1e-3 + 1e-3*X0)
    Jac = scipy.optimize._numdiff.approx_derivative(
                                fun=testfun,
                                x0=Xtest, method='2-point',
                                rel_step=1e-8)
    plt.figure()
    plt.spy(Jac)
    n_rank_jac = np.linalg.matrix_rank(Jac),
    plt.title('Jacobian (rank={}, shape={})'.format(n_rank_jac, np.shape(Jac)))
    plt.show()
    if n_rank_jac[0]!=np.size(Jac,1):
        print('The following rows of the Jacobian are nil:\n\t{}'.format( np.where( (Jac==0).all(axis=1) ) ))
        print('The following columns of the Jacobian are nil:\n\t{}'.format( np.where( (Jac==0).all(axis=0) ) ))
    if 0:#np.size(Jac,1)<500:
        try:
            eigvals, eigvecs= np.linalg.eig(Jac)
            plt.figure()
            plt.scatter(np.real(eigvals), np.imag(eigvals))
            plt.title('Eigenvalues')
        except Exception as e:
            print('caught exception "{}" while computing eigenvalues of the Jacobian'.format(e))
    else:
        print('Skipping eigenvalues computation due to matrix size')

#%% comparaison avec analytical solution
# mesh=options['mesh']['cellX']
# mesh_exact = np.linspace(np.min(mesh), np.max(mesh),int(2e2))
# exactsol = Riemann_exact(t=time[-1], g=gamma,
#                          Wl=np.array([rho[0,0], u[0,0], P[0,0]]),
#                          Wr=np.array([rho[-1,0], u[-1,0], P[-1,0]]),
#                          grid=mesh_exact)
# rho_exact = exactsol[0]
# u_exact = exactsol[1]
# P_exact = exactsol[2]
# T_exact = P_exact/rho_exact/r


# plt.figure()
# plt.plot(mesh_exact, rho_exact, color='r', label='exact')
# plt.plot(mesh, rho[:,-1], color='b', label='num', marker='+', linestyle='')
# plt.xlabel('position')
# plt.ylabel(r'$\rho$')
# plt.title('Densité')

# plt.figure()
# plt.plot(mesh_exact, u_exact, color='r', label='exact')
# plt.plot(mesh, u[:,-1], color='b', label='num', marker='+', linestyle='')
# plt.xlabel('position')
# plt.ylabel(r'$u$')
# plt.title('Vitesse')

# plt.figure()
# plt.plot(mesh_exact, P_exact, color='r', label='exact')
# plt.plot(mesh, P[:,-1], color='b', label='num', marker='+', linestyle='')
# plt.xlabel('position')
# plt.ylabel('P')
# plt.title('Pression')

#%%
varplot = ((rho, 'rho'), (P, 'P'), (T,'T'))
for var, varname in varplot:
    plt.figure(dpi=300)
    for j in range(nx-1):
         plt.plot(ycells[:,j], var[:,j,-1], label='num', marker=None, linestyle='-', linewidth = 0.2+ 5/(j+1))
    #for j in range(ny-1):
    #    plt.plot(xcells[j,:], var[j,:,-1], label='num', marker=None, linestyle='-', linewidth = 0.2+ 5/(j+1))
    plt.xlabel('position')
    plt.ylabel(varname)
    plt.title(varname)
