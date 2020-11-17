# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 19:05:23 2020

@author: bourl
"""
from main import modelfun, setupFiniteVolumeMesh, computeT, computeP, getXFromVars, getVarsFromX, getVarsFromX_vectorized, computeOtherVariables
from main import cv ,r, gamma
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy.optimize


options={'mesh':{}, 'BCs':{'up_down':None, 'left_right':None}}

xmin,xmax = -0.25,0.25
ymin,ymax = -0.75,0.75
nx,ny = 200,600

xfaces = np.linspace(xmin,xmax,nx)
yfaces = np.linspace(ymin,ymax,ny)
options['mesh'] = setupFiniteVolumeMesh(xfaces, yfaces)    
xcells = options['mesh']['cells']['x']
ycells = options['mesh']['cells']['y']

#Config Y
yc = 0.
iselec = ycells < yc
not_iselec = np.logical_not(iselec)
options["BCs"]["up_down"]="reflective"
options["BCs"]["left_right"]="periodic"

g = np.zeros((2,ny-1,nx-1))
g[1,:,:] = -9.81
options["g"] = g

# Sod shock tube (invariant along y)
rho_0 = np.zeros_like(xcells)
rho_0[iselec]  =  1.0
rho_0[not_iselec] =  2.0

u_0 = np.zeros_like(xcells)
u_0[iselec]  =  0.
u_0[not_iselec] =  0.

v_0 = np.zeros_like(xcells)
v_0 =  0.01*(1+np.cos(2*np.pi*xcells/(xmax-xmin)))*(1+np.cos(2*np.pi*ycells/(ymax-ymin)))/4.

P_0 = 2.5 * rho_0 * (options['g'][0,:,:]*xcells + options['g'][1,:,:]*ycells)
P_0 = P_0 + 3*np.max(np.abs(P_0))
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

#%% Create a method for computing the Jacobian in an optimised manner, exploiting its sparsity pattern
import scipy.sparse
import scipy.optimize._numdiff
uband=8; lband=-uband
offsets = [i for i in range(lband,uband)] + list(range(28,40)) + list(range(-28,-40,-1))
sparsity_pattern = scipy.sparse.diags(diagonals=[np.ones((4*nx*ny - abs(i))) for i in offsets], offsets=offsets) 

jacfun = lambda t,x: scipy.optimize._numdiff.approx_derivative(
                    fun=lambda y: modelfun(t=t,x=y,options=options),
                    x0=x, method='2-point', sparsity=sparsity_pattern,
                    rel_step=1e-8, abs_step=1e-8)

if 0: # test correctness of the sparse Jacobian against a naive dense estimation
  jacfun_full = lambda t,x: scipy.optimize._numdiff.approx_derivative(
                      fun=lambda y: modelfun(t=t,x=y,options=options),
                      x0=x, method='2-point', sparsity=None,
                      rel_step=1e-8, abs_step=1e-8)
  
  xtest = X0 + np.random.rand(X0.size).reshape(X0.shape)*(1e-3 + 1e-3*X0)
  
  jac_full = jacfun_full(0.,xtest)
  jac_sparse = jacfun(0.,xtest)

  plt.figure()
  plt.spy(jac_sparse)
  plt.spy(jac_full, marker='+')
  assert np.max(np.abs(jac_sparse-jac_full)) < 1e-12, 'The sparse Jacobian estimation is not correct'


#%% NUMERICAL INTEGRATION
# out  = integrate.solve_ivp(fun=lambda t,x: modelfun(t,x,options), t_span=(0.,tend), y0=X0, first_step=1e-7,
#                        # max_step=np.inf, method='RK45', atol=1e-3, rtol=1e-4)
#                        # max_step=np.inf, method='Radau', atol=1e-2, rtol=1e-2, jac=jacfun,
#                        max_step=np.inf, method='DOP853', atol=1e-5, rtol=1e-5, jac=jacfun,
#                        t_eval = np.arange(0,tend,1e-2))
# backupdict = {"t": out.t, "y": out.y, "options": options}
# with open('backup.json', 'w') as f:
#     json.dump(backupdict, f, cls=NumpyEncoder)
if 1:
  tend=  0.1 + 1e-6
  tmin = 0.
  dt_save = 1e-2
  outs=[]
  import copy
  for i, tmax in enumerate(np.arange(dt_save,tend,dt_save)):
      out = integrate.solve_ivp(fun=lambda t,x: modelfun(t,x,options), t_span=(tmin,tmax),
                                       y0=X0, first_step=1e-7,
                       # max_step=np.inf, method='RK45', atol=1e-3, rtol=1e-4,
                       # max_step=np.inf, method='Radau', atol=1e-2, rtol=1e-2, jac=jacfun,
                       max_step=np.inf, method='DOP853', atol=1e-5, rtol=1e-5, jac=jacfun,
                       t_eval = np.linspace(tmin,tmax,2))
      tmin = tmax
      outs.append(copy.deepcopy(out))
  
      # backup
      import json
      from utilities import NumpyEncoder
      backupdict = {"t": out.t[-1], "y": out.y[:,-1]}
      with open('backups/v2_200x600_backup_state_{}.json'.format(i), 'w') as f:
          json.dump(backupdict, f, cls=NumpyEncoder)
  # final backup
  import json
  from utilities import NumpyEncoder
  backupdict = {"t": out.t[-1], "y": out.y[:,-1], "options": options}
  with open('backup_final_state.json', 'w') as f:
      json.dump(backupdict, f, cls=NumpyEncoder)
#%%
else:
  #%% load backups
  filelist = ["/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_0.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_1.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_2.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_3.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_4.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_5.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_6.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_7.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_8.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_9.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_10.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_11.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_12.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_13.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_14.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_15.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_16.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_17.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_18.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_19.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_20.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_21.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_22.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_23.json",
              "/home/laurent/These/GIT/CFD2D/backups/200x600_backup_state_24.json"]
  
  import scipy.integrate
  import copy
  out  = scipy.integrate._ivp.ivp.OdeResult(t=[], y=[]) #{"t":[], "y":[]}
  from utilities import JSONtoNumpy
  import json
  for file in filelist:
    print(file)
    with open(file,'r') as f:
      temp = JSONtoNumpy( json.load(f) )
    out.t.append(copy.deepcopy(temp['t']))
    out.y.append(copy.deepcopy(temp['y']))
  out.t = np.array(out.t)
  out.y = np.array(out.y)
  

    
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
    for j in range(0,nx-1,10):
         plt.plot(ycells[:,j], var[:,j,-1], label='num', marker=None, linestyle='-', linewidth = 0.2+ 5/(j+1))
    #for j in range(ny-1):
    #    plt.plot(xcells[j,:], var[j,:,-1], label='num', marker=None, linestyle='-', linewidth = 0.2+ 5/(j+1))
    plt.xlabel('position')
    plt.ylabel(varname)
    plt.title(varname)
    

#%%
xx,yy = options['mesh']['cells']['x'], options['mesh']['cells']['y']
space_gap = 1

# compute nmerical schlieren
# schlieren_x = np.gradient((P/rho), axis=1).T/np.gradient(xx, axis=1)
# schlieren_y = np.gradient((P/rho), axis=0).T/np.gradient(yy, axis=0)
# schlieren = (schlieren_x**2 + schlieren_y**2)**0.5

schlieren_x = np.zeros_like(P)
schlieren_y = np.zeros_like(P)
for it in range(time.size):
  schlieren_x[:,:,it] = np.gradient((P[:,:,it]/rho[:,:,it]), axis=1) / np.gradient(xx, axis=1)
  schlieren_y[:,:,it] = np.gradient((P[:,:,it]/rho[:,:,it]), axis=0) / np.gradient(yy, axis=0)
schlieren = (schlieren_x**2 + schlieren_y**2)**0.5

Plog10 = np.log10(P)
#%%
varplot = (
            # (np.log10(np.abs(schlieren)), 'log10_schlieren', 'Synthetic Schlieren', None, (-10, np.max(np.log10(np.abs(schlieren))))),
            # ((np.abs(schlieren)), 'schlieren', 'Synthetic Schlieren', None,  (np.min(np.abs(schlieren)), np.max(np.abs(schlieren)))),
            # (P, 'P', 'Pressure field', 'P (Pa)', (np.min(P), np.max(P))),            
            # (rho, 'rho', 'Density field', r'$\rho$ (kg/m$^{-3}$)', (np.min(rho), np.max(rho))),
            (rho, 'rho', 'Density field', r'$\rho$ (kg/m$^{-3}$)',(None,None)),

            # (Plog10, 'P', 'Pressure field', 'P (Pa)', (np.min(Plog10), np.max(Plog10))),
           )
for var, name, title, clabel, (zmin,zmax) in varplot:
    if zmin is not None:
      levels = np.linspace(zmin, zmax, 100)
    else:
      levels = 100 # auto-levels
    for wished_t in np.linspace(time[0], time[-1], 5): # plot the solution at regular time intnervals
      plt.figure(dpi=300)
      itime = np.argmin( np.abs(wished_t-time) ) # closest simulation time available
      plt.contourf(xx[::space_gap,::space_gap], yy[::space_gap,::space_gap], var[::space_gap,::space_gap, itime],
                   levels=levels, cmap='rainbow')
      plt.xlabel(r'$x$ (m)')
      plt.ylabel(r'$t$ (s)')
      cb = plt.colorbar()
      cb.set_label(clabel)
      # plt.grid()
      plt.axis('equal')
      plt.title('t = {}'.format(wished_t))
      plt.savefig('{}_{}.png'.format(name,itime), dpi=500)
      plt.show()
      
#export list_img=$(ls | sort -V)
#convert -delay 10 $list_img animation2.gif
      
