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
import scipy.integrate
import copy
from utilities import JSONtoNumpy
import json
from utilities import NumpyEncoder


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

if 0:
    X0 = getXFromVars(rho_0, rho_0*u_0, rho_0*v_0, rho_0*E_0)
    tmin = 0.
    i_start = 0
else:
    print('loading initial solution from JSON file')
    inifile = 'D:/GIT/CFD2D/backups/200x600_backup_state_20.json'
    with open(inifile,'r') as f:
      bacdict = JSONtoNumpy( json.load(f) )
    tmin = bacdict['t']
    X0   = bacdict['y']
    i_start = 1+int(inifile.split('_')[-1].split('.')[0])
    
surfaces = options['mesh']['cells']['surfaces']
nx = options['mesh']['cells']['nx']
ny = options['mesh']['cells']['ny']

##### recover conserved variables
# rho, rhoU, rhoV, rhoE = getVarsFromX(X0, options)
# temp = computeOtherVariables(rho, rhoU, rhoV, rhoE)
# u = temp['u']
# v = temp['v']
# P = temp['P']

#%%
# dxdt0 = modelfun(t=0.,x=X0, options=options)
# dtrho0, dtrhoU0, dtrhoV0, dtrhoE0 = getVarsFromX(dxdt0, options)

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
if 0: # perform unsteady simulation
  tend=  5.0
  dt_save = 1/24
  outs=[]
  for ii, tmax in enumerate(np.arange(tmin+dt_save,tend,dt_save)):
      i = ii + i_start
      if 0: # implicit
          # linearised implicit schemes (IE at first)
          out = integrate.solve_ivp(fun=lambda t,x: modelfun(t,x,options), t_span=(tmin,tmax),
                                           y0=X0, first_step=1e-7,
                           max_step=np.inf, method='Radau', atol=1e-5, rtol=1e-5, jac=jacfun,
                           t_eval = np.linspace(tmin,tmax,10))
      else: # explicit
          out = integrate.solve_ivp(fun=lambda t,x: modelfun(t,x,options), t_span=(tmin,tmax),
                                       y0=X0, first_step=1e-7,
                       # max_step=np.inf, method='RK45', atol=1e-3, rtol=1e-4,
                       # max_step=np.inf, method='Radau', atol=1e-2, rtol=1e-2, jac=jacfun,
                       max_step=np.inf, method='DOP853', atol=1e-5, rtol=1e-5, jac=jacfun,
                       t_eval = np.linspace(tmin,tmax,10))
      raise Exception('done')
      
      tmin = tmax
      outs.append(copy.deepcopy(out))
      X0 = np.copy(out.y[:,-1])
  
      # backup
      print('intermediate back up')
      backupdict = {"t": out.t[-1], "y": out.y[:,-1]}
      with open('backups/v4_200x600_backup_state_{}.json'.format(i), 'w') as f:
          json.dump(backupdict, f, cls=NumpyEncoder)
          
      # plot progress
      rho, rhoU, rhoV, rhoE = getVarsFromX_vectorized(out.y[:,-1], options)
      xx,yy = options['mesh']['cells']['x'], options['mesh']['cells']['y']

      plt.figure(dpi=200)
      space_gap = 5
      plt.contourf(xx[::space_gap,::space_gap],
                   yy[::space_gap,::space_gap],
                   rho[::space_gap,::space_gap,0],
                   levels=100, cmap='rainbow')
      plt.xlabel(r'$x$ (m)')
      plt.ylabel(r'$t$ (s)')
      cb = plt.colorbar()
      cb.set_label(r'$\rho$ (kg.m^${-3}$)')
      # plt.grid()
      plt.axis('equal')
      plt.title('t = {}'.format(out.t[-1]))
      # plt.savefig('{}_{}.png'.format(name,itime), dpi=500)
      plt.show()
      
  # final backup
  import json
  from utilities import NumpyEncoder
  backupdict = {"t": out.t[-1], "y": out.y[:,-1], "options": options}
  with open('backup_final_state.json', 'w') as f:
      json.dump(backupdict, f, cls=NumpyEncoder)
      
  # concatenate solutions
  time = np.hstack([o.t for o in outs])
  y = np.hstack([o.y for o in outs])
  out = scipy.integrate._ivp.ivp.OdeResult(t=time, y=y)
  nfev =   [o.nfev for o in outs]
  sum(nfev)  
  
  # np.save('200x600_y_t0_t4s.npy', y, allow_pickle=False, fix_imports=False)
  # np.save('200x600_t_t0_t4s.npy', time, allow_pickle=False, fix_imports=False) 
  
  # y= np.load('200x600_y_t0_t4s.npy')
  # time= np.load('200x600_t_t0_t4s.npy')
  # out = scipy.integrate._ivp.ivp.OdeResult(t=time, y=y)

#%%
else:
  #%% load backups
  filelist = [
"D:/GIT/CFD2D/backups/v3_200x600_backup_state_0.json",
"D:/GIT/CFD2D/backups/v3_200x600_backup_state_1.json",
"D:/GIT/CFD2D/backups/v3_200x600_backup_state_2.json",
"D:/GIT/CFD2D/backups/v3_200x600_backup_state_3.json",
"D:/GIT/CFD2D/backups/v3_200x600_backup_state_4.json",
"D:/GIT/CFD2D/backups/v3_200x600_backup_state_5.json",
"D:/GIT/CFD2D/backups/v3_200x600_backup_state_6.json",
"D:/GIT/CFD2D/backups/v3_200x600_backup_state_7.json",
"D:/GIT/CFD2D/backups/v3_200x600_backup_state_8.json",
"D:/GIT/CFD2D/backups/v3_200x600_backup_state_9.json",
"D:/GIT/CFD2D/backups/v3_200x600_backup_state_10.json",
"D:/GIT/CFD2D/backups/v3_200x600_backup_state_11.json",
"D:/GIT/CFD2D/backups/v3_200x600_backup_state_12.json",
"D:/GIT/CFD2D/backups/200x600_backup_state_13.json",
"D:/GIT/CFD2D/backups/200x600_backup_state_14.json",
"D:/GIT/CFD2D/backups/200x600_backup_state_15.json",
"D:/GIT/CFD2D/backups/200x600_backup_state_16.json",
"D:/GIT/CFD2D/backups/200x600_backup_state_17.json",
"D:/GIT/CFD2D/backups/200x600_backup_state_18.json",
"D:/GIT/CFD2D/backups/200x600_backup_state_19.json",
"D:/GIT/CFD2D/backups/200x600_backup_state_20.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_21.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_22.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_23.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_24.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_25.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_26.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_27.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_28.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_29.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_30.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_31.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_32.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_33.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_34.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_35.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_36.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_37.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_38.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_39.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_40.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_41.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_42.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_43.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_44.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_45.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_46.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_47.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_48.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_49.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_50.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_51.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_52.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_53.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_54.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_55.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_56.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_57.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_58.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_59.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_60.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_61.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_62.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_63.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_64.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_65.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_66.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_67.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_68.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_69.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_70.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_71.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_72.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_73.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_74.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_75.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_76.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_77.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_78.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_79.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_80.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_81.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_82.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_83.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_84.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_85.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_86.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_87.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_88.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_89.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_90.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_91.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_92.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_93.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_94.json",
"D:/GIT/CFD2D/backups/v4_200x600_backup_state_95.json",
              ]
  out  = scipy.integrate._ivp.ivp.OdeResult(t=[], y=[]) #{"t":[], "y":[]}
  
  for file in filelist:
    print(file)
    with open(file,'r') as f:
      bacdict = JSONtoNumpy( json.load(f) )
    out.t.append(copy.deepcopy(bacdict['t']))
    out.y.append(copy.deepcopy(bacdict['y']))
  out.t = np.array(out.t)
  out.y = np.vstack(out.y).T
  

    
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
if 0:
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
from matplotlib import ticker
space_gap = 1
nlevels = 400
dpi = 300
varplot = (
            # (np.log10(np.abs(schlieren)), 'log10_schlieren', 'Synthetic Schlieren', None, (-10, np.max(np.log10(np.abs(schlieren)))), 'Greys'),
            # ((np.abs(schlieren)), 'schlieren', 'Synthetic Schlieren', None,  (np.min(np.abs(schlieren)), np.max(np.abs(schlieren))), 'Greys'),
            (P, 'P', 'Pressure field', 'P (Pa)', (np.min(P), np.max(P)), None),
            # (rho, 'rho', 'Density field', r'$\rho$ (kg.m$^{-3}$)', (np.min(rho), np.max(rho)), None),
            # (T, 'T', 'Temperature field', r'$T$ (K)', (np.min(T), np.max(T)), None),
            # (rhoE, 'rhoE', 'Total energy field', r'$\rho E$ (J.m^{-3})', (np.min(rhoE), np.max(rhoE)), None),
            # (rho, 'rho', 'Density field', r'$\rho$ (kg/m$^{-3}$)',(None,None), None),
            # (Plog10, 'P', 'Pressure field', 'P (Pa)', (np.min(Plog10), np.max(Plog10)), None),
           )
for ivar, (var, name, title, clabel, (zmin,zmax), cmap) in enumerate(varplot):
    if zmin is not None:
      levels = np.linspace(zmin, zmax, nlevels)
    else:
      levels = nlevels # auto-levels
    if cmap is None:
      cmap='rainbow'
    for it, wished_t in enumerate(time):# np.linspace(time[0], time[-1], 5): # plot the solution at regular time intnervals
      print('instant {}/{} for var {}/{}'.format(it, len(time), ivar, len(varplot)))
      plt.figure(dpi=dpi, figsize=(10*(xmin-xmax)/(ymin-ymax),10))
      itime = np.argmin( np.abs(wished_t-time) ) # closest simulation time available
      plt.contourf(xx[::space_gap,::space_gap], yy[::space_gap,::space_gap], var[::space_gap,::space_gap, itime],
                   levels=levels, cmap=cmap)
      plt.xlabel(r'$x$ (m)')
      plt.ylabel(r'$y$ (m)')
      if clabel is not None:
          cb = plt.colorbar(shrink=0.5)
          cb.set_label(clabel)
          tick_locator = ticker.MaxNLocator(nbins=5)
          cb.locator = tick_locator
          cb.update_ticks()

      # plt.grid()
      # plt.ylim(0.98*ymin, 0.98*ymax)
      # plt.xlim(xmin,xmax)
      plt.axis('equal')
      plt.gca().axis('scaled')
      plt.title('{}\nt = {:.3f} s'.format(title, time[itime]))
      plt.savefig('plots/{}_{:04d}.png'.format(name,itime), dpi=dpi)
      plt.show()
      
#export list_img=$(ls | sort -V)
#convert -delay 10 $list_img animation2.gif
     
#%% Quiver plot for velocity field
space_gap = 7
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.colors as colors
cmap = plt.get_cmap('YlOrRd')

norm = Normalize()
normV = (u**2 + v**2)**0.5
maxV = np.max(normV)
norm.autoscale(normV)

# levels_normV = np.linspace(0,np.ceil(np.max(normV)),100)
levels_normV = np.linspace(0,np.max(normV),100)
clabel=r'$||v||$ (m.s$^{-1}$)'
for it, wished_t in enumerate(time): # plot the solution at regular time intervals
    itime = np.argmin( np.abs(wished_t-time) ) # closest simulation time available
    print('instant {}/{} for var {}/{}'.format(it, len(time), ivar, len(varplot)))
    
    fig1, ax1 = plt.subplots(dpi=dpi, figsize=(10*(xmin-xmax)/(ymin-ymax),10))
    ax1.set_title('Velocity field')
    plt.contourf(xx[::space_gap,::space_gap], yy[::space_gap,::space_gap], normV[::space_gap,::space_gap, itime],
                   levels=levels_normV, cmap=cmap)
    # set colors for the speed norm, realtive the maximum speed across all time steps
    # Q = ax1.quiver(xx[::space_gap,::space_gap], yy[::space_gap,::space_gap],
    #                u[::space_gap,::space_gap,itime], v[::space_gap,::space_gap,itime],
    #                colormap(norm( normV[::space_gap,::space_gap, itime] )),#.reshape(-1,4),
    #                units='width')
    
    # equal length for all arrows
    Q = ax1.quiver(xx[::space_gap,::space_gap], yy[::space_gap,::space_gap],
               u[::space_gap,::space_gap,itime]/normV[::space_gap,::space_gap,itime],
               v[::space_gap,::space_gap,itime]/normV[::space_gap,::space_gap,itime],
               units='width', scale=30)
               
    # length of arrow normalized against all time steps
    # Q = ax1.quiver(xx[::space_gap,::space_gap], yy[::space_gap,::space_gap],
    #                u[::space_gap,::space_gap,itime]*normV[::space_gap,::space_gap,itime]/maxV,
    #                v[::space_gap,::space_gap,itime]*normV[::space_gap,::space_gap,itime]/maxV,
    #                units='width', scale=30)
    # qk = ax1.quiverkey(Q, 0.9, 0.5, 2, r'$1 \frac{m}{s}$', labelpos='E',
    #                    coordinates='figure')
    if clabel is not None:
        cb = plt.colorbar(shrink=0.5)
        cb.set_label(clabel)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
        
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$y$ (m)')
    plt.axis('equal')
    plt.gca().axis('scaled')
    plt.title('Velocity field\nt = {:.3f} s'.format(time[itime]))
    plt.savefig('plots/isolength_{}_{:04d}.png'.format('velocity',itime), dpi=dpi)
    plt.show()