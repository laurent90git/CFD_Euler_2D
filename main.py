# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:56:42 2020

@author: bourl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:23:34 2019

@author: lfrancoi
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy.optimize

nIter=0

gamma= 1.4
R = 8.314
M = 28.967e-3 # masse molaire de l'air
r = R/M
cv = r/(gamma-1) # capacité thermique à volume constant

def computeT(P,rho):
    return P/(rho*r)

def computeP(rho,T):
    return rho*r*T

def computeRho(P,T):
    return P/(r*T)

def computeOtherVariables(rho, rhoU, rhoV, rhoE):
    u = rhoU/rho
    v = rhoV/rho
    # E = cv*T + 0.5*u^2
    E = rhoE/rho
    v2 = u*u + v*v
    T = (E - 0.5*v2)/cv
    a = np.sqrt(gamma*r*T)
    P = computeP(rho, T)
    H = 0.5*v2 + a*a/(gamma-1) #E + P/rho
    M = (v2**0.5)/a
    return {'u':u,'v':v, 'T':T, 'P':P, 'H':H, 'E':E, 'a':a, 'M':M}

def fluxEulerPhysique(W, direction):
    """ Flux physique Euler selon x (direction=1) ou y (direction=0)"""
    if len(W.shape)<2:
        W = W.reshape(W.shape[0],1)
        bReshaped=True
    else:
        bReshaped=False
    rho = W[0,:]
    rhoU = W[1,:]
    rhoV = W[2,:]
    rhoE = W[3,:]

    out = computeOtherVariables(rho, rhoU, rhoV, rhoE)
    u,v,P = out['u'], out['v'], out['P']

    F = np.zeros_like(W)
    if direction==0: # flux pour une face perpendiculaire à x (donc verticale)
      F[0,:] = rhoU
      F[1,:] = rhoU*u + P
      F[2,:] = rhoU*v
      F[3,:] = (rhoE + P)*u
    elif direction==1:
      F[0,:] = rhoV
      F[1,:] = rhoV*u
      F[2,:] = rhoV*v + P
      F[3,:] = (rhoE + P)*v
    else:
      raise Exception('direction can only be 0 (y) or 1 (x)')
    if bReshaped:
        return F.reshape((F.size,))
    else:
        return F


def HLLCsolver(WL,WR,options):
    #TODO: extend to 2D, see Toro page 327
    # --> consider the direction perpendicular to the face
    # --> this direction is the "x"-drection (seep x-split 3d euler euqqations in Tor, for example)
    # the fluxes are the same as in 1D, with the addtiion of two trivial passive transport fluxes
    # for the velocity tangent to the face.
    
    if len(WR.shape)<2:
        WR = WR[:,np.newaxis]
        WL = WL[:,np.newaxis]
        bReshaped=True
    else:
        bReshaped=False
    # 1 - compute physical variables
    rhoL, rhoUL, rhoVL, rhoEL = WL[0,:], WL[1,:], WL[2,:], WR[3,:]
    rhoR, rhoUR, rhoVR, rhoER = WR[0,:], WR[1,:], WR[2,:], WR[3,:]
    out = computeOtherVariables(rhoR, rhoUR, rhoVR, rhoER)
    uR,vR,PR,ER,HR,aR = out['u'], out['v'], out['P'], out['E'], out['H'], out['a']
    out = computeOtherVariables(rhoL, rhoUL, rhoVL, rhoEL)
    uL,vL,PL,EL,HL,aL = out['u'], out['v'], out['P'], out['E'], out['H'], out['a']


    # compute fluxes
    face_flux = np.zeros_like(WL)
    # vectorized mode
    # estimate the wave speeds
    if 0: #based on Roe-average
        utilde = (np.sqrt(rhoL)*uL + np.sqrt(rhoR)*uR)/(np.sqrt(rhoL) + np.sqrt(rhoR))
        Htilde = (np.sqrt(rhoL)*HL + np.sqrt(rhoR)*HR)/(np.sqrt(rhoL) + np.sqrt(rhoR))
        atilde = np.sqrt( (gamma-1)*(Htilde-0.5*utilde*utilde) )
        SL = utilde-atilde
        SR = utilde+atilde
    else:
        SL = np.minimum(uL-aL, uR-aR)
        SR = np.minimum(uL+aL, uR+aR)
    # compute Sstar
    Sstar = ( PR-PL + rhoL*uL*(SL-uL) - rhoR*uR*(SR-uR) ) / ( rhoL*(SL-uL) - rhoR*(SR-uR) )
    Wstar_L = np.zeros_like(WL)
    coeff = rhoL*(SL-uL)/(SL - Sstar)
    Wstar_L[0,:] = coeff
    Wstar_L[1,:] = coeff * Sstar
    Wstar_L[2,:] = coeff * vL
    Wstar_L[3,:] = coeff * ( EL+ (Sstar-uL)*(Sstar + PL/(rhoL*(SL-uL))) )

    Wstar_R = np.zeros_like(WL)
    coeff = rhoR*(SR-uR)/(SR - Sstar)
    Wstar_R[0,:] = coeff
    Wstar_R[1,:] = coeff*Sstar
    Wstar_R[2,:] = coeff*vR
    Wstar_R[3,:] = coeff*( ER+ (Sstar-uR)*(Sstar + PR/(rhoR*(SR-uR))) )

    total=0
    I=np.where(SL>0)
    face_flux[:,I] = fluxEulerPhysique(WL[:,I],direction=0)
    total = total + np.size(I)

    I=np.where((SL<=0) & (Sstar>=0))
    face_flux[:,I] = fluxEulerPhysique(Wstar_L[:,I],direction=0)
    total = total + np.size(I)

    I=np.where((SR>0) & (Sstar<0))
    face_flux[:,I] = fluxEulerPhysique(Wstar_R[:,I],direction=0)
    total = total + np.size(I)

    I = np.where(SR<=0)
    face_flux[:,I] = fluxEulerPhysique(WR[:,I],direction=0)
    total = total + np.size(I)
    if total != SR.size:
#    if np.isnan(SR+SL+Sstar).any():
        raise Exception('problem HLL UNRESOLVED CASE')

    if bReshaped:
        return face_flux[:,0] #face_flux.reshape((face_flux.size,))
    else:
        return face_flux

def computeFluxes(WL,WR,direction,options):
    if direction==1: #the face is perpendicular to the y axis
      # We perform a change of reference so that the Riemann solver always solves a problem in the x-direction
      # --> x-split two-dimensional solver
      WR[1,:], WR[2,:] = WR[2,:], WR[1,:]
      WL[1,:], WL[2,:] = WL[2,:], WL[1,:]
    
    #TODO: add other solvers
    face_flux = HLLCsolver(WL,WR,options)
    
    if direction==1: # swap back the axis
      face_flux[1,:], face_flux[2,:] = face_flux[2,:], face_flux[1,:]
      # We reset the order of the components so that we don't have any issue further on
      # TODO: est-ce bien utile ?
      WR[1,:], WR[2,:] = WR[2,:], WR[1,:]
      WL[1,:], WL[2,:] = WL[2,:], WL[1,:]
    return face_flux

def modelfun(t,x,options):
    """ ODE function for Euler equation """
    print(t)
    global nIter
    nIter+=1

    ##### gather mesh
    surfaces = options['mesh']['cells']['surfaces']
    nx = options['mesh']['cells']['nx']
    ny = options['mesh']['cells']['ny']

    ##### recover conserved variables
    rho, rhoU, rhoV, rhoE = getVarsFromX(x, options)
    temp = computeOtherVariables(rho, rhoU, rhoV, rhoE)
    u = temp['u']
    P = temp['P']

    Wup = np.zeros((4,nx*(ny-1)))
    Wup[0,:] = rho[:-1,:].reshape((-1,))
    Wup[1,:] = rhoU[:-1,:].reshape((-1,))
    Wup[2,:] = rhoV[:-1,:].reshape((-1,))
    Wup[3,:] = rhoE[:-1,:].reshape((-1,))

    Wdown = np.zeros((4,nx*(ny-1)))    
    Wdown[0,:] = rho[1:,:].reshape((-1,))
    Wdown[1,:] = rhoU[1:,:].reshape((-1,))
    Wdown[2,:] = rhoV[1:,:].reshape((-1,))
    Wdown[3,:] = rhoE[1:,:].reshape((-1,))

    Wleft = np.zeros((4,(nx-1)*ny))    
    Wleft[0,:] = rho[:,:-1].reshape((-1,))
    Wleft[1,:] = rhoU[:,:-1].reshape((-1,))
    Wleft[2,:] = rhoV[:,:-1].reshape((-1,))
    Wleft[3,:] = rhoE[:,:-1].reshape((-1,))
    
    Wright = np.zeros((4,(nx-1)*ny))    
    Wright[0,:] = rho[:,1:].reshape((-1,))
    Wright[1,:] = rhoU[:,1:].reshape((-1,))
    Wright[2,:] = rhoV[:,1:].reshape((-1,))
    Wright[3,:] = rhoE[:,1:].reshape((-1,))

    ## Vertical fluxes, towards y>0
    fluxes_down = np.zeros((4,ny+1,nx))
    fluxes_down_inner  = computeFluxes(Wup,Wdown,direction=1,options=options)
    fluxes_down[:,1:-1,:] = fluxes_down_inner.reshape((4,ny-1,nx))

    fluxes_right = np.zeros((4,ny,nx+1))    
    fluxes_right_inner = computeFluxes(Wleft,Wright,direction=0,options=options)
    fluxes_right[:,:,1:-1] = fluxes_right_inner.reshape((4,ny,nx-1))
    
    
    
    # TODO: BCs
    # periodic BCs
    # fluxes_down[0,:,:]=fluxes_down[-2,:,:]
    # fluxes_down[-1,:,:]=fluxes_down[1,:,:]
    Wup = np.zeros((4,nx))
    Wup[0,:] = rho[-1,:]
    Wup[1,:] = rhoU[-1,:]
    Wup[2,:] = rhoV[-1,:]
    Wup[3,:] = rhoE[-1,:]

    Wdown = np.zeros((4,nx))    
    Wdown[0,:] = rho[0,:]
    Wdown[1,:] = rhoU[0,:]
    Wdown[2,:] = rhoV[0,:]
    Wdown[3,:] = rhoE[0,:]

    Wleft = np.zeros((4,ny))    
    Wleft[0,:] = rho[:,-1]
    Wleft[1,:] = rhoU[:,-1]
    Wleft[2,:] = rhoV[:,-1]
    Wleft[3,:] = rhoE[:,-1]
    
    Wright = np.zeros((4,ny))    
    Wright[0,:] = rho[:,0]
    Wright[1,:] = rhoU[:,0]
    Wright[2,:] = rhoV[:,0]
    Wright[3,:] = rhoE[:,0]
   
    ## Vertical fluxes, towards y>0
    fluxes_down_outer  = computeFluxes(Wup,Wdown,direction=1,options=options)
    fluxes_down[:,0,:] = fluxes_down_outer.reshape((4,nx))
    fluxes_down[:,-1,:] = fluxes_down[:,0,:]
    
    fluxes_right_outer = computeFluxes(Wleft,Wright,direction=0,options=options)
    fluxes_right[:,:,0] = fluxes_right_outer.reshape((4,ny))
    fluxes_right[:,:,-1] = fluxes_right[:,:,0]
    
    #### Calcul des dérivées temporelles
    # dxdt + div(ux) = 0
    # surface * dX/dt = somme(u_faces * longueur_face)
    
    fluxes_down[:,:,:] = options['mesh']['faces']['dx']* fluxes_down
    fluxes_right[:,:,:] =options['mesh']['faces']['dy']* fluxes_right
    
    time_deriv = (1/options['mesh']['cells']['surfaces'])*(fluxes_down[:,:-1,:] - fluxes_down[:,1:,:]+fluxes_right[:,:,:-1] - fluxes_right[:,:,1:])


    dxdt = getXFromVars(rho=time_deriv[0,:,:],  rhoU=time_deriv[1,:,:],
                        rhoV=time_deriv[2,:,:], rhoE=time_deriv[3,:,:])
    if np.isnan(dxdt).any():
        raise Exception('NaNs in time_deriv, at time t={}'.format(t))
    return dxdt

def setupFiniteVolumMesh(xfaces, yfaces):
  """ Setup the mesh data structure for a 2D cartesian structure grid. """
  mesh = {'cells':{}, 'faces':{}}
  xx, yy = np.meshgrid(xfaces,yfaces)
  nx_faces = xfaces.size
  ny_faces = yfaces.size
    
  dx = np.diff(xx,axis=1)#[:-1,:]
  dy = np.diff(yy,axis=0)#[:,:-1]
  surfaces = dx[:-1,:] * dy[:,:-1]
  # surfaces = dx*dy
  
  mesh['faces'] = {'dx' : dx, 'dy':dy}
  mesh['cells'] = {'nx' : nx_faces-1, 'ny' : ny_faces-1, 'surfaces': surfaces,
                   'x': ( xx[:-1, 1:] + xx[:-1,:-1] )*0.5,
                   'y': ( yy[1:,:-1]  + yy[:-1,:-1] )*0.5
                   }
  return mesh
    
def getXFromVars(rho, rhoU, rhoV, rhoE):
    if rho.ndim==2:
        return np.dstack((rho, rhoU, rhoV, rhoE)).reshape((-1,), order='C')
    else: # time axis or perturbations
        return np.dstack((rho, rhoU, rhoV, rhoE)).reshape((-1, rho.shape[2]), order='C')
    # rho_0 = rho_0.reshape((-1,), order='C')
    # u_0 = u_0.reshape((-1,), order='C')
    # E_0 = E_0.reshape((-1,), order='C')        
    # X0 = np.vstack((rho_0, rho_0*u_0, rho_0*E_0)).reshape((-1,),order='F')

def getVarsFromX(X, options):
    nx = options['mesh']['cells']['nx']
    ny = options['mesh']['cells']['ny']
    Xresh = X.reshape((ny,nx,4))
    rho = Xresh[:,:,0]
    rhoU = Xresh[:,:,1]
    rhoV = Xresh[:,:,2]
    rhoE = Xresh[:,:,3]
    return rho,rhoU,rhoV,rhoE


def getVarsFromX_vectorized(X, options):
    nx = options['mesh']['cells']['nx']
    ny = options['mesh']['cells']['ny']
    Xresh = X.reshape((ny,nx,4,-1))
    rho = Xresh[:,:,0,:]
    rhoU = Xresh[:,:,1,:]
    rhoV = Xresh[:,:,2,:]
    rhoE = Xresh[:,:,3,:]
    return rho,rhoU,rhoV,rhoE

if __name__=='__main__':
    #%%
    options={'mesh':{}, 'BCs':{'left':{}, 'right':{}}}
    nx,ny = 40,3
    xmin,xmax = 0,1
    ymin,ymax = 0,1
    
    xfaces = np.linspace(xmin,xmax,nx)
    yfaces = np.linspace(ymin,ymax,ny)
    options['mesh'] = setupFiniteVolumMesh(xfaces, yfaces)
    
    # Sod shock tube (invariant along y)
    
    xcells = options['mesh']['cells']['x']
    ycells = options['mesh']['cells']['y']
    P_0 = np.zeros_like(xcells)
    # xc = xmax/2.
    #iselec= xcells<xc
    xc1,xc2=0.4,0.6
    iselec = np.logical_and(xcells>xc1, xcells<xc2)
    not_iselec = np.logical_not(iselec)
    P_0[iselec]  =  1.*1e5
    P_0[not_iselec] =  1.*1e5

    rho_0 = np.zeros_like(xcells)
    rho_0[iselec]  =  1.0
    rho_0[not_iselec] =  0.125

    u_0 = np.zeros_like(xcells)
    u_0[iselec]  =  100.
    u_0[not_iselec] =  100.
    
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

            
    #%% NUMERICAL INTEGRATION
    tend= 1e-1 #1.5150127621344831e-05
    out  = integrate.solve_ivp(fun=lambda t,x: modelfun(t,x,options), t_span=(0.,tend), y0=X0, first_step=1e-9,
                               max_step=np.inf, method='BDF', atol=1e-4, rtol=1e-3)
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
        Jac = scipy.optimize._numdiff.approx_derivative(
                                    fun=testfun,
                                    x0=X0, method='2-point',
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
plt.figure()
plt.plot(xcells[0,:], rho[0,:,-1], color='b', label='num', marker='+', linestyle='')
plt.xlabel('position')
plt.ylabel(r'$\rho$')
plt.title('Densité')
