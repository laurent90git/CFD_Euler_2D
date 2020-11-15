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
    rhoL, rhoUL, rhoVL, rhoEL = WL[0,:], WL[1,:], WL[2,:], WL[3,:]
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
    # return (WL+WR)*0.5
    if direction==1: #the face is perpendicular to the y axis
      # We perform a change of reference so that the Riemann solver always solves a problem in the x-direction
      # --> x-split two-dimensional solver
      WR[[1,2],:] = WR[[2,1],:]
      WL[[1,2],:] = WL[[2,1],:]
    
    #TODO: add other solvers
    face_flux = HLLCsolver(WL,WR,options)
    
    if direction==1: # swap back the axis
      face_flux[[1,2],:] = face_flux[[2,1],:]
      # We reset the order of the components so that we don't have any issue further on
      # TODO: est-ce bien utile ?
      WR[[1,2],:] = WR[[2,1],:]
      WL[[1,2],:] = WL[[2,1],:]
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
    
    if options["BCs"]["left_right"] == "periodic":
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
        
        fluxes_right_outer = computeFluxes(Wleft,Wright,direction=0,options=options)
        fluxes_right[:,:,0] = fluxes_right_outer.reshape((4,ny))
        fluxes_right[:,:,-1] = fluxes_right[:,:,0]      


    if options["BCs"]["up_down"] == "periodic":
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
        
            ## Vertical fluxes, towards y>0
        fluxes_down_outer  = computeFluxes(Wup,Wdown,direction=1,options=options)
        fluxes_down[:,0,:] = fluxes_down_outer.reshape((4,nx))
        fluxes_down[:,-1,:] = fluxes_down[:,0,:]
        
    if options["BCs"]["left_right"] == "reflective":
        # left side
        Wleft = np.zeros((4,ny))    
        Wleft[0,:] = rho[:,0]
        Wleft[1,:] = -rhoU[:,0]
        Wleft[2,:] = rhoV[:,0]
        Wleft[3,:] = rhoE[:,0]
        
        Wright = np.zeros((4,ny))    
        Wright[0,:] = rho[:,0]
        Wright[1,:] = rhoU[:,0]
        Wright[2,:] = rhoV[:,0]
        Wright[3,:] = rhoE[:,0]
        
        fluxes_right_outer = computeFluxes(Wleft,Wright,direction=0,options=options)
        fluxes_right[:,:,0] = fluxes_right_outer.reshape((4,ny))
        
        # right side
        Wleft = np.zeros((4,ny))    
        Wleft[0,:] = rho[:,-1]
        Wleft[1,:] = rhoU[:,-1]
        Wleft[2,:] = rhoV[:,-1]
        Wleft[3,:] = rhoE[:,-1]
        
        Wright = np.zeros((4,ny))    
        Wright[0,:] = rho[:,-1]
        Wright[1,:] = -rhoU[:,-1]
        Wright[2,:] = rhoV[:,-1]
        Wright[3,:] = rhoE[:,-1]
        
        fluxes_right_outer = computeFluxes(Wleft,Wright,direction=0,options=options)
        fluxes_right[:,:,-1] = fluxes_right_outer.reshape((4,ny))


    if options["BCs"]["up_down"] == "reflective":
        # top side
        Wup = np.zeros((4,nx))
        Wup[0,:] =  rho[0,:]
        Wup[1,:] =  rhoU[0,:]
        Wup[2,:] = -rhoV[0,:]
        Wup[3,:] =  rhoE[0,:]
    
        Wdown = np.zeros((4,nx))    
        Wdown[0,:] = rho[0,:]
        Wdown[1,:] = rhoU[0,:]
        Wdown[2,:] = rhoV[0,:]
        Wdown[3,:] = rhoE[0,:]
        
        ## Vertical fluxes, towards y>0
        fluxes_down_outer  = computeFluxes(Wup,Wdown,direction=1,options=options)
        fluxes_down[:,0,:] = fluxes_down_outer.reshape((4,nx))
        
        # botoom side
        Wup = np.zeros((4,nx))
        Wup[0,:] = rho[-1,:]
        Wup[1,:] = rhoU[-1,:]
        Wup[2,:] = rhoV[-1,:]
        Wup[3,:] = rhoE[-1,:]
    
        Wdown = np.zeros((4,nx))    
        Wdown[0,:] =  rho[-1,:]
        Wdown[1,:] =  rhoU[-1,:]
        Wdown[2,:] = -rhoV[-1,:]
        Wdown[3,:] =  rhoE[-1,:]
        
        ## Vertical fluxes, towards y>0
        fluxes_down_outer  = computeFluxes(Wup,Wdown,direction=1,options=options)
        fluxes_down[:,-1,:] = fluxes_down_outer.reshape((4,nx))
            
    if options["BCs"]["left_right"] == "transmissive":
        # left side
        W = np.zeros((4,ny))    
        W[0,:] = rho[:,0]
        W[1,:] = rhoU[:,0]
        W[2,:] = rhoV[:,0]
        W[3,:] = rhoE[:,0]
        fluxes_right_outer = fluxEulerPhysique(W, direction=0)
        fluxes_right[:,:,0] = fluxes_right_outer.reshape((4,ny))
        
        # right side
        W = np.zeros((4,ny))    
        W[0,:] = rho[:,-1]
        W[1,:] = rhoU[:,-1]
        W[2,:] = rhoV[:,-1]
        W[3,:] = rhoE[:,-1]
        fluxes_right_outer = fluxEulerPhysique(W, direction=0)        
        fluxes_right[:,:,-1] = fluxes_right_outer.reshape((4,ny))
        
    if options["BCs"]["up_down"] == "transmissive":
        W = np.zeros((4,nx))    
        W[0,:] = rho[0,:]
        W[1,:] = rhoU[0,:]
        W[2,:] = rhoV[0,:]
        W[3,:] = rhoE[0,:]
        fluxes_down_outer  = fluxEulerPhysique(W, direction=1)
        fluxes_down[:,0,:] = fluxes_down_outer.reshape((4,nx))
        
        # bottom
        W = np.zeros((4,nx))    
        W[0,:] = rho[-1,:]
        W[1,:] = rhoU[-1,:]
        W[2,:] = rhoV[-1,:]
        W[3,:] = rhoE[-1,:]
        fluxes_down_outer  = fluxEulerPhysique(W, direction=1)        
        fluxes_down[:,-1,:] = fluxes_down_outer.reshape((4,nx))

        
# options["BC"]["left_right"] == "reflective"    
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

def setupFiniteVolumeMesh(xfaces, yfaces):
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
