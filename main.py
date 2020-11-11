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
      F[1,:] = rhoU*v
      F[2,:] = (rhoE + P)*u
    elif direction==1:
      F[0,:] = rhoV
      F[1,:] = rhoU*v
      F[1,:] = rhoV*v + P
      F[2,:] = (rhoE + P)*v
    else:
      raise Exception('direction can only be 0 (y) or 1 (x)')
    if bReshaped:
        return F.reshape((F.size,))
    else:
        return F


def HLLCsolver(WL,WR,direction,options):
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
    
    if direction==0: #the face is perpendicular to the y axis
      rhoUR,rhoUL = rhoUL,rhoUR
      uR,uL=uL,uR

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
    Wstar_L[0,:] = rhoL*(SL-uL)/(SL - Sstar)
    Wstar_L[1,:] = rhoL*(SL-uL)/(SL - Sstar)*Sstar
    Wstar_L[2,:] = #TODO
    Wstar_L[3,:] = rhoL*(SL-uL)/(SL - Sstar)*( EL+ (Sstar-uL)*(Sstar + PL/(rhoL*(SL-uL))) )

    Wstar_R = np.zeros_like(WL)
    Wstar_R[0,:] = rhoR*(SR-uR)/(SR - Sstar)
    Wstar_R[1,:] = rhoR*(SR-uR)/(SR - Sstar)*Sstar
    Wstar_R[2,:] = #TODO:
    Wstar_R[3,:] = rhoR*(SR-uR)/(SR - Sstar)*( ER+ (Sstar-uR)*(Sstar + PR/(rhoR*(SR-uR))) )

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


#    # OLD FOR LOOP
#    for i in range(WL.shape[1]):
#        # right cell = i, left cell = i-1
#        # estimate the wave speeds
#        if 0: #based on Roe-average
#            utilde = (np.sqrt(rhoL[i])*uL[i] + np.sqrt(rhoR[i])*uR[i])/(np.sqrt(rhoL[i]) + np.sqrt(rhoR[i]))
#            Htilde = (np.sqrt(rhoL[i])*HL[i] + np.sqrt(rhoR[i])*HR[i])/(np.sqrt(rhoL[i]) + np.sqrt(rhoR[i]))
#            atilde = np.sqrt( (gamma-1)*(Htilde-0.5*utilde*utilde) )
#            SL = utilde-atilde
#            SR = utilde+atilde
#        else:
#            SL = np.min( [uL[i]-aL[i], uR[i]-aR[i]] )
#            SR = np.min( [uL[i]+aL[i], uR[i]+aR[i]] )
#        # compute Sstar
#        Sstar = ( PR[i]-PL[i] + rhoL[i]*uL[i]*(SL-uL[i]) - rhoR[i]*uR[i]*(SR-uR[i]) ) / ( rhoL[i]*(SL-uL[i]) - rhoR[i]*(SR-uR[i]) )
#        Wstar_L = rhoL[i]*(SL-uL[i])/(SL - Sstar)*np.array( [1,Sstar, EL[i]+ (Sstar-uL[i])*(Sstar + PL[i]/(rhoL[i]*(SL-uL[i]))) ])
#        Wstar_R = rhoR[i]  *(SR-uR[i]  )/(SR - Sstar)*np.array( [1,Sstar, ER[i]  + (Sstar-uR[i]  )*(Sstar +   PR[i]/(rhoR[i]  *(SR-uR[i]  ))) ])
#        if SL>0:
#            face_flux[:,i] = fluxEulerPhysique(WL[:,i])
#        elif SL<=0 and Sstar>=0:
#            face_flux[:,i] = fluxEulerPhysique(Wstar_L)
#        elif Sstar<0 and SR>0:
#            face_flux[:,i] = fluxEulerPhysique(Wstar_R)
#        elif SR<=0:
#            face_flux[:,i] = fluxEulerPhysique(WR[:,i])
#        else:
#            raise Exception('problem HLL UNRESOLVED CASE')
    if direction==0: # swap back the axis
      face_flux[1,:], face_flux[2,:] = face_flux[2,:], face_flux[1,:]

    if bReshaped:
        return face_flux[:,0] #face_flux.reshape((face_flux.size,))
    else:
        return face_flux

def computeFluxes(WL,WR,options):
    face_flux = HLLCsolver(WL,WR,options)
    return face_flux

def modelfun(t,x,options, nMode):
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
    Wup[0,:] = rho[:-1,:,:].reshape((-1,))
    Wup[1,:] = rhoU[:-1,:,:].reshape((-1,))
    Wup[2,:] = rhoU[:-1,:,:].reshape((-1,))
    Wup[3,:] = rhoE[:-1,:,:].reshape((-1,))

    Wdown = np.zeros((4,nx*(ny-1)))    
    Wdown[0,:] = rho[1:,:,:].reshape((-1,))
    Wdown[1,:] = rhoU[1:,:,:].reshape((-1,))
    Wdown[2,:] = rhoV[1:,:,:].reshape((-1,))
    Wdown[3,:] = rhoE[1:,:,:].reshape((-1,))

    Wleft = np.zeros((4,(nx-1)*ny))    
    Wleft[0,:] = rho[:,:-1,:].reshape((-1,))
    Wleft[1,:] = rhoU[:,:-1,:].reshape((-1,))
    Wleft[2,:] = rhoU[:,:-1,:].reshape((-1,))
    Wleft[3,:] = rhoE[:,:-1,:].reshape((-1,))
    
    Wright = np.zeros((4,(nx-1)*ny))    
    Wright[0,:] = rho[:,1:,:].reshape((-1,))
    Wright[1,:] = rhoU[:,1:,:].reshape((-1,))
    Wright[2,:] = rhoU[:,1:,:].reshape((-1,))
    Wright[3,:] = rhoE[:,1:,:].reshape((-1,))

    ## Vertical fluxes, towards y>0
    fluxes_down  = computeFluxes(Wup,Wdown,options)
    fluxes_right = computeFluxes(Wleft,Wright,options)
    
    # TODO: BCs
    
    
    if nMode==1: #centered with artificial dissipation
        Wfaces = np.zeros((W.shape[0], W.shape[1]+1))*np.nan # valeurs aux faces
        Wfaces[:,1:-1] = (W[:,:-1]+W[:,1:])/2
        Wfaces[:,0]   = W[:,0]
        Wfaces[:,-1]  = W[:,-1]

        face_flux = fluxEulerPhysique(Wfaces)
        dWdx_faces = np.zeros_like(Wfaces)

        for i in range(3):
            dWdx_faces[i,1:-1] = (W[i,1:]-W[i,:-1])/options['mesh']['dxBetweenCellCenters']
            dWdx_faces[i,0] = (W[i,0]-Wfaces[i,0])/(xcells[0]-xfaces[0])
            dWdx_faces[i,-1] = (W[i,-1]-Wfaces[i,-1])/(xcells[-1]-xfaces[-1])

        D = 2*1e1
        dissipation_flux = -D*dWdx_faces
        face_flux = face_flux + dissipation_flux

    elif nMode==2: # Flux Vector Splitting
        out = computeOtherVariables(rho, rhoU, rhoV, rhoE)
        u,v,P,E,H,a = out['u'], out['v'], out['P'], out['E'], out['H'], out['a']
#        dt_max = np.min( np.abs( cell_width /a) )
#        print('t={}, dt_max={:.2e}'.format(t,dt_max))

        face_flux = np.zeros( (W.shape[0], W.shape[1]+1) )
        for i in range(nx): #cellule i --> faces i et i+1
            # face_flux(i) = Fmoins(i)+Fplus(i-1)
            valeurs_propres = np.array([u[i]-a[i], u[i], u[i]+a[i]])
#            lbda_plus    = (valeurs_propres + np.abs(valeurs_propres))/2
#            lbda_moins   = (valeurs_propres - np.abs(valeurs_propres))/2
            epsilon = 1e-2
            lbda_plus    = (valeurs_propres + np.sqrt(valeurs_propres*valeurs_propres + epsilon**2))/2
            lbda_moins   = (valeurs_propres - np.sqrt(valeurs_propres*valeurs_propres + epsilon**2))/2

            # calcul des différentes parties des flux
            # pour rester cohérent avec la notation de Toro page 276
            Fplus = np.zeros(3)
            Fplus[0] = rho[i]/(2*gamma) * (                  lbda_plus[0] +         2*(gamma-1)*lbda_plus[1] +                  lbda_plus[2] )
            Fplus[1] = rho[i]/(2*gamma) * (      (u[i]-a[i])*lbda_plus[0] +    2*(gamma-1)*u[i]*lbda_plus[1] +      (u[i]+a[i])*lbda_plus[2] )
            Fplus[2] = rho[i]/(2*gamma) * ( (H[i]-u[i]*a[i])*lbda_plus[0] + (gamma-1)*u[i]*u[i]*lbda_plus[1] + (H[i]+u[i]*a[i])*lbda_plus[2] )

            Fmoins    = np.zeros(3)
            Fmoins[0] = rho[i]/(2*gamma) * (                 lbda_moins[0] +         2*(gamma-1)*lbda_moins[1] +                  lbda_moins[2] )
            Fmoins[1] = rho[i]/(2*gamma) * (     (u[i]-a[i])*lbda_moins[0] +    2*(gamma-1)*u[i]*lbda_moins[1] +      (u[i]+a[i])*lbda_moins[2] )
            Fmoins[2] = rho[i]/(2*gamma) * ((H[i]-u[i]*a[i])*lbda_moins[0] + (gamma-1)*u[i]*u[i]*lbda_moins[1] + (H[i]+u[i]*a[i])*lbda_moins[2] )

            face_flux[:,i]   = face_flux[:,i]   + Fmoins
            face_flux[:,i+1] = face_flux[:,i+1] + Fplus
            if i==0: #left BC contribution (transmissive)
                face_flux[:,0]    = Fmoins + Fplus
            if i==nx-1: #right BC
                face_flux[:,nx]   = Fmoins + Fplus

    elif nMode==3: #HLL
        time_deriv = np.zeros_like(W)
        face_flux = np.zeros( (W.shape[0], W.shape[1]+1) )
        face_flux[:,1:-1] = HLLCsolver(WL=W[:,:-1],WR=W[:,1:], options=options)
        # compute the fluxes at the BCs
        nBC = 0
        if nBC==0 : #transmissive
            face_flux[:,0]  = fluxEulerPhysique(W[:,0])
            face_flux[:,-1] = fluxEulerPhysique(W[:,-1])
        else: #reflection
            face_flux[:,0]  = fluxEulerPhysique(np.array([ -W[0,0], -W[1,0], W[2,0]]))
            face_flux[:,-1] = fluxEulerPhysique(np.array([ -W[0,-1], -W[1,-1], W[2,-1]]))
            
    elif nMode==5: # non-TVD MUSCL-Hancock (see Toro page 505)
        #### 1 - RECONSTRUCTION DES VALEURS AUX FACES
        # calcul des pentes
        delta_faces = np.zeros( (W.shape[0], W.shape[1]+1) )
        delta_faces[:,1:-1] = (W[:,1:]-W[:,:-1])/xgaps
        # transmissive BCs --> delta_faces[0 ou -1] = 0

        # calcul des pentes au centre des cellules
        omega = 0.5 #should be in [-1,1]
        delta_cells = 0.5*(1+omega)*delta_faces[:,:-1] + 0.5*(1-omega)*delta_faces[:,1:]
        delta_cells = 0*delta_cells

        # calcul des valeurs aux faces
        WL = np.zeros_like(W)
        WR = np.zeros_like(W)
        for i in range(3):
            WL[i,:] =  W[i,:] + delta_cells[i,:]*(xcells-xfaces[:-1]) # /!\ on ne fait pas comme Toro, on utilise les gradients, pas les différences --> mieux pour maillage non-uniforme
            WR[i,:] =  W[i,:] + delta_cells[i,:]*(xfaces[1:]-xcells)

        #### 2 - EVOLUTION INDEPENDANTE DES VALEURS AUX FACES
        # TODO: add t as a variable so to access dt=Delta_T for evolution step
        WL_evo = WL
        WR_evo = WR

        if 0:
            for i in range(3):
                plt.figure()
                plt.plot(WR_evo[i,:], label='WR_evo', color='r')
                plt.plot(WL_evo[i,:], label='WL_evo', color='b')
                plt.title('Variable {} at t={}'.format(i, t))
                plt.show()

        #### 3 - RESOLUTION DU PROBLEME DE RIEMANN
        nSolver = 1
        if nSolver==1:
            RiemannSolver = HLLCsolver # ce solveur donne le flux directement
            face_flux = np.zeros( (W.shape[0], W.shape[1]+1) )
            face_flux[:,1:-1] = RiemannSolver(WL=WR_evo[:,:-1],WR=WL_evo[:,1:],options=options)
            # transmissive BCs
            face_flux[:,0] = RiemannSolver(WL=WL[:,0],WR=WL[:,0],options=options)
            face_flux[:,-1] = RiemannSolver(WL=WR[:,-1],WR=WR[:,-1],options=options)
        else:
            raise Exception('!')
            face_flux = np.zeros( (W.shape[0], W.shape[1]+1) )
            Wfaces   = np.zeros( (W.shape[0], W.shape[1]+1) )
            # ...
            face_flux = fluxEulerPhysique(Wfaces)

    else:
        raise Exception('unknown solver mode {}'.format(nMode))

    #### Calcul des dérivées temporelles
    time_deriv = (1/options['mesh']['cellSize'])*(face_flux[:,:-1]-face_flux[:,1:])
#    time_deriv = time_deriv.T.flatten() #### FAUX!!!!!
    time_deriv = np.hstack([ time_deriv[i,:] for i in range(time_deriv.shape[0]) ] )
    if np.isnan(time_deriv).any():
        raise Exception('NaNs in time_deriv, at time t={}'.format(t))
    return time_deriv

def setupFiniteVolumMesh(xfaces, yfaces):
  """ Setup the mesh data structure for a 2D cartesian structure grid. """
  mesh = {'cells':{}, 'faces':{}}
  xx, yy = np.meshgrid(xfaces,yfaces)
  nx_faces = xfaces.size
  ny_faces = yfaces.size
    
  dx = np.diff(xx,axis=1)[:-1,:]
  dy = np.diff(yy,axis=0)[:,:-1]
  surfaces = dx*dy
  
  mesh['faces'] = {'dx' : dx, 'dy':dy}
  mesh['cells'] = {'nx' : nx_faces-1, 'ny' : ny_faces-1, 'surfaces': surfaces,
                   'x': ( xx[:-1, 1:] + xx[:-1,:-1] )*0.5,
                   'y': ( yy[1:,:-1]  + yy[:-1,:-1] )*0.5
                   }
  return mesh
    
def getXFromVars(rho, rhoU, rhoE):
    if rho.ndim==2:
        return np.dstack((rho, rhoU, rhoE)).reshape((-1,), order='C')
    else: # time axis or perturbations
        return np.dstack((rho, rhoU, rhoE)).reshape((-1, rho.shape[2]), order='C')
    # rho_0 = rho_0.reshape((-1,), order='C')
    # u_0 = u_0.reshape((-1,), order='C')
    # E_0 = E_0.reshape((-1,), order='C')        
    # X0 = np.vstack((rho_0, rho_0*u_0, rho_0*E_0)).reshape((-1,),order='F')

def getVarsFromX(X, options):
    nx = options['mesh']['cells']['nx']
    ny = options['mesh']['cells']['ny']
    Xresh = X.reshape((ny,nx,3,-1))
    rho = Xresh[:,:,0,:]
    rhoU = Xresh[:,:,1,:]
    rhoE = Xresh[:,:,2,:]
    return rho,rhoU,rhoE

if __name__=='__main__':
    #%%
    options={'mesh':{}, 'BCs':{'left':{}, 'right':{}}}
    nx,ny = 10,11
    xmin,xmax = 0,1
    ymin,ymax = 0,1
    
    xfaces = np.linspace(xmin,xmax,nx)
    yfaces = np.linspace(ymin,ymax,ny)
    options['mesh'] = setupFiniteVolumMesh(xfaces, yfaces)
    
    # Sod shock tube (invariant along y)
    xc = xmax/2.
    xcells = options['mesh']['cells']['x']
    ycells = options['mesh']['cells']['y']
    P_0 = np.zeros_like(xcells)
    P_0[xcells<xc]  =  1.*1e5
    P_0[xcells>=xc] =  0.1*1e5

    rho_0 = np.zeros_like(xcells)
    rho_0[xcells<xc]  =  1.0
    rho_0[xcells>=xc] =  0.125

    u_0 = np.zeros_like(xcells)
    u_0[xcells<xc]  =  100.
    u_0[xcells>=xc] =  50.


    T_0 = computeT(P_0, rho_0)
    E_0 = cv*T_0 + 0.5*u_0*u_0
    

    X0 = getXFromVars(rho_0, rho_0*u_0, rho_0*E_0)
    rho, rhoU, rhoE = getVarsFromX(X0, options)
    
    surfaces = options['mesh']['cells']['surfaces']
    nx = options['mesh']['cells']['nx']
    ny = options['mesh']['cells']['ny']

    ##### recover conserved variables
    rho, rhoU, rhoE = getVarsFromX(X0, options)
    temp = computeOtherVariables(rho, rhoU, rhoE)
    u = temp['u']
    P = temp['P']

            
    #%% NUMERICAL INTEGRATION
    tend=0.01
    out  = integrate.solve_ivp(fun=lambda t,x: modelfun(t,x,options, nMode=2), t_span=(0.,tend), y0=X0, first_step=1e-9,
                               max_step=np.inf, method='RK45', atol=1e-9, rtol=1e-9)
    #%% GATHER RESULTS
    rho  = out.y[:nx,:]
    rhoU = out.y[nx:2*nx,:]
    rhoE = out.y[2*nx:,:]
    temp = computeOtherVariables(rho, rhoU, rhoE)
    u,v,T,P = temp['u'], temp['v'], temp['T'], temp['P']
    time = out.t
    # u[i,:] correspond au champ u au i-ème pas de temps

    selected_time_indices = [ i.astype(int) for i in np.linspace(0,time.size-1,20) ]
    plotfuncustom(xcells, P.T, time, 'P', selected_time_indices, marker=None)
    plotfuncustom(xcells, u.T, time, 'u', selected_time_indices, marker=None)
    plotfuncustom(xcells, rho.T, time, 'rho', selected_time_indices, marker=None)
    plotfuncustom(xcells, rhoU.T, time, 'rhoU', selected_time_indices, marker=None)
    plotfuncustom(xcells, rhoE.T, time, 'rhoE', selected_time_indices, marker=None)
    plotfuncustom(xcells, T.T, time, 't', selected_time_indices, marker=None)

    #%% JACOBIAN ANALYSIS
    if 0:
        options['bUseComplexStep'] = False
        options['bVectorisedModelFun'] = False
        Jac = computeJacobian(modelfun=lambda x: modelfun(t=out.t[-1],x=x,options=options),
                              x=out.y[:,-1], options=options, bReturnResult=False)
        Jac = computeJacobian(modelfun=lambda x: modelfun(t=0,x=x,options=options),
                          x=X0, options=options, bReturnResult=False)

        plt.figure()
        plt.spy(Jac)
        n_rank_jac = np.linalg.matrix_rank(Jac),
        plt.title('Jacobian (rank={}, shape={})'.format(n_rank_jac, np.shape(Jac)))
        plt.show()
        if n_rank_jac[0]!=np.size(Jac,1):
            print('The following rows of the Jacobian are nil:\n\t{}'.format( np.where( (Jac==0).all(axis=1) ) ))
            print('The following columns of the Jacobian are nil:\n\t{}'.format( np.where( (Jac==0).all(axis=0) ) ))
        if np.size(Jac,1)<500:
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
