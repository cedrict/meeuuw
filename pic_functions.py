import numpy as np
import numba
from basis_functions import basis_functions_V

###############################################################################
# these functions are used in the Runge-Kutta algorithm advection. As such 
# they need to locate the particle before interpolating the velocity onto it.
###############################################################################

@numba.njit
def interpolate_vel_on_pt___box(xp,yp,u,v,hx,hy,nelx,icon,x_V,y_V):
    """
    Args:
       xp,yp: coordinates of point
       u,v: velocity arrays (size nn_V)
       hx,hy: element size
       nelx: nb of elements in x direction
       icon: connectivity array of V nodes
       x_V,y_V: cartesian coordinate arrays of V nodes
    Returns:
    """
    ielx=int(xp/hx)
    iely=int(yp/hy)
    iel=nelx*iely+ielx
    rp=((xp-x_V[icon[0,iel]])/hx-0.5)*2
    sp=((yp-y_V[icon[0,iel]])/hy-0.5)*2
    N=basis_functions_V(rp,sp)
    up=np.dot(N,u[icon[:,iel]])
    vp=np.dot(N,v[icon[:,iel]])
    return up,vp,iel

@numba.njit
def interpolate_vel_on_pt___quarter(xp,yp,u,v,hrad,htheta,nelx,icon,rad_V,theta_V,Rinner):
    """
    Args:
       xp,yp: coordinates of point
       u,v: velocity arrays (size nn_V)
       hrad,htheta: element size if polar coordinates
       nelx: nb of elements in x direction
       icon: connectivity array of V nodes
       rad_V,theta_V: polar coordinate arrays of V nodes
       Rinner: inner radius of annulus
    Returns:
    """
    radp=np.sqrt(xp**2+yp**2)
    thetap=np.pi/2-np.arctan2(xp,yp)
    ielx=int((np.pi/2-thetap)/htheta)
    iely=int((radp-Rinner)/hrad)
    iel=nelx*iely+ielx
    rp=((-thetap+theta_V[icon[0,iel]])/htheta-0.5)*2
    sp=((radp-rad_V[icon[0,iel]])/hrad-0.5)*2
    N=basis_functions_V(rp,sp)
    up=np.dot(N,u[icon[:,iel]])
    vp=np.dot(N,v[icon[:,iel]])
    return up,vp,iel

###############################################################################

@numba.njit
def interpolate_field_on_particle(rp,sp,iel,phi,icon):
    """
    Args:
       rp,sp: reduced coordinates of point
       iel: element index it sits in
       phi: array of size nn_V
       icon: connectivity of V nodes
    Returns:
    """
    N=basis_functions_V(rp,sp)
    phip=np.dot(N,phi[icon[:,iel]])
    return phip

@numba.njit
def interpolate_field_on_particles(nparticle,swarm_r,swarm_s,swarm_iel,phi,icon):
    """
    Args:
       nparticle: number of particles
       swarm_r,swarm_s: reduced coordinates arrays of all particles
       swarm_iel: cell index of all particles
       phi: array of size nn_V
       icon: connectivity of V nodes
    Returns:
    """
    swarm_field=np.zeros(nparticle,dtype=np.float64)
    for ip in range(0,nparticle):
        N=basis_functions_V(swarm_r[ip],swarm_s[ip])
        swarm_field[ip]=np.dot(N,phi[icon[:,swarm_iel[ip]]])
    return swarm_field

###############################################################################

@numba.njit
def locate_particles___box(nparticle,swarm_x,swarm_y,hx,hy,x_V,y_V,icon,nelx):
    """
    Args:
       nparticle: number of particles
       swarm_x,swarm_y: cartesian coordinates arrays of all particles
       hx,hy: element size
       x_V,y_V: cartesian coordinate arrays of V nodes
       icon: connectivity of V nodes
       nelx: nb of elements in x direction
    Returns:
    """
    swarm_r=np.zeros(nparticle,dtype=np.float64)
    swarm_s=np.zeros(nparticle,dtype=np.float64)
    swarm_iel=np.zeros(nparticle,dtype=np.int32)
    for ip in range(0,nparticle):
        ielx=int(swarm_x[ip]/hx)
        iely=int(swarm_y[ip]/hy)
        iel=nelx*iely+ielx
        swarm_r[ip]=((swarm_x[ip]-x_V[icon[0,iel]])/hx-0.5)*2
        swarm_s[ip]=((swarm_y[ip]-y_V[icon[0,iel]])/hy-0.5)*2
        swarm_iel[ip]=iel
    return swarm_r,swarm_s,swarm_iel

###############################################################################

@numba.njit
def locate_particles___quarter(nparticle,swarm_rad,swarm_theta,hrad,htheta,rad_V,theta_V,icon,nelx,Rinner):
    """
    Args:
       nparticle: number of particles
       swarm_rad,swarm_theta: polar coordinates arrays of all particles
       hrad,htheta: element size if polar coordinates
       rad_V,theta_V: polar coordinate arrays of V nodes
       icon: connectivity array of V nodes
       nelx: nb of elements in x direction
       Rinner: inner radius of annulus
    Returns:
       swarm_r,swarm_s: reduced coordinates of all particles
       swarm_iel: cell index of all particles
    """
    swarm_r=np.zeros(nparticle,dtype=np.float64)
    swarm_s=np.zeros(nparticle,dtype=np.float64)
    swarm_iel=np.zeros(nparticle,dtype=np.int32)
    for ip in range(0,nparticle):
        ielx=int((np.pi/2-swarm_theta[ip])/htheta)
        iely=int((swarm_rad[ip]-Rinner)/hrad)
        iel=nelx*iely+ielx
        swarm_r[ip]=((-swarm_theta[ip]+theta_V[icon[0,iel]])/htheta-0.5)*2
        swarm_s[ip]=((swarm_rad[ip]-rad_V[icon[0,iel]])/hrad-0.5)*2
        swarm_iel[ip]=iel

    return swarm_r,swarm_s,swarm_iel

###############################################################################

@numba.njit
def advect_particles___box(RKorder,dt,nparticle,swarm_x,swarm_y,swarm_active,u,v,\
                           Lx,Ly,hx,hy,nelx,icon,x_V,y_V):
    """
    Args:
       RKorder: Runge-Kutta algorithm order (1,2,4)
       dt: time step value
       nparticle: number of particles
       swarm_x,swarm_y: cartesian coordinates of particles
       swarm_active: boolean array (True if particle still in domain)
       u,v: velocity arrays on FE mesh
       Lx,Ly: domain dimensions
       hx,hy: element size
       nelx: nb of elements in x direction
       icon: connectivity array of V nodes
       x_V,y_V: cartesian coordinate arrays of V nodes
    Returns:
       swarm_x,swarm_y: cartesian coordinates of particles
       swarm_u,swarm_v: velocity of particles
       swarm_active: boolean array (True if particle still in domain)
    """

    swarm_u=np.zeros(nparticle,dtype=np.float64)
    swarm_v=np.zeros(nparticle,dtype=np.float64)

    if RKorder==1:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              swarm_u[ip],swarm_v[ip],iel=\
              interpolate_vel_on_pt___box(swarm_x[ip],swarm_y[ip],u,v,\
                                          hx,hy,nelx,icon,x_V,y_V)
              swarm_x[ip]+=swarm_u[ip]*dt
              swarm_y[ip]+=swarm_v[ip]*dt
              if swarm_x[ip]<0 or swarm_x[ip]>Lx or swarm_y[ip]<0 or swarm_y[ip]>Ly:
                 swarm_active[ip]=False
           # end if active
       # end for ip

    elif RKorder==2:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              xA=swarm_x[ip]
              yA=swarm_y[ip]
              uA,vA,iel=interpolate_vel_on_pt___box(xA,yA,u,v,\
                                                    hx,hy,nelx,icon,x_V,y_V)
              xB=xA+uA*dt/2.
              yB=yA+vA*dt/2.
              if xB<0 or xB>Lx or yB<0 or yB>Ly:
                 swarm_active[ip]=False
              else:
                 uB,vB,iel=interpolate_vel_on_pt___box(xB,yB,u,v,\
                                                       hx,hy,nelx,icon,x_V,y_V)
                 swarm_x[ip]=xA+uB*dt
                 swarm_y[ip]=yA+vB*dt
                 swarm_u[ip]=uB
                 swarm_v[ip]=vB
              # end if active
           # end if active
       # end for ip

    elif RKorder==4:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              xA=swarm_x[ip]
              yA=swarm_y[ip]
              uA,vA,iel=interpolate_vel_on_pt___box(xA,yA,u,v,hx,hy,nelx,icon,x_V,y_V)
              xB=xA+uA*dt/2.
              yB=yA+vA*dt/2.
              if xB<0 or xB>Lx or yB<0 or yB>Ly:
                 swarm_active[ip]=False
              else:
                 uB,vB,iel=interpolate_vel_on_pt___box(xB,yB,u,v,hx,hy,nelx,icon,x_V,y_V)
                 xC=xA+uB*dt/2.
                 yC=yA+vB*dt/2.
                 if xC<0 or xC>Lx or yC<0 or yC>Ly:
                    swarm_active[ip]=False
                 else:
                    uC,vC,iel=interpolate_vel_on_pt___box(xC,yC,u,v,hx,hy,nelx,icon,x_V,y_V)
                    xD=xA+uC*dt
                    yD=yA+vC*dt
                    if xD<0 or xD>Lx or yD<0 or yD>Ly:
                       swarm_active[ip]=False
                    else:
                       uD,vD,iel=interpolate_vel_on_pt___box(xD,yD,u,v,hx,hy,nelx,icon,x_V,y_V)
                       swarm_u[ip]=(uA+2*uB+2*uC+uD)/6
                       swarm_v[ip]=(vA+2*vB+2*vC+vD)/6
                       swarm_x[ip]=xA+swarm_u[ip]*dt
                       swarm_y[ip]=yA+swarm_v[ip]*dt
                    # end if active
                 # end if active
              # end if active
           # end if active
       # end for im

    for im in range(0,nparticle):
        if not swarm_active[im]:
           swarm_x[im]=0
           swarm_y[im]=0

    return swarm_x,swarm_y,swarm_u,swarm_v,swarm_active

###############################################################################

@numba.njit
def advect_particles___quarter(RKorder,dt,nparticle,swarm_x,swarm_y,
                               swarm_rad,swarm_theta,swarm_active,u,v,
                               Rinner,Router,hrad,htheta,nelx,icon,rad_V,theta_V):
    """
    Args:
       RKorder: Runge-Kutta algorithm order (1,2,4)
       dt: time step value
       nparticle: number of particles
       swarm_x,swarm_y: cartesian coordinates of particles
       swarm_rad,swarm_theta: polar coordinates of particles
       swarm_active: boolean array (True if particle still in domain)
       u,v: velocity arrays on FE mesh
       Rinner,Router: inner and outer radius of annulus
       hrad,htheta: element size if polar coordinates
       nelx: nb of elements in x direction
       icon: connectivity array of V nodes
       rad_V,theta_V: polar coordinates of V nodes
    Returns:
       swarm_x,swarm_y: cartesian coordinates of particles
       swarm_rad,swarm_theta: polar coordinates of particles
       swarm_u,swarm_v: velocity of particles
       swarm_active: boolean array (True if particle still in domain)
    """

    swarm_u=np.zeros(nparticle,dtype=np.float64)
    swarm_v=np.zeros(nparticle,dtype=np.float64)

    if RKorder==1:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              swarm_u[ip],swarm_v[ip],iel=\
              interpolate_vel_on_pt___quarter(swarm_x[ip],swarm_y[ip],u,v,\
                                              hrad,htheta,nelx,icon,rad_V,theta_V,Rinner)
              swarm_x[ip]+=swarm_u[ip]*dt
              swarm_y[ip]+=swarm_v[ip]*dt
              swarm_rad[ip]=np.sqrt(swarm_x[ip]**2+swarm_y[ip]**2)
              swarm_theta[ip]=np.pi/2-np.arctan2(swarm_x[ip],swarm_y[ip])
              if swarm_x[ip]<0 or swarm_y[ip]<0 or swarm_rad[ip]<Rinner or swarm_rad[ip]>Router:
                 swarm_active[ip]=False
           # end if active
       # end for ip

    elif RKorder==2:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              xA=swarm_x[ip]
              yA=swarm_y[ip]
              uA,vA,iel=\
              interpolate_vel_on_pt___quarter(xA,yA,u,v,\
                                              hrad,htheta,nelx,icon,rad_V,theta_V,Rinner)

              xB=xA+uA*dt/2.
              yB=yA+vA*dt/2.
              rB=np.sqrt(xA**2+yA**2)
              if xB<0 or yB<0 or rB<Rinner or rB>Router:
                 swarm_active[ip]=False
              else:
                 uB,vB,iel=\
                 interpolate_vel_on_pt___quarter(xB,yB,u,v,\
                                                 hrad,htheta,nelx,icon,rad_V,theta_V,Rinner)
                 swarm_x[ip]=xA+uB*dt
                 swarm_y[ip]=yA+vB*dt
                 swarm_u[ip]=uB
                 swarm_v[ip]=vB
                 swarm_rad[ip]=np.sqrt(swarm_x[ip]**2+swarm_y[ip]**2)
                 swarm_theta[ip]=np.pi/2-np.arctan2(swarm_x[ip],swarm_y[ip])
                 if swarm_x[ip]<0 or swarm_y[ip]<0 or swarm_rad[ip]<Rinner or swarm_rad[ip]>Router:
                    swarm_active[ip]=False
              # end if active
           # end if active
       # end for ip

    for ip in range(0,nparticle):
        if not swarm_active[ip]:
           swarm_x[ip]=0
           swarm_y[ip]=0

    return swarm_x,swarm_y,swarm_rad,swarm_theta,swarm_u,swarm_v,swarm_active

###############################################################################

# iel inside or out ?

@numba.njit
def project_particles_on_elements(nel,nparticle,swarm_rho,swarm_eta,swarm_iel,averaging):
    """
    Args:
       nel: number of elements 
       nparticle: number of particles
       swarm_rho: density carried by particles
       swarm_eta: viscosity carried by particles
       swarm_iel: element index of particles
       averaging: arithmetic, geometric or harmonic
    Returns:
       rho_elemental: elemental density field
       eta_elemental: elemental viscosity field
       nparticle_per_element: nb of particles per element
    """

    rho_elemental=np.zeros(nel,dtype=np.float64) 
    eta_elemental=np.zeros(nel,dtype=np.float64) 
    nparticle_per_element=np.zeros(nel,dtype=np.float64) 

    # density averaging is always arithmetic 
    for ip in range(0,nparticle):
        iel=swarm_iel[ip]
        rho_elemental[iel]+=swarm_rho[ip]
        nparticle_per_element[iel]+=1
    rho_elemental/=nparticle_per_element

    # viscosity
    if averaging=='arithmetic':
       for ip in range(0,nparticle):
           eta_elemental[swarm_iel[ip]]+=swarm_eta[ip] 
       eta_elemental/=nparticle_per_element

    elif averaging=='geometric':
       for ip in range(0,nparticle):
           eta_elemental[swarm_iel[ip]]+=np.log10(swarm_eta[ip])
       eta_elemental=10.**(eta_elemental/nparticle_per_element)

    elif averaging=='harmonic':
       for ip in range(0,nparticle):
           eta_elemental[swarm_iel[ip]]+=1./swarm_eta[ip]
       eta_elemental=nparticle_per_element/eta_elemental

    return rho_elemental,eta_elemental,nparticle_per_element

###############################################################################

@numba.njit
def project_particle_field_on_nodes(nel,nn_V,nparticle,swarm_phi,icon,swarm_iel,averaging):
    """
    Args:
       nel: number of elements
       nn_V: number of V nodes
       nparticle: number of particles
       swarm_phi: field carried by all particles
       icon: connectivity array of V nodes
       swarm_iel: cell index of all particles
       averaging: arithmetic, geometric or harmonic
    Returns:
    """

    phi_nodal=np.zeros(nn_V,dtype=np.float64) 
    count_nodal=np.zeros(nn_V,dtype=np.float64) 

    if averaging=='arithmetic':
       for ip in range(0,nparticle):
           iel=swarm_iel[ip]
           for k in (0,1,2,3):
               phi_nodal[icon[k,iel]]+=swarm_phi[ip]
               count_nodal[icon[k,iel]]+=1
       for i in range(0,nn_V):
           if count_nodal[i]!=0:
              phi_nodal[i]/=count_nodal[i]

    elif averaging=='geometric':
       for ip in range(0,nparticle):
           iel=swarm_iel[ip]
           for k in (0,1,2,3):
               phi_nodal[icon[k,iel]]+=np.log10(swarm_phi[ip])
               count_nodal[icon[k,iel]]+=1
       for i in range(0,nn_V):
           if count_nodal[i]!=0:
              phi_nodal[i]=10**(phi_nodal[i]/count_nodal[i])

    elif averaging=='harmonic':
       for ip in range(0,nparticle):
           iel=swarm_iel[ip]
           for k in (0,1,2,3):
               phi_nodal[icon[k,iel]]+=1./swarm_phi[ip]
               count_nodal[icon[k,iel]]+=1
       for i in range(0,nn_V):
           if count_nodal[i]!=0:
              phi_nodal[i]=count_nodal[i]/phi_nodal[i]

    # (implicitely) use Q1 basis functions to compute field at other nodes
    for iel in range(0,nel):
        phi_nodal[icon[4,iel]]=0.50*(phi_nodal[icon[0,iel]]+phi_nodal[icon[1,iel]])
        phi_nodal[icon[5,iel]]=0.50*(phi_nodal[icon[1,iel]]+phi_nodal[icon[2,iel]])
        phi_nodal[icon[6,iel]]=0.50*(phi_nodal[icon[2,iel]]+phi_nodal[icon[3,iel]])
        phi_nodal[icon[7,iel]]=0.50*(phi_nodal[icon[0,iel]]+phi_nodal[icon[3,iel]])
        phi_nodal[icon[8,iel]]=0.25*(phi_nodal[icon[0,iel]]+phi_nodal[icon[1,iel]]\
                                    +phi_nodal[icon[2,iel]]+phi_nodal[icon[3,iel]])

    return phi_nodal

###############################################################################
