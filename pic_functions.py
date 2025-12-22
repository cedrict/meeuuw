import numpy as np
import numba
from basis_functions import basis_functions_V

###############################################################################
# these functions are used in the Runge-Kutta algorithm advection. As such 
# they need to locate the particle before interpolating the velocity onto it.
###############################################################################

@numba.njit
def interpolate_vel_on_pt___box(xp,zp,u,w,hx,hz,nelx,icon,x_V,z_V):
    """
    Args:
       xp,zp: coordinates of point
       u,w: velocity arrays (size nn_V)
       hx,hz: element size
       nelx: nb of elements in x direction
       icon: connectivity array of V nodes
       x_V,z_V: cartesian coordinate arrays of V nodes
    Returns:
    """
    ielx=int(xp/hx)
    ielz=int(zp/hz)
    iel=nelx*ielz+ielx
    rp=((xp-x_V[icon[0,iel]])/hx-0.5)*2
    tp=((zp-z_V[icon[0,iel]])/hz-0.5)*2
    N=basis_functions_V(rp,tp)
    up=np.dot(N,u[icon[:,iel]])
    wp=np.dot(N,w[icon[:,iel]])
    return up,wp,iel

@numba.njit
def interpolate_vel_on_pt___annulus(xp,yp,u,w,hrad,htheta,nelx,icon,rad_V,theta_V,Rinner):
    """
    Args:
       xp,yp: coordinates of point
       u,w: velocity arrays (size nn_V)
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
    ielz=int((radp-Rinner)/hrad)
    iel=nelx*ielz+ielx
    rp=((-thetap+theta_V[icon[0,iel]])/htheta-0.5)*2
    tp=((radp-rad_V[icon[0,iel]])/hrad-0.5)*2
    N=basis_functions_V(rp,tp)
    up=np.dot(N,u[icon[:,iel]])
    wp=np.dot(N,w[icon[:,iel]])
    return up,wp,iel

###############################################################################

@numba.njit
def interpolate_field_on_particle(rp,tp,iel,phi,icon):
    """
    Args:
       rp,tp: reduced coordinates of point
       iel: element index it sits in
       phi: array of size nn_V
       icon: connectivity of V nodes
    Returns:
    """
    N=basis_functions_V(rp,tp)
    phip=np.dot(N,phi[icon[:,iel]])
    return phip

@numba.njit
def interpolate_field_on_particles(nparticle,swarm_r,swarm_t,swarm_iel,phi,icon):
    """
    Args:
       nparticle: number of particles
       swarm_r,swarm_t: reduced coordinates arrays of all particles
       swarm_iel: cell index of all particles
       phi: array of size nn_V
       icon: connectivity of V nodes
    Returns:
    """
    swarm_field=np.zeros(nparticle,dtype=np.float64)
    for ip in range(0,nparticle):
        N=basis_functions_V(swarm_r[ip],swarm_t[ip])
        swarm_field[ip]=np.dot(N,phi[icon[:,swarm_iel[ip]]])
    return swarm_field

###############################################################################

@numba.njit
def locate_particles___box(nparticle,swarm_x,swarm_z,hx,hz,x_V,z_V,icon,nelx):
    """
    Args:
       nparticle: number of particles
       swarm_x,swarm_z: cartesian coordinates arrays of all particles
       hx,hz: element size
       x_V,z_V: cartesian coordinate arrays of V nodes
       icon: connectivity of V nodes
       nelx: nb of elements in x direction
    Returns:
    """
    swarm_r=np.zeros(nparticle,dtype=np.float64)
    swarm_t=np.zeros(nparticle,dtype=np.float64)
    swarm_iel=np.zeros(nparticle,dtype=np.int32)
    for ip in range(0,nparticle):
        ielx=int(swarm_x[ip]/hx)
        ielz=int(swarm_z[ip]/hz)
        iel=nelx*ielz+ielx
        swarm_r[ip]=((swarm_x[ip]-x_V[icon[0,iel]])/hx-0.5)*2
        swarm_t[ip]=((swarm_z[ip]-z_V[icon[0,iel]])/hz-0.5)*2
        swarm_iel[ip]=iel
    return swarm_r,swarm_t,swarm_iel

###############################################################################

@numba.njit
def locate_particles___annulus(nparticle,swarm_rad,swarm_theta,hrad,htheta,rad_V,theta_V,icon,nelx,Rinner):
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
       swarm_r,swarm_t: reduced coordinates of all particles
       swarm_iel: cell index of all particles
    """
    swarm_r=np.zeros(nparticle,dtype=np.float64)
    swarm_t=np.zeros(nparticle,dtype=np.float64)
    swarm_iel=np.zeros(nparticle,dtype=np.int32)
    for ip in range(0,nparticle):
        ielx=int((np.pi/2-swarm_theta[ip])/htheta)
        ielz=int((swarm_rad[ip]-Rinner)/hrad)
        iel=nelx*ielz+ielx
        swarm_r[ip]=((-swarm_theta[ip]+theta_V[icon[0,iel]])/htheta-0.5)*2
        swarm_t[ip]=((swarm_rad[ip]-rad_V[icon[0,iel]])/hrad-0.5)*2
        swarm_iel[ip]=iel

    return swarm_r,swarm_t,swarm_iel

###############################################################################

@numba.njit
def advect_particles___box(RKorder,dt,nparticle,swarm_x,swarm_z,swarm_active,u,w,\
                           Lx,Lz,hx,hz,nelx,icon,x_V,z_V):
    """
    Args:
       RKorder: Runge-Kutta algorithm order (1,2,4)
       dt: time step value
       nparticle: number of particles
       swarm_x,swarm_z: cartesian coordinates of particles
       swarm_active: boolean array (True if particle still in domain)
       u,w: velocity arrays on FE mesh
       Lx,Lz: domain dimensions
       hx,hz: element size
       nelx: nb of elements in x direction
       icon: connectivity array of V nodes
       x_V,z_V: cartesian coordinate arrays of V nodes
    Returns:
       swarm_x,swarm_z: cartesian coordinates of particles
       swarm_u,swarm_w: velocity of particles
       swarm_active: boolean array (True if particle still in domain)
    """

    swarm_u=np.zeros(nparticle,dtype=np.float64)
    swarm_w=np.zeros(nparticle,dtype=np.float64)

    if RKorder==1:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              swarm_u[ip],swarm_w[ip],iel=\
              interpolate_vel_on_pt___box(swarm_x[ip],swarm_z[ip],u,w,\
                                          hx,hz,nelx,icon,x_V,z_V)
              swarm_x[ip]+=swarm_u[ip]*dt
              swarm_z[ip]+=swarm_w[ip]*dt
              if swarm_x[ip]<0 or swarm_x[ip]>Lx or swarm_z[ip]<0 or swarm_z[ip]>Lz:
                 swarm_active[ip]=False
           # end if active
       # end for ip

    elif RKorder==2:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              xA=swarm_x[ip]
              zA=swarm_z[ip]
              uA,wA,iel=interpolate_vel_on_pt___box(xA,zA,u,w,\
                                                    hx,hz,nelx,icon,x_V,z_V)
              xB=xA+uA*dt/2.
              yB=zA+wA*dt/2.
              if xB<0 or xB>Lx or yB<0 or yB>Lz:
                 swarm_active[ip]=False
              else:
                 uB,wB,iel=interpolate_vel_on_pt___box(xB,yB,u,w,\
                                                       hx,hz,nelx,icon,x_V,z_V)
                 swarm_x[ip]=xA+uB*dt
                 swarm_z[ip]=zA+wB*dt
                 swarm_u[ip]=uB
                 swarm_w[ip]=wB
              # end if active
           # end if active
       # end for ip

    elif RKorder==4:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              xA=swarm_x[ip]
              zA=swarm_z[ip]
              uA,wA,iel=interpolate_vel_on_pt___box(xA,zA,u,w,hx,hz,nelx,icon,x_V,z_V)
              xB=xA+uA*dt/2.
              yB=zA+wA*dt/2.
              if xB<0 or xB>Lx or yB<0 or yB>Lz:
                 swarm_active[ip]=False
              else:
                 uB,wB,iel=interpolate_vel_on_pt___box(xB,yB,u,w,hx,hz,nelx,icon,x_V,z_V)
                 xC=xA+uB*dt/2.
                 yC=zA+wB*dt/2.
                 if xC<0 or xC>Lx or yC<0 or yC>Lz:
                    swarm_active[ip]=False
                 else:
                    uC,wC,iel=interpolate_vel_on_pt___box(xC,yC,u,w,hx,hz,nelx,icon,x_V,z_V)
                    xD=xA+uC*dt
                    yD=zA+wC*dt
                    if xD<0 or xD>Lx or yD<0 or yD>Lz:
                       swarm_active[ip]=False
                    else:
                       uD,wD,iel=interpolate_vel_on_pt___box(xD,yD,u,w,hx,hz,nelx,icon,x_V,z_V)
                       swarm_u[ip]=(uA+2*uB+2*uC+uD)/6
                       swarm_w[ip]=(wA+2*wB+2*wC+wD)/6
                       swarm_x[ip]=xA+swarm_u[ip]*dt
                       swarm_z[ip]=zA+swarm_w[ip]*dt
                    # end if active
                 # end if active
              # end if active
           # end if active
       # end for im

    for im in range(0,nparticle):
        if not swarm_active[im]:
           swarm_x[im]=0
           swarm_z[im]=0

    return swarm_x,swarm_z,swarm_u,swarm_w,swarm_active

###############################################################################

@numba.njit
def advect_particles___eighth(RKorder,dt,nparticle,swarm_x,swarm_z,
                              swarm_rad,swarm_theta,swarm_active,u,w,
                              Rinner,Router,hrad,htheta,nelx,icon,rad_V,theta_V):

    swarm_u=np.zeros(nparticle,dtype=np.float64)
    swarm_w=np.zeros(nparticle,dtype=np.float64)

    if True: #RKorder==1:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              swarm_u[ip],swarm_w[ip],iel=\
              interpolate_vel_on_pt___annulus(swarm_x[ip],swarm_z[ip],u,w,\
                                              hrad,htheta,nelx,icon,rad_V,theta_V,Rinner)
              swarm_x[ip]+=swarm_u[ip]*dt
              swarm_z[ip]+=swarm_w[ip]*dt
              swarm_rad[ip]=np.sqrt(swarm_x[ip]**2+swarm_z[ip]**2)
              swarm_theta[ip]=np.pi/2-np.arctan2(swarm_x[ip],swarm_z[ip])
              if swarm_x[ip]<0 or swarm_rad[ip]<Rinner or swarm_rad[ip]>Router or swarm_z[ip]<swarm_x[ip]:
                 swarm_active[ip]=False
           # end if active
       # end for ip

    for ip in range(0,nparticle):
        if not swarm_active[ip]:
           swarm_x[ip]=0
           swarm_z[ip]=0

    return swarm_x,swarm_z,swarm_rad,swarm_theta,swarm_u,swarm_w,swarm_active

###############################################################################

@numba.njit
def advect_particles___quarter(RKorder,dt,nparticle,swarm_x,swarm_z,
                               swarm_rad,swarm_theta,swarm_active,u,w,
                               Rinner,Router,hrad,htheta,nelx,icon,rad_V,theta_V):
    """
    Args:
       RKorder: Runge-Kutta algorithm order (1,2,4)
       dt: time step value
       nparticle: number of particles
       swarm_x,swarm_z: cartesian coordinates of particles
       swarm_rad,swarm_theta: polar coordinates of particles
       swarm_active: boolean array (True if particle still in domain)
       u,w: velocity arrays on FE mesh
       Rinner,Router: inner and outer radius of annulus
       hrad,htheta: element size if polar coordinates
       nelx: nb of elements in x direction
       icon: connectivity array of V nodes
       rad_V,theta_V: polar coordinates of V nodes
    Returns:
       swarm_x,swarm_z: cartesian coordinates of particles
       swarm_rad,swarm_theta: polar coordinates of particles
       swarm_u,swarm_w: velocity of particles
       swarm_active: boolean array (True if particle still in domain)
    """

    swarm_u=np.zeros(nparticle,dtype=np.float64)
    swarm_w=np.zeros(nparticle,dtype=np.float64)

    if RKorder==1:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              swarm_u[ip],swarm_w[ip],iel=\
              interpolate_vel_on_pt___annulus(swarm_x[ip],swarm_z[ip],u,w,\
                                              hrad,htheta,nelx,icon,rad_V,theta_V,Rinner)
              swarm_x[ip]+=swarm_u[ip]*dt
              swarm_z[ip]+=swarm_w[ip]*dt
              swarm_rad[ip]=np.sqrt(swarm_x[ip]**2+swarm_z[ip]**2)
              swarm_theta[ip]=np.pi/2-np.arctan2(swarm_x[ip],swarm_z[ip])
              if swarm_x[ip]<0 or swarm_z[ip]<0 or swarm_rad[ip]<Rinner or swarm_rad[ip]>Router:
                 swarm_active[ip]=False
           # end if active
       # end for ip

    elif RKorder==2:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              xA=swarm_x[ip]
              zA=swarm_z[ip]
              uA,wA,iel=\
              interpolate_vel_on_pt___annulus(xA,zA,u,w,\
                                              hrad,htheta,nelx,icon,rad_V,theta_V,Rinner)

              xB=xA+uA*dt/2.
              yB=zA+wA*dt/2.
              rB=np.sqrt(xA**2+zA**2)
              if xB<0 or yB<0 or rB<Rinner or rB>Router:
                 swarm_active[ip]=False
              else:
                 uB,wB,iel=\
                 interpolate_vel_on_pt___annulus(xB,yB,u,w,\
                                                 hrad,htheta,nelx,icon,rad_V,theta_V,Rinner)
                 swarm_x[ip]=xA+uB*dt
                 swarm_z[ip]=zA+wB*dt
                 swarm_u[ip]=uB
                 swarm_w[ip]=wB
                 swarm_rad[ip]=np.sqrt(swarm_x[ip]**2+swarm_z[ip]**2)
                 swarm_theta[ip]=np.pi/2-np.arctan2(swarm_x[ip],swarm_z[ip])
                 if swarm_x[ip]<0 or swarm_z[ip]<0 or swarm_rad[ip]<Rinner or swarm_rad[ip]>Router:
                    swarm_active[ip]=False
              # end if active
           # end if active
       # end for ip

    for ip in range(0,nparticle):
        if not swarm_active[ip]:
           swarm_x[ip]=0
           swarm_z[ip]=0

    return swarm_x,swarm_z,swarm_rad,swarm_theta,swarm_u,swarm_w,swarm_active

###############################################################################

@numba.njit
def advect_particles___half(RKorder,dt,nparticle,swarm_x,swarm_z,
                            swarm_rad,swarm_theta,swarm_active,u,w,
                            Rinner,Router,hrad,htheta,nelx,icon,rad_V,theta_V):
    """
    Args:
       RKorder: Runge-Kutta algorithm order (1,2,4)
       dt: time step value
       nparticle: number of particles
       swarm_x,swarm_z: cartesian coordinates of particles
       swarm_rad,swarm_theta: polar coordinates of particles
       swarm_active: boolean array (True if particle still in domain)
       u,w: velocity arrays on FE mesh
       Rinner,Router: inner and outer radius of annulus
       hrad,htheta: element size if polar coordinates
       nelx: nb of elements in x direction
       icon: connectivity array of V nodes
       rad_V,theta_V: polar coordinates of V nodes
    Returns:
       swarm_x,swarm_z: cartesian coordinates of particles
       swarm_rad,swarm_theta: polar coordinates of particles
       swarm_u,swarm_w: velocity of particles
       swarm_active: boolean array (True if particle still in domain)
    """

    swarm_u=np.zeros(nparticle,dtype=np.float64)
    swarm_w=np.zeros(nparticle,dtype=np.float64)

    if RKorder==1:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              swarm_u[ip],swarm_w[ip],iel=\
              interpolate_vel_on_pt___annulus(swarm_x[ip],swarm_z[ip],u,w,\
                                              hrad,htheta,nelx,icon,rad_V,theta_V,Rinner)
              swarm_x[ip]+=swarm_u[ip]*dt
              swarm_z[ip]+=swarm_w[ip]*dt
              swarm_rad[ip]=np.sqrt(swarm_x[ip]**2+swarm_z[ip]**2)
              swarm_theta[ip]=np.pi/2-np.arctan2(swarm_x[ip],swarm_z[ip])
              if swarm_x[ip]<0 or swarm_rad[ip]<Rinner or swarm_rad[ip]>Router:
                 swarm_active[ip]=False
           # end if active
       # end for ip

    elif RKorder==2:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              xA=swarm_x[ip]
              zA=swarm_z[ip]
              uA,wA,iel=\
              interpolate_vel_on_pt___annulus(xA,zA,u,w,\
                                              hrad,htheta,nelx,icon,rad_V,theta_V,Rinner)

              xB=xA+uA*dt/2.
              yB=zA+wA*dt/2.
              rB=np.sqrt(xA**2+zA**2)
              if xB<0 or rB<Rinner or rB>Router:
                 swarm_active[ip]=False
              else:
                 uB,wB,iel=\
                 interpolate_vel_on_pt___annulus(xB,yB,u,w,\
                                                 hrad,htheta,nelx,icon,rad_V,theta_V,Rinner)
                 swarm_x[ip]=xA+uB*dt
                 swarm_z[ip]=zA+wB*dt
                 swarm_u[ip]=uB
                 swarm_w[ip]=wB
                 swarm_rad[ip]=np.sqrt(swarm_x[ip]**2+swarm_z[ip]**2)
                 swarm_theta[ip]=np.pi/2-np.arctan2(swarm_x[ip],swarm_z[ip])
                 if swarm_x[ip]<0 or swarm_rad[ip]<Rinner or swarm_rad[ip]>Router:
                    swarm_active[ip]=False
              # end if active
           # end if active
       # end for ip

    for ip in range(0,nparticle):
        if not swarm_active[ip]:
           swarm_x[ip]=0
           swarm_z[ip]=0

    return swarm_x,swarm_z,swarm_rad,swarm_theta,swarm_u,swarm_w,swarm_active

###############################################################################

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

###################################################################################################
# this version of the algorithm uses all the particles around a given node, without any weighing
###################################################################################################

@numba.njit
def project_particle_field_on_nodes_1(nel,nn_V,nparticle,swarm_phi,icon,swarm_iel,
                                      swarm_r,swarm_t,averaging):
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
       phi_nodal: field phi on V nodes 
    """

    phi_nodal=np.zeros(nn_V,dtype=np.float64) 
    count_nodal=np.zeros(nn_V,dtype=np.int16) 

    if averaging=='arithmetic':
       for ip in range(0,nparticle):
           iel=swarm_iel[ip]
           for k in (0,1,2,3):
               phi_nodal[icon[k,iel]]+=swarm_phi[ip]
               count_nodal[icon[k,iel]]+=1         
           #end for
       #end for
       phi_nodal/=count_nodal

    elif averaging=='geometric':
       for ip in range(0,nparticle):
           iel=swarm_iel[ip]
           for k in (0,1,2,3):
               phi_nodal[icon[k,iel]]+=np.log10(swarm_phi[ip])
               count_nodal[icon[k,iel]]+=1                  
           #end for
       #end for
       phi_nodal=10**(phi_nodal/count_nodal)

    elif averaging=='harmonic':
       for ip in range(0,nparticle):
           iel=swarm_iel[ip]
           for k in (0,1,2,3):
               phi_nodal[icon[k,iel]]+=1./swarm_phi[ip]
               count_nodal[icon[k,iel]]+=1            
           #end for
       #end for
       phi_nodal=count_nodal/phi_nodal

    # (implicitely) use Q1 basis functions to compute field at other nodes
    for iel in range(0,nel):
        phi_nodal[icon[4,iel]]=0.50*(phi_nodal[icon[0,iel]]+phi_nodal[icon[1,iel]])
        phi_nodal[icon[5,iel]]=0.50*(phi_nodal[icon[1,iel]]+phi_nodal[icon[2,iel]])
        phi_nodal[icon[6,iel]]=0.50*(phi_nodal[icon[2,iel]]+phi_nodal[icon[3,iel]])
        phi_nodal[icon[7,iel]]=0.50*(phi_nodal[icon[0,iel]]+phi_nodal[icon[3,iel]])
        phi_nodal[icon[8,iel]]=0.25*(phi_nodal[icon[0,iel]]+phi_nodal[icon[1,iel]]\
                                    +phi_nodal[icon[2,iel]]+phi_nodal[icon[3,iel]])

    return phi_nodal

###################################################################################################
# this version of the algorithm uses all the particles around a given node, without Q1 weighing
###################################################################################################

@numba.njit
def project_particle_field_on_nodes_2(nel,nn_V,nparticle,swarm_phi,icon,swarm_iel,
                                      swarm_r,swarm_t,averaging):
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
       phi_nodal: field phi on V nodes 
    """

    phi_nodal=np.zeros(nn_V,dtype=np.float64) 
    count_nodal=np.zeros(nn_V,dtype=np.float64) 
    N=np.zeros(4,dtype=np.float64) 

    if averaging=='arithmetic':
       for ip in range(0,nparticle):
           N[0]=0.25*(1-swarm_r[ip])*(1-swarm_t[ip])
           N[1]=0.25*(1+swarm_r[ip])*(1-swarm_t[ip])
           N[2]=0.25*(1+swarm_r[ip])*(1+swarm_t[ip])
           N[3]=0.25*(1-swarm_r[ip])*(1+swarm_t[ip])
           iel=swarm_iel[ip]
           for k in (0,1,2,3):
               phi_nodal[icon[k,iel]]+=swarm_phi[ip] * N[k]
               count_nodal[icon[k,iel]]+=1           * N[k]
           #end for
       #end for
       phi_nodal/=count_nodal

    elif averaging=='geometric':
       for ip in range(0,nparticle):
           N[0]=0.25*(1-swarm_r[ip])*(1-swarm_t[ip])
           N[1]=0.25*(1+swarm_r[ip])*(1-swarm_t[ip])
           N[2]=0.25*(1+swarm_r[ip])*(1+swarm_t[ip])
           N[3]=0.25*(1-swarm_r[ip])*(1+swarm_t[ip])
           iel=swarm_iel[ip]
           for k in (0,1,2,3):
               phi_nodal[icon[k,iel]]+=np.log10(swarm_phi[ip]) * N[k]
               count_nodal[icon[k,iel]]+=1                     * N[k]
           #end for
       #end for
       phi_nodal=10**(phi_nodal/count_nodal)

    elif averaging=='harmonic':
       for ip in range(0,nparticle):
           N[0]=0.25*(1-swarm_r[ip])*(1-swarm_t[ip])
           N[1]=0.25*(1+swarm_r[ip])*(1-swarm_t[ip])
           N[2]=0.25*(1+swarm_r[ip])*(1+swarm_t[ip])
           N[3]=0.25*(1-swarm_r[ip])*(1+swarm_t[ip])
           iel=swarm_iel[ip]
           for k in (0,1,2,3):
               phi_nodal[icon[k,iel]]+=1./swarm_phi[ip] * N[k]
               count_nodal[icon[k,iel]]+=1              * N[k]
           #end for
       #end for
       phi_nodal=count_nodal/phi_nodal

    # (implicitely) use Q1 basis functions to compute field at other nodes
    for iel in range(0,nel):
        phi_nodal[icon[4,iel]]=0.50*(phi_nodal[icon[0,iel]]+phi_nodal[icon[1,iel]])
        phi_nodal[icon[5,iel]]=0.50*(phi_nodal[icon[1,iel]]+phi_nodal[icon[2,iel]])
        phi_nodal[icon[6,iel]]=0.50*(phi_nodal[icon[2,iel]]+phi_nodal[icon[3,iel]])
        phi_nodal[icon[7,iel]]=0.50*(phi_nodal[icon[0,iel]]+phi_nodal[icon[3,iel]])
        phi_nodal[icon[8,iel]]=0.25*(phi_nodal[icon[0,iel]]+phi_nodal[icon[1,iel]]\
                                    +phi_nodal[icon[2,iel]]+phi_nodal[icon[3,iel]])

    return phi_nodal

###################################################################################################
# this version of the algorithm uses the particles around a given node, but only those 
# in the corresponding quadrants of each enighouring element, without Q1 weighing
###################################################################################################

@numba.njit
def project_particle_field_on_nodes_3(nel,nn_V,nparticle,swarm_phi,icon,swarm_iel,
                                      swarm_r,swarm_t,averaging):
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
       phi_nodal: field phi on V nodes 
    """

    phi_nodal=np.zeros(nn_V,dtype=np.float64) 
    count_nodal=np.zeros(nn_V,dtype=np.int16) 

    if averaging=='arithmetic':
       for ip in range(0,nparticle):
           iel=swarm_iel[ip]
           #lower left corner - node 0
           if swarm_r[ip]<=0 and swarm_t[ip]<=0:
              phi_nodal[icon[0,iel]]+=swarm_phi[ip] ; count_nodal[icon[0,iel]]+=1          
           #lower left corner - node 1
           if swarm_r[ip]>=0 and swarm_t[ip]<=0:
              phi_nodal[icon[1,iel]]+=swarm_phi[ip] ; count_nodal[icon[1,iel]]+=1          
           #lower left corner - node 2
           if swarm_r[ip]>=0 and swarm_t[ip]>=0:
              phi_nodal[icon[2,iel]]+=swarm_phi[ip] ; count_nodal[icon[2,iel]]+=1          
           #lower left corner - node 3
           if swarm_r[ip]<=0 and swarm_t[ip]>=0:
              phi_nodal[icon[3,iel]]+=swarm_phi[ip] ; count_nodal[icon[3,iel]]+=1          
           #end for
       #end for
       phi_nodal/=count_nodal

    elif averaging=='geometric':
       for ip in range(0,nparticle):
           iel=swarm_iel[ip]
           #lower left corner - node 0
           if swarm_r[ip]<=0 and swarm_t[ip]<=0:
              phi_nodal[icon[0,iel]]+=np.log10(swarm_phi[ip]) ; count_nodal[icon[0,iel]]+=1          
           #lower left corner - node 1
           if swarm_r[ip]>=0 and swarm_t[ip]<=0:
              phi_nodal[icon[1,iel]]+=np.log10(swarm_phi[ip]) ; count_nodal[icon[1,iel]]+=1          
           #lower left corner - node 2
           if swarm_r[ip]>=0 and swarm_t[ip]>=0:
              phi_nodal[icon[2,iel]]+=np.log10(swarm_phi[ip]) ; count_nodal[icon[2,iel]]+=1          
           #lower left corner - node 3
           if swarm_r[ip]<=0 and swarm_t[ip]>=0:
              phi_nodal[icon[3,iel]]+=np.log10(swarm_phi[ip]) ; count_nodal[icon[3,iel]]+=1          
           #end for
       #end for
       phi_nodal=10**(phi_nodal/count_nodal)

    elif averaging=='harmonic':
       for ip in range(0,nparticle):
           iel=swarm_iel[ip]
           #lower left corner - node 0
           if swarm_r[ip]<=0 and swarm_t[ip]<=0:
              phi_nodal[icon[0,iel]]+=1./swarm_phi[ip] ; count_nodal[icon[0,iel]]+=1          
           #lower left corner - node 1
           if swarm_r[ip]>=0 and swarm_t[ip]<=0:
              phi_nodal[icon[1,iel]]+=1./swarm_phi[ip] ; count_nodal[icon[1,iel]]+=1          
           #lower left corner - node 2
           if swarm_r[ip]>=0 and swarm_t[ip]>=0:
              phi_nodal[icon[2,iel]]+=1./swarm_phi[ip] ; count_nodal[icon[2,iel]]+=1          
           #lower left corner - node 3
           if swarm_r[ip]<=0 and swarm_t[ip]>=0:
              phi_nodal[icon[3,iel]]+=1./swarm_phi[ip] ; count_nodal[icon[3,iel]]+=1          
           #end for
       #end for
       phi_nodal=count_nodal/phi_nodal

    # (implicitely) use Q1 basis functions to compute field at other nodes
    for iel in range(0,nel):
        phi_nodal[icon[4,iel]]=0.50*(phi_nodal[icon[0,iel]]+phi_nodal[icon[1,iel]])
        phi_nodal[icon[5,iel]]=0.50*(phi_nodal[icon[1,iel]]+phi_nodal[icon[2,iel]])
        phi_nodal[icon[6,iel]]=0.50*(phi_nodal[icon[2,iel]]+phi_nodal[icon[3,iel]])
        phi_nodal[icon[7,iel]]=0.50*(phi_nodal[icon[0,iel]]+phi_nodal[icon[3,iel]])
        phi_nodal[icon[8,iel]]=0.25*(phi_nodal[icon[0,iel]]+phi_nodal[icon[1,iel]]\
                                    +phi_nodal[icon[2,iel]]+phi_nodal[icon[3,iel]])

    return phi_nodal

###################################################################################################
# this version of the algorithm uses the particles around a given node, but only those 
# in the corresponding quadrants of each enighouring element, with Q1 weighing
# Be careful that count_nodal is nn_V long, but only the subset of Q1 nodes are assigned 
# a value, so that the min value of that array is always zero. 
###################################################################################################

@numba.njit
def project_particle_field_on_nodes_4(nel,nn_V,nparticle,swarm_phi,icon,swarm_iel,
                                      swarm_r,swarm_t,averaging):
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
       phi_nodal: field phi on V nodes 
    """

    phi_nodal=np.zeros(nn_V,dtype=np.float64) 
    count_nodal=np.zeros(nn_V,dtype=np.float64) 

    if averaging=='arithmetic':
       for ip in range(0,nparticle):
           iel=swarm_iel[ip]
           #lower left corner - node 0
           if swarm_r[ip]<=0 and swarm_t[ip]<=0:
              N=-swarm_r[ip]*-swarm_t[ip]
              phi_nodal[icon[0,iel]]+=swarm_phi[ip]*N ; count_nodal[icon[0,iel]]+=1*N          
           #lower left corner - node 1
           if swarm_r[ip]>=0 and swarm_t[ip]<=0:
              N=(1-swarm_r[ip])*-swarm_t[ip]
              phi_nodal[icon[1,iel]]+=swarm_phi[ip]*N ; count_nodal[icon[1,iel]]+=1*N          
           #lower left corner - node 2
           if swarm_r[ip]>=0 and swarm_t[ip]>=0:
              N=(1-swarm_r[ip])*(1-swarm_t[ip])
              phi_nodal[icon[2,iel]]+=swarm_phi[ip]*N ; count_nodal[icon[2,iel]]+=1*N          
           #lower left corner - node 3
           if swarm_r[ip]<=0 and swarm_t[ip]>=0:
              N=-swarm_r[ip]*(1-swarm_t[ip])
              phi_nodal[icon[3,iel]]+=swarm_phi[ip]*N ; count_nodal[icon[3,iel]]+=1*N
           #end for
       #end for
       phi_nodal/=count_nodal

    elif averaging=='geometric':
       for ip in range(0,nparticle):
           iel=swarm_iel[ip]
           #lower left corner - node 0
           if swarm_r[ip]<=0 and swarm_t[ip]<=0:
              N=-swarm_r[ip]*-swarm_t[ip]
              phi_nodal[icon[0,iel]]+=np.log10(swarm_phi[ip])*N ; count_nodal[icon[0,iel]]+=1*N
           #lower left corner - node 1
           if swarm_r[ip]>=0 and swarm_t[ip]<=0:
              N=(1-swarm_r[ip])*-swarm_t[ip]
              phi_nodal[icon[1,iel]]+=np.log10(swarm_phi[ip])*N ; count_nodal[icon[1,iel]]+=1*N          
           #lower left corner - node 2
           if swarm_r[ip]>=0 and swarm_t[ip]>=0:
              N=(1-swarm_r[ip])*(1-swarm_t[ip])
              phi_nodal[icon[2,iel]]+=np.log10(swarm_phi[ip])*N ; count_nodal[icon[2,iel]]+=1*N          
           #lower left corner - node 3
           if swarm_r[ip]<=0 and swarm_t[ip]>=0:
              N=-swarm_r[ip]*(1-swarm_t[ip])
              phi_nodal[icon[3,iel]]+=np.log10(swarm_phi[ip])*N ; count_nodal[icon[3,iel]]+=1*N          
           #end for
       #end for
       phi_nodal=10**(phi_nodal/count_nodal)

    elif averaging=='harmonic':
       for ip in range(0,nparticle):
           iel=swarm_iel[ip]
           #lower left corner - node 0
           if swarm_r[ip]<=0 and swarm_t[ip]<=0:
              N=-swarm_r[ip]*-swarm_t[ip]
              phi_nodal[icon[0,iel]]+=1./swarm_phi[ip]*N ; count_nodal[icon[0,iel]]+=1*N          
           #lower left corner - node 1
           if swarm_r[ip]>=0 and swarm_t[ip]<=0:
              N=(1-swarm_r[ip])*-swarm_t[ip]
              phi_nodal[icon[1,iel]]+=1./swarm_phi[ip]*N ; count_nodal[icon[1,iel]]+=1*N          
           #lower left corner - node 2
           if swarm_r[ip]>=0 and swarm_t[ip]>=0:
              N=(1-swarm_r[ip])*(1-swarm_t[ip])
              phi_nodal[icon[2,iel]]+=1./swarm_phi[ip]*N ; count_nodal[icon[2,iel]]+=1*N          
           #lower left corner - node 3
           if swarm_r[ip]<=0 and swarm_t[ip]>=0:
              N=-swarm_r[ip]*(1-swarm_t[ip])
              phi_nodal[icon[3,iel]]+=1./swarm_phi[ip]*N ; count_nodal[icon[3,iel]]+=1*N          
           #end for
       #end for
       phi_nodal=count_nodal/phi_nodal

    # (implicitely) use Q1 basis functions to compute field at other nodes
    for iel in range(0,nel):
        phi_nodal[icon[4,iel]]=0.50*(phi_nodal[icon[0,iel]]+phi_nodal[icon[1,iel]])
        phi_nodal[icon[5,iel]]=0.50*(phi_nodal[icon[1,iel]]+phi_nodal[icon[2,iel]])
        phi_nodal[icon[6,iel]]=0.50*(phi_nodal[icon[2,iel]]+phi_nodal[icon[3,iel]])
        phi_nodal[icon[7,iel]]=0.50*(phi_nodal[icon[0,iel]]+phi_nodal[icon[3,iel]])
        phi_nodal[icon[8,iel]]=0.25*(phi_nodal[icon[0,iel]]+phi_nodal[icon[1,iel]]\
                                    +phi_nodal[icon[2,iel]]+phi_nodal[icon[3,iel]])

    return phi_nodal


###################################################################################################
# These two functions were written by G. Mack
###################################################################################################

def sign(n):
    if n < 0:
        return -1.0
    return 1.0

def limiter(c_0, c_1, c_2, S = 0, L=1, h=1):
    """
    Limits a the slopes of a plane in 2D
    See latex for a complete explanation
    float- c_0, c_1, c_2 should be given by a least squares approximation.
    float - S is the lowest allowed property value
    float - L is the highest allowed property value
    float - h is the width of the cell
    """
    k_u = min(c_0 - S, L-c_0)
    k_l = (abs(c_1) + abs(c_2)) * h/2
    if k_l > k_u:
        c_max = k_u *2/h
        c_1 = sign(c_1) * min(abs(c_1), c_max)
        c_2 = sign(c_2) * min(abs(c_2), c_max)
    k_l = (abs(c_1) + abs(c_2)) * h/2
    if k_l > k_u:
        c_change = (k_l - k_u)/h # denominator is written up as 2*h/2
        c_1 = sign(c_1) * (abs(c_1) - c_change)
        c_2 = sign(c_2) * (abs(c_2) - c_change)
    return c_1, c_2

###################################################################################################
# least square process
# for each element I loop over the particles in it and build the corresponding A matrix and b rhs. 
# The solution consists of the coefficients a,b,c for viscosity, d,e,f for density, for each elt. 
###################################################################################################

@numba.njit
def compute_ls_coefficients(nel,x_e,z_e,swarm_x,swarm_z,swarm_iel,swarm_rho,swarm_eta):
    """
    Args:
       nel: number of elements
       x_e,z_e: coordinates of elements center
       swarm_x,swarm_z: coordinates of particles
       swarm_rho,swarm_eta: density, viscosity on particles 
       swarm_iel: cell index of all particles
    Returns:
       ls_rho_a,ls_rho_n,ls_rho_c: coeffs for ls density linear fit
       ls_eta_a,ls_eta_n,ls_eta_c: coeffs for ls viscosity linear fit
    """

    ls_rho_a=np.zeros(nel,dtype=np.float64) 
    ls_rho_b=np.zeros(nel,dtype=np.float64) 
    ls_rho_c=np.zeros(nel,dtype=np.float64) 
    ls_eta_a=np.zeros(nel,dtype=np.float64) 
    ls_eta_b=np.zeros(nel,dtype=np.float64) 
    ls_eta_c=np.zeros(nel,dtype=np.float64) 
    rho_min_e=np.zeros(nel,dtype=np.float64) 
    rho_max_e=np.zeros(nel,dtype=np.float64) 
    eta_min_e=np.zeros(nel,dtype=np.float64) 
    eta_max_e=np.zeros(nel,dtype=np.float64) 

    for iel in range(0,nel):
        A_ls=np.zeros((3,3),dtype=np.float64)
        rhs_eta_ls=np.zeros((3),dtype=np.float64)
        rhs_rho_ls=np.zeros((3),dtype=np.float64)

        mask=(swarm_iel==iel) 
        s_eta=swarm_eta[mask]
        s_rho=swarm_rho[mask]
        s_x=swarm_x[mask]-x_e[iel]
        s_z=swarm_z[mask]-z_e[iel]
        nb_ptcls=np.count_nonzero(mask)

        A_ls[0,0]=nb_ptcls
        for ip in range(0,nb_ptcls):        
            A_ls[0,1]+=s_x[ip]
            A_ls[0,2]+=s_z[ip]
            A_ls[1,1]+=s_x[ip]**2
            A_ls[1,2]+=s_x[ip]*s_z[ip]
            A_ls[2,2]+=s_z[ip]**2
            rhs_eta_ls[0]+=s_eta[ip]
            rhs_eta_ls[1]+=s_eta[ip]*s_x[ip]
            rhs_eta_ls[2]+=s_eta[ip]*s_z[ip]
            rhs_rho_ls[0]+=s_rho[ip]
            rhs_rho_ls[1]+=s_rho[ip]*s_x[ip]
            rhs_rho_ls[2]+=s_rho[ip]*s_z[ip]
        #end for ip
        A_ls[1,0]=A_ls[0,1]
        A_ls[2,0]=A_ls[0,2]
        A_ls[2,1]=A_ls[1,2]
        sol=np.linalg.solve(A_ls,rhs_eta_ls)
        ls_eta_a[iel]=sol[0]
        ls_eta_b[iel]=sol[1]
        ls_eta_c[iel]=sol[2]
        sol=np.linalg.solve(A_ls,rhs_rho_ls)
        ls_rho_a[iel]=sol[0]
        ls_rho_b[iel]=sol[1]
        ls_rho_c[iel]=sol[2]

        rho_min_e[iel]=np.min(s_rho) ; rho_max_e[iel]=np.max(s_rho)
        eta_min_e[iel]=np.min(s_eta) ; eta_max_e[iel]=np.max(s_eta)

    #end for iel

    return ls_rho_a,ls_rho_b,ls_rho_c,ls_eta_a,ls_eta_b,ls_eta_c,rho_min_e,rho_max_e,eta_min_e,eta_max_e

###################################################################################################

def output_fields_ls(istep,nel,x,z,icon,x_e,z_e,ls_rho_a,ls_rho_b,ls_rho_c,ls_eta_a,ls_eta_b,ls_eta_c):

       filename = 'OUTPUT/fields_ls_{:04d}.vtu'.format(istep) 
       vtufile=open(filename,"w")
       vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
       vtufile.write("<UnstructuredGrid> \n")
       vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(4*nel,nel))
       #####
       vtufile.write("<Points> \n")
       vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
       for iel in range(0,nel):
           vtufile.write("%.4e %.1e %.4e \n" %(x[icon[0,iel]],0.,z[icon[0,iel]]))
           vtufile.write("%.4e %.1e %.4e \n" %(x[icon[1,iel]],0.,z[icon[1,iel]]))
           vtufile.write("%.4e %.1e %.4e \n" %(x[icon[2,iel]],0.,z[icon[2,iel]]))
           vtufile.write("%.4e %.1e %.4e \n" %(x[icon[3,iel]],0.,z[icon[3,iel]]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</Points> \n")
       #--
       vtufile.write("<CellData Scalars='scalars'>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_rho_a' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_rho_a[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_rho_b' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_rho_b[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_rho_c' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_rho_c[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_eta_a' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_eta_a[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_eta_b' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_eta_b[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("<DataArray type='Float32' Name='ls_eta_c' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_eta_c[iel]))
       vtufile.write("</DataArray>\n")
       vtufile.write("</CellData>\n")
       vtufile.write("<PointData Scalars='scalars'>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Density' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_rho_a[iel]+ls_rho_b[iel]*(x[icon[0,iel]]-x_e[iel])+ls_rho_c[iel]*(z[icon[0,iel]]-z_e[iel]) ))
           vtufile.write("%e\n" % (ls_rho_a[iel]+ls_rho_b[iel]*(x[icon[1,iel]]-x_e[iel])+ls_rho_c[iel]*(z[icon[1,iel]]-z_e[iel]) ))
           vtufile.write("%e\n" % (ls_rho_a[iel]+ls_rho_b[iel]*(x[icon[2,iel]]-x_e[iel])+ls_rho_c[iel]*(z[icon[2,iel]]-z_e[iel]) ))
           vtufile.write("%e\n" % (ls_rho_a[iel]+ls_rho_b[iel]*(x[icon[3,iel]]-x_e[iel])+ls_rho_c[iel]*(z[icon[3,iel]]-z_e[iel]) ))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Float32' Name='Viscosity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%e\n" % (ls_eta_a[iel]+ls_eta_b[iel]*(x[icon[0,iel]]-x_e[iel])+ls_eta_c[iel]*(z[icon[0,iel]]-z_e[iel]) ))
           vtufile.write("%e\n" % (ls_eta_a[iel]+ls_eta_b[iel]*(x[icon[1,iel]]-x_e[iel])+ls_eta_c[iel]*(z[icon[1,iel]]-z_e[iel]) ))
           vtufile.write("%e\n" % (ls_eta_a[iel]+ls_eta_b[iel]*(x[icon[2,iel]]-x_e[iel])+ls_eta_c[iel]*(z[icon[2,iel]]-z_e[iel]) ))
           vtufile.write("%e\n" % (ls_eta_a[iel]+ls_eta_b[iel]*(x[icon[3,iel]]-x_e[iel])+ls_eta_c[iel]*(z[icon[3,iel]]-z_e[iel]) ))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</PointData>\n")
       #####
       vtufile.write("<Cells>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d %d %d %d\n" %(iel*4,iel*4+1,iel*4+2,iel*4+3))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %((iel+1)*4))
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
       for iel in range (0,nel):
           vtufile.write("%d \n" %5)
       vtufile.write("</DataArray>\n")
       #--
       vtufile.write("</Cells>\n")
       #####
       vtufile.write("</Piece>\n")
       vtufile.write("</UnstructuredGrid>\n")
       vtufile.write("</VTKFile>\n")
       vtufile.close()

###################################################################################################
