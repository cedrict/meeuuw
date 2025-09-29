import numpy as np
import numba
from basis_functions import basis_functions_V

# jit on these functions really makes a difference - keep it

#todo remove xmin,ymin

###############################################################################
# this function is used in the Runge-Kutta algorithm. As such it needs to
# locate the particle before interpolating the velocity onto it.
###############################################################################

@numba.njit
def interpolate_vel_on_pt(xp,yp,u,v,hx,hy,nelx,nely,icon_V,x_V,y_V):
    ielx=int(xp/hx)
    iely=int(yp/hy)
    #if ielx<0: exit('ielx<0')
    #if iely<0: exit('iely<0')
    #if ielx>=nelx: exit('ielx>nelx')
    #if iely>=nely: exit('iely>nely')
    iel=nelx*iely+ielx
    xmin=x_V[icon_V[0,iel]] 
    ymin=y_V[icon_V[0,iel]] 
    rp=((xp-xmin)/hx-0.5)*2
    sp=((yp-ymin)/hy-0.5)*2
    N=basis_functions_V(rp,sp)
    up=np.dot(N,u[icon_V[:,iel]])
    vp=np.dot(N,v[icon_V[:,iel]])
    return up,vp,iel

###############################################################################

@numba.njit
def interpolate_field_on_particle(rp,sp,iel,phi,icon):
    N=basis_functions_V(rp,sp)
    phip=np.dot(N,phi[icon[:,iel]])
    return phip

@numba.njit
def interpolate_field_on_particles(nparticle,swarm_r,swarm_s,swarm_iel,phi,icon):

    swarm_field=np.zeros(nparticle,dtype=np.float64)

    for ip in range(0,nparticle):
        N=basis_functions_V(swarm_r[ip],swarm_s[ip])
        swarm_field[ip]=np.dot(N,phi[icon[:,swarm_iel[ip]]])

    return swarm_field

###############################################################################

#@numba.njit
#def locate_pt(xp,yp,hx,hy,x_V,y_V,icon_V,nelx):
#    ielx=int(xp/hx)
#    iely=int(yp/hy)
#    iel=nelx*iely+ielx
#    xmin=x_V[icon_V[0,iel]] 
#    ymin=y_V[icon_V[0,iel]] 
#    rp=((xp-xmin)/hx-0.5)*2
#    sp=((yp-ymin)/hy-0.5)*2
#    return rp,sp,iel

###############################################################################

@numba.njit
def locate_particles(nparticle,swarm_x,swarm_y,hx,hy,x_V,y_V,icon_V,nelx):

    swarm_r=np.zeros(nparticle,dtype=np.float64)
    swarm_s=np.zeros(nparticle,dtype=np.float64)
    swarm_iel=np.zeros(nparticle,dtype=np.int32)

    for ip in range(0,nparticle):
        ielx=int(swarm_x[ip]/hx)
        iely=int(swarm_y[ip]/hy)
        iel=nelx*iely+ielx
        xmin=x_V[icon_V[0,iel]] 
        ymin=y_V[icon_V[0,iel]] 
        swarm_r[ip]=((swarm_x[ip]-xmin)/hx-0.5)*2
        swarm_s[ip]=((swarm_y[ip]-ymin)/hy-0.5)*2
        swarm_iel[ip]=iel

    return swarm_r,swarm_s,swarm_iel

###############################################################################

@numba.njit
def advect_particles(RKorder,dt,nparticle,swarm_x,swarm_y,swarm_active,u,v,\
                     Lx,Ly,hx,hy,nelx,nely,icon_V,x_V,y_V):

    swarm_u=np.zeros(nparticle,dtype=np.float64)
    swarm_v=np.zeros(nparticle,dtype=np.float64)

    if RKorder==1:

       for ip in range(0,nparticle):
           if swarm_active[ip]:
              swarm_u[ip],swarm_v[ip],iel=interpolate_vel_on_pt(swarm_x[ip],swarm_y[ip],u,v,\
                                                                hx,hy,nelx,nely,icon_V,x_V,y_V)
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
              uA,vA,iel=interpolate_vel_on_pt(xA,yA,u,v,\
                                              hx,hy,nelx,nely,icon_V,x_V,y_V)
              xB=xA+uA*dt/2.
              yB=yA+vA*dt/2.
              if xB<0 or xB>Lx or yB<0 or yB>Ly:
                 swarm_active[ip]=False
              else:
                 uB,vB,iel=interpolate_vel_on_pt(xB,yB,u,v,\
                                                 hx,hy,nelx,nely,icon_V,x_V,y_V)
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
              uA,vA,iel=interpolate_vel_on_pt(xA,yA,u,v,hx,hy,nelx,nely,icon_V,x_V,y_V)
              xB=xA+uA*dt/2.
              yB=yA+vA*dt/2.
              if xB<0 or xB>Lx or yB<0 or yB>Ly:
                 swarm_active[ip]=False
              else:
                 uB,vB,iel=interpolate_vel_on_pt(xB,yB,u,v,hx,hy,nelx,nely,icon_V,x_V,y_V)
                 xC=xA+uB*dt/2.
                 yC=yA+vB*dt/2.
                 if xC<0 or xC>Lx or yC<0 or yC>Ly:
                    swarm_active[ip]=False
                 else:
                    uC,vC,iel=interpolate_vel_on_pt(xC,yC,u,v,hx,hy,nelx,nely,icon_V,x_V,y_V)
                    xD=xA+uC*dt
                    yD=yA+vC*dt
                    if xD<0 or xD>Lx or yD<0 or yD>Ly:
                       swarm_active[ip]=False
                    else:
                       uD,vD,iel=interpolate_vel_on_pt(xD,yD,u,v,hx,hy,nelx,nely,icon_V,x_V,y_V)
                       swarm_u[ip]=(uA+2*uB+2*uC+uD)/6
                       swarm_v[ip]=(vA+2*vB+2*vC+vD)/6
                       swarm_x[ip]=xA+swarm_u[ip]*dt
                       swarm_y[ip]=yA+swarm_v[ip]*dt
                    # end if active
                 # end if active
              # end if active
           # end if active
       # end for im

    #else:
    #   exit('RKorder not available')

    for im in range(0,nparticle):
        if not swarm_active[im]:
           swarm_x[im]=0
           swarm_y[im]=0

    return swarm_x,swarm_y,swarm_u,swarm_v,swarm_active

###############################################################################

@numba.njit
def project_particles_on_elements(nel,nparticle,swarm_rho,swarm_eta,swarm_iel):

    rho_elemental=np.zeros(nel,dtype=np.float64) 
    eta_elemental=np.zeros(nel,dtype=np.float64) 
    nparticle_elemental=np.zeros(nel,dtype=np.float64) 

    for ip in range(0,nparticle):
        iel=swarm_iel[ip]
        rho_elemental[iel]+=swarm_rho[ip] # arithmetic 
        eta_elemental[iel]+=swarm_eta[ip] # arithmetic 
        nparticle_elemental[iel]+=1

    #if np.min(nparticle_elemental)<=0: exit('Abort: element without particle')

    rho_elemental/=nparticle_elemental
    eta_elemental/=nparticle_elemental

    return rho_elemental,eta_elemental,nparticle_elemental

###############################################################################

@numba.njit
def project_particles_on_nodes(nel,nn_V,nparticle,swarm_rho,swarm_eta,icon_V,swarm_iel):

    rho_nodal=np.zeros(nn_V,dtype=np.float64) 
    eta_nodal=np.zeros(nn_V,dtype=np.float64) 
    count_nodal=np.zeros(nn_V,dtype=np.float64) 

    for ip in range(0,nparticle):
        iel=swarm_iel[ip]
        for k in (0,1,2,3):
            rho_nodal[icon_V[k,iel]]+=swarm_rho[ip] # arithmetic 
            eta_nodal[icon_V[k,iel]]+=swarm_eta[ip] # arithmetic 
            count_nodal[icon_V[k,iel]]+=1

    for i in range(0,nn_V):
        if count_nodal[i]!=0:
            rho_nodal[i]/=count_nodal[i]
            eta_nodal[i]/=count_nodal[i]

    for iel in range(0,nel):
        rho_nodal[icon_V[4,iel]]=0.5*(rho_nodal[icon_V[0,iel]]+rho_nodal[icon_V[1,iel]])
        rho_nodal[icon_V[5,iel]]=0.5*(rho_nodal[icon_V[1,iel]]+rho_nodal[icon_V[2,iel]])
        rho_nodal[icon_V[6,iel]]=0.5*(rho_nodal[icon_V[2,iel]]+rho_nodal[icon_V[3,iel]])
        rho_nodal[icon_V[7,iel]]=0.5*(rho_nodal[icon_V[0,iel]]+rho_nodal[icon_V[3,iel]])
        rho_nodal[icon_V[8,iel]]=0.25*(rho_nodal[icon_V[0,iel]]+rho_nodal[icon_V[1,iel]]\
                                      +rho_nodal[icon_V[2,iel]]+rho_nodal[icon_V[3,iel]])
        eta_nodal[icon_V[4,iel]]=0.5*(eta_nodal[icon_V[0,iel]]+eta_nodal[icon_V[1,iel]])
        eta_nodal[icon_V[5,iel]]=0.5*(eta_nodal[icon_V[1,iel]]+eta_nodal[icon_V[2,iel]])
        eta_nodal[icon_V[6,iel]]=0.5*(eta_nodal[icon_V[2,iel]]+eta_nodal[icon_V[3,iel]])
        eta_nodal[icon_V[7,iel]]=0.5*(eta_nodal[icon_V[0,iel]]+eta_nodal[icon_V[3,iel]])
        eta_nodal[icon_V[8,iel]]=0.25*(eta_nodal[icon_V[0,iel]]+eta_nodal[icon_V[1,iel]]\
                                      +eta_nodal[icon_V[2,iel]]+eta_nodal[icon_V[3,iel]])

    return rho_nodal,eta_nodal

###############################################################################
