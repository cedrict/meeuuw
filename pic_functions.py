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
def interpolate_vel_on_pt(xm,ym,u,v,hx,hy,nelx,nely,icon_V,x_V,y_V):
    ielx=int(xm/hx)
    iely=int(ym/hy)
    #if ielx<0: exit('ielx<0')
    #if iely<0: exit('iely<0')
    #if ielx>=nelx: exit('ielx>nelx')
    #if iely>=nely: exit('iely>nely')
    iel=nelx*iely+ielx
    xmin=x_V[icon_V[0,iel]] 
    ymin=y_V[icon_V[0,iel]] 
    rm=((xm-xmin)/hx-0.5)*2
    sm=((ym-ymin)/hy-0.5)*2
    N=basis_functions_V(rm,sm)
    um=np.dot(N,u[icon_V[:,iel]])
    vm=np.dot(N,v[icon_V[:,iel]])
    return um,vm,iel

###############################################################################

@numba.njit
def interpolate_field_on_particle(rp,sp,iel,phi,icon):
    N=basis_functions_V(rp,sp)
    phip=np.dot(N,phi[icon[:,iel]])
    return phip

###############################################################################

@numba.njit
def locate_pt(xp,yp,hx,hy,x_V,y_V,icon_V,nelx):
    ielx=int(xp/hx)
    iely=int(yp/hy)
    iel=nelx*iely+ielx
    xmin=x_V[icon_V[0,iel]] 
    ymin=y_V[icon_V[0,iel]] 
    rm=((xp-xmin)/hx-0.5)*2
    sm=((yp-ymin)/hy-0.5)*2
    return rm,sm,iel

###############################################################################
