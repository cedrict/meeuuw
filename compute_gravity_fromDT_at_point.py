import numpy as np
import numba
from constants import *


@numba.njit
def compute_gravity_fromDT_at_point(xs,ys,y_ref,nelx,x,rho,dyn_topo,rho_ref):
    # xs,ys: coordinates of the 'satellite' point
    # where gravity is computed

    ggx=0.
    ggy=0.
    for i in range(1,2*nelx+1+1):

        rhoo=(rho[i-1]+rho[i])/2-rho_ref
        vol=(x[i]-x[i-1])*(dyn_topo[i-1]+dyn_topo[i])/2  #signed volume!
        xc=(x[i-1]+x[i])/2
        yc=y_ref+(dyn_topo[i-1]+dyn_topo[i])/2

        #print(xc,yc,vol,rhoo)

        xx=xs-xc
        yy=ys-yc
        rr=np.sqrt(xx**2+yy**2)
        ggx+=Ggrav*rhoo*vol*xx/rr**3
        ggy+=Ggrav*rhoo*vol*yy/rr**3

    gnorm=np.sqrt(ggx**2+ggy**2)

    return ggx,ggy,gnorm
