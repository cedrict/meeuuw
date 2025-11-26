import numpy as np
import numba
from constants import *

###############################################################################

@numba.njit
def compute_gravity_at_point(xs,ys,nel,xc,yc,rho,vol,rho_ref):
    """
    Args:
     xs,ys: coordinates of the 'satellite' point where gravity is computed
     nel: number of elements in the mesh
     xc,yc: coordinates of all element centers
     rho: density of all elements
     vol: volume of all elements
     rho_ref: background reference density
    Returns:
     gx,gy: gravity vector at satellite point
     gnorm: gravity norm at satellite point
    """

    ggx=0.
    ggy=0.
    for iel in range(0,nel):
        xx=xs-xc[iel]
        yy=ys-yc[iel]
        rr=np.sqrt(xx**2+yy**2)
        ggx+=Ggrav*(rho[iel]-rho_ref)*vol[iel]*xx/rr**3
        ggy+=Ggrav*(rho[iel]-rho_ref)*vol[iel]*yy/rr**3

    gnorm=np.sqrt(ggx**2+ggy**2)

    return ggx,ggy,gnorm

###############################################################################

def compute_gravity_at_point_axisymmetric(xs,ys,nel,xc,yc,rho,vol,rho_ref):

    ggx=0.
    ggy=0.

    gnorm=np.sqrt(ggx**2+ggy**2)

    return ggx,ggy,gnorm

###############################################################################
