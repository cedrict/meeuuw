import numpy as np
from constants import *

def compute_gravity_at_point(xs,ys,nel,xc,yc,rho,hx,hy,rho_ref):
    # xs,ys: coordinates of the 'satellite' point
    # where gravity is computed

    vol=hx*hy

    ggx=0.
    ggy=0.
    for iel in range(0,nel):
        xx=xs-xc[iel]
        yy=ys-yc[iel]
        rr=np.sqrt(xx**2+yy**2)
        ggx+=Ggrav*(rho[iel]-rho_ref)*vol*xx/rr**3
        ggy+=Ggrav*(rho[iel]-rho_ref)*vol*yy/rr**3

    gnorm=np.sqrt(ggx**2+ggy**2)

    return ggx,ggy,gnorm
