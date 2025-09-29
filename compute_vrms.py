import numpy as np
import numba

###############################################################################

@numba.njit
def compute_vrms(nel,nqel,weightq,icon_V,u,v,N_V,Lx,Ly,jcob):

    vrms=0.
    for iel in range(0,nel):
        for iq in range(0,nqel):
            JxWq=jcob*weightq[iq]
            uq=np.dot(N_V[iq,:],u[icon_V[:,iel]])
            vq=np.dot(N_V[iq,:],v[icon_V[:,iel]])
            vrms+=(uq**2+vq**2)*JxWq
        #end for iq
    #end for iel
    vrms=np.sqrt(vrms/(Lx*Ly)) 

    return vrms

###############################################################################
