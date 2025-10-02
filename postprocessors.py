import numpy as np
import numba

###############################################################################

@numba.njit
def global_quantities(nel,nqel,xq,yq,uq,vq,Tq,rhoq,hcapaq,etaq,exxq,eyyq,exyq,Lx,Ly,JxWq,gy):

    TM=0  # Total mass
    EK=0  # Kinetic Energy
    WAG=0 # Work against gravity
    TVD=0 # Total viscous dissipation
    GPE=0 # Gravitational potential energy
    ITE=0 # Internal thermal energy
    vrms=0 # root mean square velocity

    for iel in range(0,nel):
        for iq in range(0,nqel):
            TM+=rhoq[iel,iq]                                                        *JxWq[iq]
            EK+=0.5*rhoq[iel,iq]*(uq[iel,iq]**2+vq[iel,iq]**2)                      *JxWq[iq]
            WAG-=rhoq[iel,iq]*vq[iel,iq]*gy                                         *JxWq[iq]
            TVD+=2*etaq[iel,iq]*(exxq[iel,iq]**2+eyyq[iel,iq]**2+2*exyq[iel,iq]**2) *JxWq[iq]
            GPE+=rhoq[iel,iq]*gy*(Ly-yq[iel,iq])                                    *JxWq[iq]
            ITE+=rhoq[iel,iq]*hcapaq[iel,iq]*Tq[iel,iq]                             *JxWq[iq]
            vrms+=(uq[iel,iq]**2+vq[iel,iq]**2)                                     *JxWq[iq]
        #end for iq
    #end for iel
    vrms=np.sqrt(vrms/(Lx*Ly)) 

    return vrms,EK,WAG,TVD,GPE,ITE

###############################################################################
