import numpy as np
import numba
from basis_functions import *

###############################################################################

@numba.njit
def global_quantities(nel,nq_per_element,xq,zq,uq,wq,Tq,rhoq,hcapaq,etaq,exxq,ezzq,exzq,volume,JxWq,gxq,gzq):

    TM=0  # Total mass
    EK=0  # Kinetic Energy
    WAG=0 # Work against gravity
    TVD=0 # Total viscous dissipation
    GPE=0 # Gravitational potential energy
    ITE=0 # Internal thermal energy
    vrms=0 # root mean square velocity

    for iel in range(0,nel):
        for iq in range(0,nq_per_element):
            TM+=rhoq[iel,iq]                                                        *JxWq[iel,iq]
            EK+=0.5*rhoq[iel,iq]*(uq[iel,iq]**2+wq[iel,iq]**2)                      *JxWq[iel,iq]
            WAG-=rhoq[iel,iq]*(uq[iel,iq]*gxq[iel,iq]+wq[iel,iq]*gzq[iel,iq])       *JxWq[iel,iq]
            TVD+=2*etaq[iel,iq]*(exxq[iel,iq]**2+ezzq[iel,iq]**2+2*exzq[iel,iq]**2) *JxWq[iel,iq]
            #GPE+=rhoq[iel,iq]*gzq[iel,iq]*(Lz-zq[iel,iq])                           *JxWq[iel,iq]
            ITE+=rhoq[iel,iq]*hcapaq[iel,iq]*Tq[iel,iq]                             *JxWq[iel,iq]
            vrms+=(uq[iel,iq]**2+wq[iel,iq]**2)                                     *JxWq[iel,iq]
        #end for iq
    #end for iel
    vrms=np.sqrt(vrms/volume) 

    return vrms,EK,WAG,TVD,GPE,ITE,TM

###############################################################################

def compute_Nu(Lx,Lz,nel,top_element,bottom_element,icon_V,T,dTdy_nodal,\
               nq_per_dim,qcoords,qweights,hx):

    avrg_T_top=0    ; avrg_dTdy_top=0    
    avrg_T_bottom=0 ; avrg_dTdy_bottom=0 

    jcob=hx/2

    for iel in range(0,nel):

        if top_element[iel]: 
           sq=+1
           ny=+1
           for iq in range(0,nq_per_dim):
               rq=qcoords[iq]
               N=basis_functions_V(rq,sq)
               Tq=np.dot(N,T[icon_V[:,iel]])
               dTdyq=np.dot(N,dTdy_nodal[icon_V[:,iel]])
               avrg_T_top+=Tq*jcob*qweights[iq]
               avrg_dTdy_top+=dTdyq*jcob*qweights[iq]*ny
           #end for
        #end if

        if bottom_element[iel]: 
           sq=-1
           ny=-1
           for iq in range(0,nq_per_dim):
               rq=qcoords[iq]
               N=basis_functions_V(rq,sq)
               Tq=np.dot(N,T[icon_V[:,iel]])
               dTdyq=np.dot(N,dTdy_nodal[icon_V[:,iel]])
               avrg_T_bottom+=Tq*jcob*qweights[iq]
               avrg_dTdy_bottom+=dTdyq*jcob*qweights[iq]*ny
           #end for
        #end if
    #end for

    avrg_T_top/=Lx
    avrg_T_bottom/=Lx
    avrg_dTdy_top/=Lx
    avrg_dTdy_bottom/=Lx

    Nu=np.abs(avrg_dTdy_top)/avrg_T_bottom*Lz

    return avrg_T_bottom,avrg_T_top,avrg_dTdy_bottom,avrg_dTdy_top,Nu

###############################################################################
