
import numpy as np
import numba
import scipy.sparse as sps
from scipy import sparse
from basis_functions import *

###################################################################################################
# this function computes the strain rate in the middle of each element
###################################################################################################

@numba.njit
def compute_elemental_strain_rate(icon_V,u,v,nn_V,nel,x_V,y_V):

    exx=np.zeros(nel,dtype=np.float64)  
    eyy=np.zeros(nel,dtype=np.float64)  
    exy=np.zeros(nel,dtype=np.float64)  
    jcb=np.zeros((2,2),dtype=np.float64)
    jcbi=np.zeros((2,2),dtype=np.float64)

    rq=sq=00
    dNdr=basis_functions_V_dr(rq,sq)
    dNdt=basis_functions_V_dt(rq,sq)

    for iel in range(0,nel):
        jcb[0,0]=np.dot(dNdr,x_V[icon_V[:,iel]])
        jcb[0,1]=np.dot(dNdr,y_V[icon_V[:,iel]])
        jcb[1,0]=np.dot(dNdt,x_V[icon_V[:,iel]])
        jcb[1,1]=np.dot(dNdt,y_V[icon_V[:,iel]])
        jcbi=np.linalg.inv(jcb)
        dNdx=jcbi[0,0]*dNdr+jcbi[0,1]*dNdt
        dNdy=jcbi[1,0]*dNdr+jcbi[1,1]*dNdt
        exx[iel]=np.dot(dNdx,u[icon_V[:,iel]])
        eyy[iel]=np.dot(dNdy,v[icon_V[:,iel]])
        exy[iel]=np.dot(dNdx,v[icon_V[:,iel]])*0.5+\
                 np.dot(dNdy,u[icon_V[:,iel]])*0.5

    return exx,eyy,exy

###################################################################################################
