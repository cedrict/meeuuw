import numpy as np
import numba

###############################################################################

@numba.njit
def compute_nodal_pressure_gradient(icon_V,q,nn_V,m_V,nel,dNdx_V_n,dNdy_V_n):

    count=np.zeros(nn_V,dtype=np.float64)  
    dpdx_n=np.zeros(nn_V,dtype=np.float64)  
    dpdy_n=np.zeros(nn_V,dtype=np.float64)  

    for iel in range(0,nel):
        for i in range(0,m_V):
            inode=icon_V[i,iel]
            dpdx_n[inode]+=np.dot(dNdx_V_n[i,:],q[icon_V[:,iel]])
            dpdy_n[inode]+=np.dot(dNdy_V_n[i,:],q[icon_V[:,iel]])
            count[inode]+=1
        #end for
    #end for
    dpdx_n/=count
    dpdy_n/=count

    return dpdx_n,dpdy_n

###############################################################################
