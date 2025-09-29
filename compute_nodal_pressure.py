import numpy as np
import numba

###############################################################################

@numba.njit
def compute_nodal_pressure(m_V,nn_V,icon_V,icon_P,p,N_P_n):

    count=np.zeros(nn_V,dtype=np.int32)  
    q=np.zeros(nn_V,dtype=np.float64)

    for iel,nodes in enumerate(icon_V.T):
        for k in range(0,m_V):
            q[nodes[k]]+=np.dot(N_P_n[k,:],p[icon_P[:,iel]])
            count[nodes[k]]+=1
        #end for
    #end for
    q/=count

    return q

###############################################################################
