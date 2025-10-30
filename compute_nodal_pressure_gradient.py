import numpy as np
import numba

###################################################################################################

@numba.njit
def compute_nodal_pressure_gradient(icon_V,q,nn_V,m_V,nel,dNdr_V_n,dNds_V_n,jcbi00n,jcbi01n,jcbi10n,jcbi11n):

    count=np.zeros(nn_V,dtype=np.float64)  
    dpdx_n=np.zeros(nn_V,dtype=np.float64)  
    dpdy_n=np.zeros(nn_V,dtype=np.float64)  

    for iel in range(0,nel):
        for i in range(0,m_V):
            inode=icon_V[i,iel]
            dNdx=jcbi00n[iel,i]*dNdr_V_n[i,:]+jcbi01n[iel,i]*dNds_V_n[i,:]
            dNdy=jcbi10n[iel,i]*dNdr_V_n[i,:]+jcbi11n[iel,i]*dNds_V_n[i,:]
            dpdx_n[inode]+=np.dot(dNdx,q[icon_V[:,iel]])
            dpdy_n[inode]+=np.dot(dNdy,q[icon_V[:,iel]])
            count[inode]+=1
        #end for
    #end for
    dpdx_n/=count
    dpdy_n/=count

    return dpdx_n,dpdy_n

###################################################################################################
