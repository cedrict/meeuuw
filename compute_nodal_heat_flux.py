import numpy as np
import numba

###################################################################################################

@numba.njit
def compute_nodal_heat_flux(icon_V,T,hcond_nodal,nn_V,m_V,nel,dNdr_V_n,dNds_V_n,jcbi00n,jcbi01n,jcbi10n,jcbi11n):

    qx_n=np.zeros(nn_V,dtype=np.float64)  
    qy_n=np.zeros(nn_V,dtype=np.float64)  
    dTdx_n=np.zeros(nn_V,dtype=np.float64)  
    dTdy_n=np.zeros(nn_V,dtype=np.float64)  
    count=np.zeros(nn_V,dtype=np.float64)  

    for iel in range(0,nel):
        for i in range(0,m_V):
            inode=icon_V[i,iel]
            dNdx=jcbi00n[iel,i]*dNdr_V_n[i,:]+jcbi01n[iel,i]*dNds_V_n[i,:]
            dNdy=jcbi10n[iel,i]*dNdr_V_n[i,:]+jcbi11n[iel,i]*dNds_V_n[i,:]
            dTdx_n[inode]-=np.dot(dNdx,T[icon_V[:,iel]])
            dTdy_n[inode]-=np.dot(dNdy,T[icon_V[:,iel]])
            qx_n[inode]-=hcond_nodal[i]*np.dot(dNdx,T[icon_V[:,iel]])
            qy_n[inode]-=hcond_nodal[i]*np.dot(dNdy,T[icon_V[:,iel]])
            count[inode]+=1
        #end for
    #end for
    
    qx_n/=count
    qy_n/=count
    dTdx_n/=count
    dTdy_n/=count

    return dTdx_n,dTdy_n,qx_n,qy_n 

###################################################################################################
