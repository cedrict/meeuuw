import numpy as np
import numba

###################################################################################################

@numba.njit
def compute_nodal_strain_rate(icon_V,u,v,nn_V,m_V,nel,dNdr_V_n,dNds_V_n,jcbi00n,jcbi01n,jcbi10n,jcbi11n):

    count=np.zeros(nn_V,dtype=np.float64)  
    exx_n=np.zeros(nn_V,dtype=np.float64)  
    eyy_n=np.zeros(nn_V,dtype=np.float64)  
    exy_n=np.zeros(nn_V,dtype=np.float64)  

    for iel in range(0,nel):
        for i in range(0,m_V):
            inode=icon_V[i,iel]
            dNdx=jcbi00n[iel,i]*dNdr_V_n[i,:]+jcbi01n[iel,i]*dNds_V_n[i,:]
            dNdy=jcbi10n[iel,i]*dNdr_V_n[i,:]+jcbi11n[iel,i]*dNds_V_n[i,:]
            exx_n[inode]+=np.dot(dNdx,u[icon_V[:,iel]])
            eyy_n[inode]+=np.dot(dNdy,v[icon_V[:,iel]])
            exy_n[inode]+=0.5*np.dot(dNdx,v[icon_V[:,iel]])+\
                          0.5*np.dot(dNdy,u[icon_V[:,iel]])
            count[inode]+=1
        #end for
    #end for
    exx_n/=count
    eyy_n/=count
    exy_n/=count

    e_n=np.sqrt(0.5*(exx_n**2+eyy_n**2)+exy_n**2)

    return exx_n,eyy_n,exy_n,e_n

###################################################################################################
