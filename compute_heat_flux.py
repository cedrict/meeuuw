import numpy as np

###############################################################################

def compute_heat_flux(icon_V,T,hcond,nn_V,m_V,nel,dNdx_V_n,dNdy_V_n):

    qx_n=np.zeros(nn_V,dtype=np.float64)  
    qy_n=np.zeros(nn_V,dtype=np.float64)  
    count=np.zeros(nn_V,dtype=np.float64)  

    for iel in range(0,nel):
           for i in range(0,m_V):
               inode=icon_V[i,iel]
               qx_n[inode]-=np.dot(hcond*dNdx_V_n[i,:],T[icon_V[:,iel]])
               qy_n[inode]-=np.dot(hcond*dNdy_V_n[i,:],T[icon_V[:,iel]])
               count[inode]+=1
           #end for
    #end for
    
    qx_n/=count
    qy_n/=count

    return qx_n,qy_n 

###############################################################################
