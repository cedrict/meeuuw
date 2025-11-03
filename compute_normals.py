import numpy as np

###################################################################################################
# Eq 14 of ensg82

def compute_normals(geometry,nel,nn_V,nqel,m_V,icon_V,dNdr_V,dNds_V,\
                    JxWq,hull_nodes,jcbi00q,jcbi01q,jcbi10q,jcbi11q):

    nx=np.zeros(nn_V,dtype=np.float64)
    ny=np.zeros(nn_V,dtype=np.float64)

    for iel in range(0,nel):
        for iq in range(0,nqel):
            dNdx=jcbi00q[iel,iq]*dNdr_V[iq,:]+jcbi01q[iel,iq]*dNds_V[iq,:]
            dNdy=jcbi10q[iel,iq]*dNdr_V[iq,:]+jcbi11q[iel,iq]*dNds_V[iq,:]
            for i in range(0,m_V):
                nx[icon_V[i,iel]]+=dNdx[i]*JxWq[iel,iq]
                ny[icon_V[i,iel]]+=dNdy[i]*JxWq[iel,iq]

    for i in range(0,nn_V):
        if hull_nodes[i]:
           norm=np.sqrt(nx[i]**2+ny[i]**2)
           nx[i]/=norm
           ny[i]/=norm

    return nx,ny

###################################################################################################
