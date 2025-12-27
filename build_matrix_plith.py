###################################################################################################
# MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW - MEEUUW
###################################################################################################

import numpy as np
import numba

###################################################################################################
# boundary conditions are hard wired to be p=0 at the top boundary.

@numba.njit
def build_matrix_plith(bignb,nel,nq_per_element,m_T,Nfem_T,icon_V,rhoq,gxq,gyq,JxWq,\
                       N_V,dNdr_V,dNds_V,jcbi00q,jcbi01q,jcbi10q,jcbi11q,top_nodes):
                        
    VV_T=np.zeros(bignb,dtype=np.float64)    
    rhs=np.zeros(Nfem_T,dtype=np.float64)
    B=np.zeros((2,m_T),dtype=np.float64)

    counter=0
    for iel in range(0,nel):
        b_el=np.zeros(m_T,dtype=np.float64)
        A_el=np.zeros((m_T,m_T),dtype=np.float64)
        for iq in range(0,nq_per_element):
            dNdx=jcbi00q[iel,iq]*dNdr_V[iq,:]+jcbi01q[iel,iq]*dNds_V[iq,:]
            dNdy=jcbi10q[iel,iq]*dNdr_V[iq,:]+jcbi11q[iel,iq]*dNds_V[iq,:]
            B[0,:]=dNdx
            B[1,:]=dNdy
            A_el+=B.T.dot(B)*JxWq[iel,iq]
            b_el+=(dNdx*gxq[iel,iq]+dNdy*gyq[iel,iq])*JxWq[iel,iq]*rhoq[iel,iq]
        #end for

        # apply boundary conditions
        for k1 in range(0,m_T):
            m1=icon_V[k1,iel]
            if top_nodes[m1]:
               Aref=A_el[k1,k1]
               A_el[k1,:]=0
               A_el[:,k1]=0
               A_el[k1,k1]=Aref
               b_el[k1]=0
            #end if
        #end for

        # assemble matrix K_mat and right hand side rhs
        for ikk in range(m_T):
            m1=icon_V[ikk,iel]
            for jkk in range(m_T):
                VV_T[counter]=A_el[ikk,jkk]
                counter+=1
            rhs[m1]+=b_el[ikk]
        #end for

    return VV_T,rhs

###################################################################################################
